import torch
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

@torch.jit.script
def dtw_compute_all_script(dtw: torch.Tensor, dist_grad: torch.Tensor, grad: torch.Tensor, w: float) -> None:
    '''
        dtw of shape (n, k, pattern_len, window_size)
        dist_grad of shape (n, k, dims, pattern_len, window_size)
        grad of shape (n, k, dims, pattern_len)
        grads of shape (n, k, dims, pattern_len, pattern_len, window_size)
    '''
    n, k, len_pattern, len_window = dtw.shape
    # very big tensor
    # grads.zero_()
    # grads shape (pattern_len(2), window_size, n, k, dims, pattern_len)
    grads = torch.zeros(len_pattern, len_window, n, k, grad.shape[2], len_pattern, device=grad.device)
    temp = torch.cumsum(dist_grad, dim=4).permute(3, 4, 0, 1, 2) # (pattern_len, window_size, n, k, dim)

    for i in range(len_pattern):
        grads[i, :, :, :, :, i] = temp[i, :, :, :, :]
        grads[i:, 0, :, :, :, i] = temp[i, :1, :, :, :]

    for i in range(1, len_pattern): # pl
        for j in range(1, len_window): # ws
            value = torch.minimum(w * torch.minimum(dtw[:, :, i, j-1], dtw[:, :, i-1, j-1]), dtw[:, :, i-1, j])
            temp_1 = dtw[:, :, i, j-1] < dtw[:, :, i-1, j-1] # path (i, j-1) or (i-1, j)
            temp_2 = w * dtw[:, :, i, j-1] < dtw[:, :, i-1, j] # path (i, j-1) or (i-1, j-1)
            temp_3 = w * dtw[:, :, i-1, j-1] < dtw[:, :, i-1, j] # path (i-1, j-1) or (i-1, j)

            dtw[:, :, i, j] += value

            #print(temp_1.shape, grads[temp_1 & temp_2].shape)
            grads[i, j][temp_1 & temp_2] += w * grads[i, j-1][temp_1 & temp_2]
            grads[i, j][temp_1 & temp_3] += grads[i-1, j][temp_1 & temp_3]
            grads[i, j][temp_2 & temp_3] += w * grads[i-1, j-1][temp_2 & temp_3]

    grad += grads.sum(dim=(0, 1))

def dtw_compute_all(dtw: torch.Tensor, dist_grad: torch.Tensor, grad: torch.Tensor, w: float) -> None:
    dtw_compute_all_script(dtw, dist_grad, grad, w)

@torch.jit.script
def dtw_compute_no_grad(dtw: torch.Tensor, w: float) -> None:
    '''
        dtw of shape (n, k, pattern_len, window_size)
        grad of shape (n, k, dims, pattern_len)
    '''

    n, k, len_pattern, len_window = dtw.shape

    for i in range(1, len_pattern): # pl
        for j in range(1, len_window): # ws
            value = torch.minimum(w * torch.minimum(dtw[:, :, i, j-1], dtw[:, :, i-1, j-1]), dtw[:, :, i-1, j])

            dtw[:, :, i, j] += value

#@torch.jit.script
def dtw_fast_no_n(x: torch.Tensor, y: torch.Tensor, w: float, eps: float = 1e-5, compute_gradients: bool=True, batched: int = 8):
    # shape of x (n, dim, x_len) y (m, dim, y_len)

    # performs convolution-like operation, for each kernel the DF
    # (of shape (kernel_size, T)) is computed, then summed across channels
    # x has shape (batch, c, time_dimension)

    # compute pairwise diffs (squared)
    p_diff = x[:,None,:,None,:] - y[None,:,:,:,None] # shape (n, n_kernel, d, Kernel_size, T)
    euc_d = torch.square(p_diff).sum(2) # shape (n, n_kernel, kernel_size, T)

    # compute dtw
    euc_d[:,:,0,:] = torch.cumsum(euc_d[:,:,0,:], dim=2)
    euc_d[:,:,:,0] = torch.cumsum(euc_d[:,:,:,0], dim=2)

    if compute_gradients:
        # p_diff now contains the partial derivatives of DTW[n, k, i, j] wrt K[k, d, i] (dims (n, k, d, i, j))
        p_diff = p_diff / torch.sqrt(euc_d[:,:, None, :, :] + eps)
        
        grads = torch.zeros((x.shape[0], y.shape[0], y.shape[1], y.shape[2]), device=y.device) # dims (n, k, d, i)
        # grads_buffer = torch.empty((num_workers, y.shape[0], y.shape[1], y.shape[2], y.shape[2], x.shape[2]), device=y.device)

        grads.share_memory_()
        p_diff.share_memory_()
        euc_d.share_memory_()

        for i in range(0, x.shape[0], batched):
            initial = i
            last = min(initial + batched, x.shape[0])
            dtw_compute_all(euc_d[initial:last], p_diff[initial:last], grads[initial:last], w)

            # processes = [Process(target=dtw_compute_all, args=(euc_d[]))]
            # j = min(i+batched, x.shape[0])

            # dtw_compute_all(euc_d[i:j], p_diff[i:j], grads[i:j], w)
        
        return euc_d.sqrt(), grads
    else:
        dtw_compute_no_grad(euc_d, w)

        return euc_d.sqrt(), None