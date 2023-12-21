import torch

@torch.jit.script
def dtw_compute_no_n(dtw: torch.Tensor, dist_grad: torch.Tensor, grad: torch.Tensor, w: float) -> None:
    '''
        dtw of shape (k, pattern_len, window_size)
        dist_grad of shape (k, dims, pattern_len, window_size)
        grad of shape (k, dims, pattern_len)
    '''
    k, len_pattern, len_window = dtw.shape
    # very big tensor
    grads = torch.zeros(k, grad.shape[1], len_pattern, len_pattern, len_window) # shape (n, k, dims, pattern_len, pattern_len, window_size)

    for i in range(len_pattern):
        grads[:, :, i, i, :] = torch.cumsum(dist_grad[:, :, i, :], dim=2)
        grads[:, :, i, i:, 0] = grads[:, :, i, i, :1]

    for i in range(1, len_pattern): # pl
        for j in range(1, len_window): # ws
            value = torch.minimum(w * torch.minimum(dtw[:, i, j-1], dtw[:, i-1, j-1]), dtw[:, i-1, j])
            temp_1 = dtw[:, i, j-1] < dtw[:, i-1, j-1] # path (i, j-1) or (i-1, j)
            temp_2 = w * dtw[:, i, j-1] < dtw[:, i-1, j] # path (i, j-1) or (i-1, j-1)
            temp_3 = w * dtw[:, i-1, j-1] < dtw[:, i-1, j] # path (i-1, j-1) or (i-1, j)

            dtw[:, i, j] += value

            grads[temp_1 & temp_2][:, :i, i, j] += w * grads[temp_1 & temp_2][:, :i, i, j-1]
            grads[temp_1 & temp_3][:, :i, i, j] += grads[temp_1 & temp_3][:, :i, i-1, j]
            grads[temp_2 & temp_3][:, :i, i, j] += w * grads[temp_2 & temp_3][:, :i, i-1, j-1]

    grad[:,:,:] += grads.sum(dim=(-2, -1))

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

@torch.jit.script
def dtw_fast_no_n(x: torch.Tensor, y: torch.Tensor, w: float, eps: float = 1e-5, compute_gradients: bool=True):
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
        
        grads = torch.zeros((x.shape[0], y.shape[0], y.shape[1], y.shape[2])) # dims (n, k, d, i)

        futures = [torch.jit.fork(dtw_compute_no_n, euc_d[i], p_diff[i], grads[i], w) for i in range(x.shape[0])] 
        results = [torch.jit.wait(future) for future in futures]
        
        return euc_d.sqrt(), grads
    else:
        dtw_compute_no_grad(euc_d, w)

        return euc_d.sqrt(), None

class torch_dtw(torch.autograd.Function):

    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor, w: float):
        DTW, p_diff = dtw_fast_no_n(x, y, w, compute_gradients=y.requires_grad)
        return DTW, p_diff
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        DTW, p_diff = output
        ctx.save_for_backward(p_diff)
    
    @staticmethod
    def backward(ctx, dtw_grad, p_diff_grad):
        # dtw_grad dims (n, k, i, j) p_diff dims (n, k, d, i)
        p_diff, = ctx.saved_tensors
        mult = (p_diff[:, :, :, :, None] * dtw_grad[:, :, None, :, :]) # dims (n, k, d, i, j)
        return None, mult.mean(dim=(0, 4)), None