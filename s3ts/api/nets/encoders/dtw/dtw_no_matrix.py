import torch

@torch.jit.script
def dtw_compute_full(dtw: torch.Tensor, dist_grad: torch.Tensor, w: float) -> torch.Tensor:
    '''
        dtw of shape (n, k, pattern_len, window_size)
        dist_grad of shape (n, k, dims, pattern_len, window_size)
        grad of shape (n, k, dims, pl)
    '''
    n, k, len_pattern, len_window = dtw.shape
    grads = torch.zeros((n, k, dist_grad.shape[2], len_pattern), device=dtw.device)

    for i in range(1, len_pattern): # pl
        for j in range(1, len_window): # ws
            value = torch.minimum(w * torch.minimum(dtw[:, :, i, j-1], dtw[:, :, i-1, j-1]), dtw[:, :, i-1, j])

            dtw[:, :, i, j] += value

    temp = dtw[:, :, -1, -1] # max value of dtw to find path
    pathfind = torch.empty((4, ) + temp.shape, device=dtw.device)

    # find path and compute gradients
    for i in range(len_pattern - 1, -1, -1): # pl
        for j in range(len_window - 1, -1, -1): # ws
            pathfind[0] = temp
            pathfind[1] = dtw[:, :, i-1, j]
            pathfind[2] = dtw[:, :, i-1, j-1]
            pathfind[3] = dtw[:, :, i, j-1]
            
            id = torch.argmin(pathfind, dim=0) # shape n, k (values 0, 1, 2, 3)
            temp = torch.gather(pathfind, 0, id.unsqueeze(0)).squeeze(0)

            grads[id!=0][:,:,i] = dist_grad[id!=0][:, :, i, j]

    return grads

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
def dtw_fast_no_image(x: torch.Tensor, y: torch.Tensor, w: float, eps: float = 1e-5, compute_gradients: bool=True):
    # shape of x (n, dim, x_len) y (m, dim, y_len)

    # performs convolution-like operation, for each kernel the DF
    # (of shape (kernel_size, T)) is computed, then summed across channels
    # x has shape (batch, c, time_dimension)

    # compute pairwise diffs (squared)
    p_diff = x[:,None,:,None,:] - y[None,:,:,:,None] # shape (n, n_kernel, d, Kernel_size, T)
    euc_d = torch.square(p_diff).sum(2) # shape (n, n_kernel, kernel_size, T)

    if compute_gradients:
        p_diff /= torch.sqrt(euc_d[:,:, None, :, :] + eps)

    # compute dtw
    euc_d[:,:,0,:] = torch.cumsum(euc_d[:,:,0,:], dim=2)
    euc_d[:,:,:,0] = torch.cumsum(euc_d[:,:,:,0], dim=2)

    if compute_gradients:
        # p_diff now contains the partial derivatives of DTW[n, k, i, j] wrt K[k, d, i] (dims (n, k, d, i, j))
        
        grads = dtw_compute_full(euc_d, p_diff, w) # dims (n, k, d, i, i, j)
        
        return euc_d.sqrt(), grads
    else:
        dtw_compute_no_grad(euc_d, w)

        return euc_d.sqrt(), None

class torch_dtw_no_image(torch.autograd.Function):

    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor, w: float):
        DTW, p_diff = dtw_fast_no_image(x, y, w, compute_gradients=y.requires_grad)
        return DTW[:, :, -1, -1], p_diff
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        DTW, p_diff = output
        ctx.save_for_backward(p_diff)
    
    @staticmethod
    def backward(ctx, dtw_grad, p_diff_grad):
        # dtw_grad dims (n, k) p_diff dims (n, k, d, pl)
        p_diff, = ctx.saved_tensors
        mult = (p_diff * dtw_grad[:, :, None, None]) # dims (n, k, d)
        return None, mult.mean(0), None # dims (n, d, k)