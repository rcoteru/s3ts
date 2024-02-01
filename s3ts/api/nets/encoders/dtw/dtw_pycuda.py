import numpy as np
import torch
import torch.cuda
from numba import cuda, jit, prange 

@cuda.jit
def dtw_forward(dtw, w):
    '''
        dtw of shape (n, k, pattern_len, window_size)
    '''
    n, k, len_pattern, len_window = dtw.shape

    x, y = cuda.grid(2)

    if x < n and y < k:
        for i in range(1, len_pattern): # pl
            for j in range(1, len_window): # ws
                value = min(w * min(dtw[x, y, i, j-1], dtw[x, y, i-1, j-1]), dtw[x, y, i-1, j])
                dtw[x, y, i, j] += value

@cuda.jit
def dtw_backward(dtw, dist_grad, grad):
    '''
        dtw of shape (n, k, pattern_len, window_size)
        dist_grad of shape (n, k, dims, pattern_len, window_size)
        grad of shape (n, k, dims, pl)
    '''
    n, k, d, len_pattern, len_window = dist_grad.shape

    x, y = cuda.grid(2)

    if x < n and y < k:
        for i0 in range(len_pattern-1, -1, -1):
            for j0 in range(len_window-1, -1, -1):

                # A = dtw[x, y, i0, j0-1]
                # B = dtw[x, y, i0-1, j0]
                # C = dtw[x, y, i0-1, j0-1]

                # path is A if (A<B) & (A<C) -> path is not A if (A>=B) | (A>=C)
                # path is B if (B<A) & (B<C) -> path is not B if (B>=A) | (B>=C)

                if dtw[x, y, i0, j0] != np.inf:

                    for l in range(d):
                        cuda.atomic.add(grad, (x, y, l, i0), dist_grad[x, y, l, i0, j0])      
              
                    if j0==0 or i0==0:
                        continue

                    if dtw[x, y, i0, j0-1] >= dtw[x, y, i0-1, j0] or dtw[x, y, i0, j0-1] >= dtw[x, y, i0-1, j0-1]: # path is not A
                        for j in range(j0):
                            dtw[x, y, i0, j] = np.inf
                    if dtw[x, y, i0-1, j0] >= dtw[x, y, i0, j0-1] or dtw[x, y, i0-1, j0] >= dtw[x, y, i0-1, j0-1]: # path is not B
                        for i in range(i0):
                            dtw[x, y, i, j0] = np.inf

# @torch.jit.script
def dtw_fast_cuda(x: torch.Tensor, y: torch.Tensor, w: float, eps: float = 1e-5, compute_gradients: bool=True):
    # shape of x (n, dim, x_len) y (m, dim, y_len)

    # performs convolution-like operation, for each kernel the DF
    # (of shape (kernel_size, T)) is computed, then summed across channels
    # x has shape (batch, c, time_dimension)

    # compute pairwise diffs (squared)
    p_diff = x[:,None,:,None,:] - y[None,:,:,:,None] # shape (n, n_kernel, d, Kernel_size, T)
    euc_d = torch.square(p_diff).sum(2).sqrt() # shape (n, n_kernel, kernel_size, T)

    if compute_gradients:
        p_diff /= torch.sqrt(euc_d[:,:, None, :, :] + eps)

    # compute dtw
    euc_d[:,:,0,:] = torch.cumsum(euc_d[:,:,0,:], dim=2)
    euc_d[:,:,:,0] = torch.cumsum(euc_d[:,:,:,0], dim=2)

    if compute_gradients:
        # p_diff now contains the partial derivatives of DTW[n, k, i, j] wrt K[k, d, i] (dims (n, k, d, i, j))
        
        grads = torch.zeros((x.shape[0],) + y.shape, device=x.device)
        
        dtw_forward[(16, 16), (16, 16)](cuda.as_cuda_array(euc_d), w) # dims (n, k, d, i, i, j)
        dtw_backward[(16, 16), (16, 16)](cuda.as_cuda_array(euc_d), cuda.as_cuda_array(p_diff), cuda.as_cuda_array(grads))
        
        return euc_d, grads
    else:
        dtw_forward[(16, 16), (16, 16)](cuda.as_cuda_array(euc_d), w)

        return euc_d, None
    
class torch_dtw_cuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, w: float):
        DTW, p_diff = dtw_fast_cuda(x, y.detach(), w, compute_gradients=y.requires_grad)

        ctx.save_for_backward(p_diff)

        return DTW[:, :, -1, -1]
    
    @staticmethod
    def backward(ctx, dtw_grad):
        # dtw_grad dims (n, k) p_diff dims (n, k, d, pl)
        p_diff, = ctx.saved_tensors
        mult = (p_diff * dtw_grad[:, :, None, None]) # dims (n, k, d)
        return None, mult.mean(0), None # dims (n, d, k)