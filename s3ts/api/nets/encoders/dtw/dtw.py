import torch

@torch.jit.script
def dtw_compute(dist_tensor: torch.Tensor, grad_tensor: torch.Tensor, w: float) -> None:
    for i in range(1, dist_tensor.shape[0]):
        for j in range(1, dist_tensor.shape[1]):
            value = torch.minimum(w * torch.minimum(dist_tensor[i, j-1], dist_tensor[i-1, j-1]), dist_tensor[i-1, j])
            id = (w * dist_tensor[i, j-1] < dist_tensor[i-1, j]) & (dist_tensor[i, j-1] < dist_tensor[i-1, j-1])

            dist_tensor[i, j] += value

            grad_tensor[id][:, :, i, j] += w * grad_tensor[id][:, :, i, j-1]

@torch.jit.script
def torch_dtw_fast(x: torch.Tensor, y: torch.Tensor, w: float, eps: float = 1e-5):
    # shape of x (n, dim, x_len) y (m, dim, y_len)

    # performs convolution-like operation, for each kernel the DF
    # (of shape (kernel_size, T)) is computed, then summed across channels
    # x has shape (batch, c, time_dimension)

    # compute pairwise diffs (squared)
    p_diff = x[:,None,:,None,:] - y[None,:,:,:,None] # shape (n, n_kernel, d, Kernel_size, T)
    euc_d = torch.square(p_diff).sum(2) # shape (n, n_kernel, kernel_size, T)

    # p_diff contains the partial derivatives of DTW[n, k, i, j] wrt K[k, d, i] (dims (n, k, d, i, j))
    p_diff = p_diff / torch.sqrt(euc_d[:,:, None, :, :] + eps)

    # compute dtw
    euc_d[:,:,0,:] = torch.cumsum(euc_d[:,:,0,:], dim=2)
    euc_d[:,:,:,0] = torch.cumsum(euc_d[:,:,:,0], dim=2)

    # rearrange dims
    DTW = torch.permute(euc_d, (2, 3, 0, 1)).contiguous()

    dtw_compute(DTW, p_diff, w)

    # recover dimensions
    DTW = torch.permute(DTW, (2, 3, 0, 1)).contiguous()

    return DTW.sqrt(), p_diff

class torch_dtw(torch.autograd.Function):

    @staticmethod
    def forward(x, y, w):
        DTW, p_diff = torch_dtw_fast(x, y, w)
        return DTW, p_diff
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        DTW, p_diff = output
        ctx.save_for_backward(p_diff)
    
    @staticmethod
    def backward(ctx, dtw_grad, p_diff_grad):
        p_diff, = ctx.saved_tensors
        mult = (p_diff * dtw_grad[:,:,None,:,:])
        return mult.mean(dim=(1, 3)), mult.mean(dim=(0, 4)), None