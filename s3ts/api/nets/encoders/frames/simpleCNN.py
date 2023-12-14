import torch

class SimpleCNN_IMG(torch.nn.Module):
    
    def __init__(self, channels=1, ref_size=32, 
            wdw_size=32, n_feature_maps=32):
        super().__init__()
        
        self.channels = channels
        self.ref_size = ref_size
        self.wdw_size = wdw_size
        self.n_feature_maps = n_feature_maps

        layer_input = [ref_size]
        layer_kernel_size = [0]
        layer_n_kernels = [channels, n_feature_maps]
        
        while layer_input[-1] > 3:
            layer_kernel_size.append(4 if layer_input[-1]%2 else 3)
            layer_input.append((layer_input[-1]-layer_kernel_size[-1]+1)//2)
            layer_n_kernels.append(layer_n_kernels[-1]*2)

        layer_kernel_size.append(layer_input[-1])
        layer_input.append(1)

        self.model = torch.nn.Sequential()

        for i in range(1, len(layer_input)):
            self.model.append(
                torch.nn.Conv2d(in_channels=layer_n_kernels[i-1], out_channels=layer_n_kernels[i], kernel_size=layer_kernel_size[i])
            )
            self.model.append(torch.nn.ReLU())

            if i!=(len(layer_input)-1):
                self.model.append(torch.nn.MaxPool2d(kernel_size=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_output_shape(self) -> torch.Size:
        x = torch.rand((1, self.channels, self.ref_size, self.wdw_size))
        print("Input shape: ", x.shape)
        x: torch.Tensor = self(x)
        print("Latent shape: ", x.shape)
        return x.shape
