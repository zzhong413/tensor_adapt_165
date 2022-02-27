import torch
from .functional import cp_conv_adaptive
from tltorch.factorized_layers import FactorizedConv


class SelfAdaptiveConv(FactorizedConv):
    """Self Adaptive Factorized Convolutions
    
    Parameters
    ----------
    in_channels : int
        Number of channels in the input image
    
    out_channels : int
        Number of channels produced by the convolution
    
    kernel_size : (int or tuple)
        Size of the convolving kernel
    
    order : int
        order of the convolution, e.g. 2 for Conv2d, 3 for Conv3d
        
    stride : int or tuple, default is 1
        Stride of the convolution
        
    padding : int, tuple or str, default is 0 
        Padding added to all sides of the input
        
    dilation : int or tuple, default is 1
        Spacing between kernel elements
        
    groups : int, default is 1
        Number of blocked connections from input channels to output channels

    bias : bool, default is True
        If ``True``, adds a learnable bias to the output
    """

    def __init__(self, in_channels, out_channels, kernel_size, order=None,
                 stride=1, padding=0, dilation=1, bias=False, has_bias=False, n_layers=1,
                 rank='same', fixed_rank_modes=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, order=order,
                         stride=stride, padding=padding, dilation=dilation, bias=bias,
                         has_bias=has_bias, n_layers=n_layers, factorization='cp',
                         rank=rank, implementation='factorized', fixed_rank_modes=fixed_rank_modes)

        self.weight().normal_()

    def forward(self, x, adaptive_weights, indices=0):
        """Performs a forward pass of the adaptive factorized convolution
        
        Parameters
        ----------
        x : torch.tensor
            input tensor of shape (batch_size, in_channels, height, width)

        adaptive_weights : torch.tensor
            of shape (batch_size, rank)
            weights that adapt the weights to each input sample
        """

        # # If adaptive_weights not specified, set them to ones (used during training)
        # if adaptive_weights == None:
        #     adaptive_weights = torch.ones(x.shape[0], self.rank).to(x)

        # Single layer parametrized
        if self.n_layers == 1:
            if indices == 0:
                return cp_conv_adaptive(x, self.weight(), adaptive_weights,
                                        bias=self.bias, stride=self.stride.tolist(),
                                        padding=self.padding.tolist(), dilation=self.dilation.tolist())
            else:
                raise ValueError(f'Only one convolution was parametrized (n_layers=1) but tried to access {indices}.')

        # Multiple layers parameterized
        if isinstance(self.n_layers, int):
            if not isinstance(indices, int):
                raise ValueError(f'Expected indices to be in int but got indices={indices}'
                                 f', but this conv was created with n_layers={self.n_layers}.')
        elif len(indices) != len(self.n_layers):
            raise ValueError(f'Got indices={indices}, but this conv was created with n_layers={self.n_layers}.')

        bias = self.bias[indices] if self.has_bias[indices] else None
        return cp_conv_adaptive(x, self.weight(indices), adaptive_weights, bias=bias,
                                stride=self.stride[indices].tolist(),
                                padding=self.padding[indices].tolist(),
                                dilation=self.dilation[indices].tolist())
