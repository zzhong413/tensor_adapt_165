import tensorly as tl
tl.set_backend('pytorch')
from torch.nn import functional as F
from tltorch.functional.convolution import general_conv1d


def cp_conv_adaptive(x, cp_tensor, adaptive_weights, bias=None, stride=1, padding=0, dilation=1):
    """Perform a factorized CP convolution

    Parameters
    ----------
    Parameters
    x : torch.tensor
        tensor of shape (batch_size, C, I_2, I_3, ..., I_N)
        
    cp_tensor : tltorch.CPTensor
        convolutional kernel in CP factorized form

    adaptive_weights : torch.tensor
        tensor of shape (batch_size, rank)
        specifies the weight of the linear combination of rank one tensors for each sample

    Returns
    -------
    NDConv(x) with an CP kernel
    """
    shape = cp_tensor.shape
    rank = cp_tensor.rank

    batch_size = x.shape[0]
    order = len(shape) - 2

    if isinstance(padding, int):
        padding = (padding, )*order
    if isinstance(stride, int):
        stride = (stride, )*order
    if isinstance(dilation, int):
        dilation = (dilation, )*order

    # Change the number of channels to the rank
    x_shape = list(x.shape)
    x = x.reshape((batch_size, x_shape[1], -1)).contiguous()

    # First conv == tensor contraction
    # from (in_channels, rank) to (rank == out_channels, in_channels, 1)
    x = F.conv1d(x, tl.transpose(cp_tensor.factors[1]).unsqueeze(2))

    x_shape[1] = rank
    x = x.reshape(x_shape)

    # convolve over non-channels
    for i in range(order):
        # From (kernel_size, rank) to (rank, 1, kernel_size)
        kernel = tl.transpose(cp_tensor.factors[i+2]).unsqueeze(1)             
        x = general_conv1d(x.contiguous(), kernel, i+2, stride=stride[i], padding=padding[i], groups=rank)

    # Revert back number of channels from rank to output_channels
    x_shape = list(x.shape)

    # # If adaptive_weights's shape smaller than batch_size, repeat it to have the same dimension
    # if adaptive_weights.shape[0] < batch_size:
    #     adaptive_weights = adaptive_weights.repeat(batch_size, 1)

    x = x.reshape((batch_size, rank, -1)) * adaptive_weights.reshape((batch_size, rank, 1))         
    # Last conv == tensor contraction
    # From (out_channels, rank) to (out_channels, in_channels == rank, 1)
    x = F.conv1d(x*cp_tensor.weights.unsqueeze(1).unsqueeze(0), cp_tensor.factors[0].unsqueeze(2), bias=bias)

    x_shape[1] = x.shape[1] # = out_channels
    x = x.reshape(x_shape)

    return x

