import tensorly as tl
tl.set_backend('pytorch')
import torch
import tltorch
from tensorly import testing
from ..self_adaptive_conv import SelfAdaptiveConv

def test_SelfAdaptiveConv():
    device='cpu'
    input_channels = 10
    output_channels = 11
    kernel_size = 3
    batch_size = 2
    size = 4
    order = 2
    rank = 5

    input_shape = (batch_size, input_channels) + (size, )*order
    kernel_shape = (output_channels, input_channels) + (kernel_size, )*order
    data = torch.randn(input_shape, dtype=torch.float32, device=device)
    cp_tensor = tltorch.CPTensor.new(kernel_shape, rank=rank).normal_()
    
    # Create a SAConv from the cp-tensor
    cut_point = 3
    weights, factors = cp_tensor.weights, cp_tensor.factors
    weights_before = weights[:cut_point]
    weights_after = weights[cut_point:]
    factors_before = [f[:, :cut_point] for f in factors]
    factors_after = [f[:, cut_point:] for f in factors]
    cp_before = tltorch.CPTensor(weights_before, factors_before)
    cp_after = tltorch.CPTensor(weights_after, factors_after)
    
    conv_before = fact_conv = SelfAdaptiveConv.from_factorization(cp_before)
    res_before = conv_before(data, tl.ones(batch_size, cut_point))

    conv_after = fact_conv = SelfAdaptiveConv.from_factorization(cp_after)
    res_after = conv_after(data, tl.ones(batch_size, rank - cut_point))
    
    # Same mixing factors (adaptive weights) for all samples
    fact_conv = SelfAdaptiveConv.from_factorization(cp_tensor)
    w_before = tl.zeros(batch_size, rank)
    w_before[:, :cut_point] = 1
    w_after = tl.zeros(batch_size, rank)
    w_after[:, cut_point:] = 1
    full_w = tl.ones(batch_size, rank)
    estimate_before = fact_conv(data, w_before)
    estimate_after = fact_conv(data, w_after)
    estimate_full = fact_conv(data, full_w)
    testing.assert_array_almost_equal(res_before, estimate_before)
    testing.assert_array_almost_equal(res_after, estimate_after)

    # Different values for different samples
    w_mixed = tl.zeros(batch_size, rank)
    w_mixed[0, :cut_point:] = 1
    w_mixed[1, cut_point:] = 1
    w_mixed.requires_grad = True
    estimate_mixed = fact_conv(data, w_mixed)
    tl.testing.assert_array_almost_equal(res_before[0], estimate_mixed[0])
    tl.testing.assert_array_almost_equal(res_after[1], estimate_mixed[1])
    
