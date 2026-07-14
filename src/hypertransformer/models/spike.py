"""Surrogate gradient function for differentiable spiking neurons."""

import torch


class SurrogateSpike(torch.autograd.Function):
    """Binary spike with sigmoid surrogate gradient for backpropagation.

    Forward: threshold step function
    Backward: sigmoid surrogate gradient (smooth approximation)
    """

    @staticmethod
    def forward(ctx, surprise_scores, threshold):
        scale = 4.0
        ctx.save_for_backward(surprise_scores, threshold)
        ctx.scale = scale
        return (surprise_scores > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        surprise_scores, threshold = ctx.saved_tensors
        scale = ctx.scale
        sigmoid = torch.sigmoid((surprise_scores - threshold) * scale)
        grad_x = grad_output * sigmoid * (1 - sigmoid) * scale
        grad_threshold = (grad_output * (sigmoid - 0.5)).sum()
        return grad_x, grad_threshold
