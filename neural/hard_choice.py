'''
A non-differentiable random choice, where (expected) derivatives are computed with policy gradients.
'''

import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HardChoice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, outputs, probs) -> int:
        # outputs: length×N
        # probs: length, sum(probs) == 1.
        ctx.outputs_size = outputs.size()
        ctx.probs_size = probs.size()
        length, = probs.size() # there is one dimension: length
        ctx.idx = idx = np.random.choice(length, p=probs.cpu().numpy())
        result = outputs[idx]
        ctx.save_for_backward(result, probs)
        return result

    @staticmethod
    def backward(ctx, result_grad):
        idx = ctx.idx
        result, probs = ctx.saved_tensors
        outputs_grad = torch.zeros(ctx.outputs_size, device=device)
        outputs_grad[idx] = result_grad
        probs_grad = torch.zeros(ctx.probs_size, device=device)
        probs_grad[idx] = torch.inner(result_grad, result) / probs[idx]
        return outputs_grad, probs_grad


