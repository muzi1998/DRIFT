"""Standard FP32 AdamW with moment states kept on CPU.

Parameters and gradients remain on their original device. Each update transfers
at most ``chunk_elements`` entries of each moment to that device, applies the
same scalar AdamW equations, and copies updated moments back to CPU. This trades
PCIe bandwidth for device memory without quantizing or changing the optimizer.
"""
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class CPUOffloadedAdamW(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.98), eps=1e-8,
                 weight_decay=0.01, chunk_elements=4194304):
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError('lr, eps and weight_decay must be nonnegative')
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError('AdamW betas must be in [0,1)')
        if int(chunk_elements) < 1:
            raise ValueError('chunk_elements must be positive')
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                       weight_decay=weight_decay,
                                       chunk_elements=int(chunk_elements)))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for parameter in group['params']:
                grad = parameter.grad
                if grad is None:
                    # Standard AdamW neither advances this parameter's step nor
                    # applies decoupled decay when its gradient is absent.
                    continue
                if grad.is_sparse:
                    raise RuntimeError('CPUOffloadedAdamW requires dense gradients')
                if parameter.dtype != torch.float32 or grad.dtype != torch.float32:
                    raise TypeError('This optimizer is intentionally FP32 only')
                if not parameter.is_contiguous() or not grad.is_contiguous():
                    raise ValueError('CPUOffloadedAdamW requires contiguous tensors')
                state = self.state[parameter]
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros(parameter.shape, dtype=torch.float32, device='cpu')
                    state['exp_avg_sq'] = torch.zeros(parameter.shape, dtype=torch.float32, device='cpu')
                if state['exp_avg'].device.type != 'cpu' or state['exp_avg_sq'].device.type != 'cpu':
                    raise RuntimeError('AdamW moment states must remain on CPU')
                state['step'] += 1
                step = state['step']
                step_size = group['lr']/(1-beta1**step)
                correction2_sqrt = math.sqrt(1-beta2**step)
                flat_parameter = parameter.view(-1)
                flat_grad = grad.view(-1)
                flat_avg = state['exp_avg'].view(-1)
                flat_sq = state['exp_avg_sq'].view(-1)
                for start in range(0, parameter.numel(), group['chunk_elements']):
                    stop = min(start+group['chunk_elements'], parameter.numel())
                    p = flat_parameter[start:stop]
                    g = flat_grad[start:stop]
                    avg_cpu = flat_avg[start:stop]
                    sq_cpu = flat_sq[start:stop]
                    avg = avg_cpu.to(parameter.device, non_blocking=False)
                    sq = sq_cpu.to(parameter.device, non_blocking=False)
                    p.mul_(1-group['lr']*group['weight_decay'])
                    avg.lerp_(g, 1-beta1)
                    sq.mul_(beta2).addcmul_(g, g, value=1-beta2)
                    denominator = sq.sqrt().div_(correction2_sqrt).add_(group['eps'])
                    p.addcdiv_(avg, denominator, value=-step_size)
                    avg_cpu.copy_(avg, non_blocking=False)
                    sq_cpu.copy_(sq, non_blocking=False)
                    # Do not let a previous chunk coexist with the next .to().
                    del avg, sq, denominator
        return loss

    def state_memory_bytes(self):
        return sum(value.numel()*value.element_size()
                   for state in self.state.values() for value in state.values()
                   if torch.is_tensor(value))
