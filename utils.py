import torch
from torch.optim import Optimizer
import numpy as np
import time

class MirrorDescent(Optimizer):
    def __init__(self, params, lr, mirror_map):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, mirror_map=mirror_map)
        super(MirrorDescent, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            mirror_map = group['mirror_map']
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad.data
                    p.data = mirror_map(p.data, grad, lr)
        return loss
    
def test_loop(mirror_map, x_init, func, noise_const = 0.02, max_iter=5000, thresholds=True, lr=1e-2):
    x = torch.nn.Parameter(x_init)
    optimizer = MirrorDescent([x], lr=lr, mirror_map=mirror_map)

    trajectory = [x.detach().clone()]
    losses = []
    tolerance = 1e-3
    start_time = time.time()

    for i in range(max_iter):
        optimizer.zero_grad()
        noise = torch.randn_like(x) * noise_const / (1 + i * 0.01)
        loss = func(x)
        losses.append(loss.item())
        loss = func(x + noise)
        loss.backward()
        optimizer.step()
        trajectory.append(x.detach().clone())
        grad_norm = torch.norm(x.grad.detach())
        if grad_norm < tolerance:
            print(f"Early stopping at step {i}, final point: {x.detach()}, grad norm: {grad_norm:.6f}")
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    if thresholds:
        thresholds = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        for t in thresholds:
            steps = next((i for i, l in enumerate(losses) if l < t), None)
            if steps is not None:
                print(f"Loss < {t} reached at iteration {steps}")
            else:
                print(f"Loss < {t} not reached")

    trajectory = torch.stack(trajectory).numpy()
    return trajectory, losses

    
def entropic_mirror_map(x, grad, lr):
    x_new = x * torch.exp(-lr * grad)
    return x_new / x_new.sum()

def barycentric_to_cartesian(x):
    v1 = np.array([0, 0])
    v2 = np.array([1, 0])
    v3 = np.array([0.5, np.sqrt(3)/2])
    return x[:,0,None]*v1 + x[:,1,None]*v2 + x[:,2,None]*v3