# Copyright (c) Meta Platforms, Inc. and affiliates.
import optree
import torch
from functools import partial
from typing import Optional, Literal

from sam3d_objects.data.utils import tree_tensor_map


def linear_approximation_step(x_t, dt, velocity):
    # x_tp1 = x_t + velocity * dt
    x_tp1 = tree_tensor_map(lambda x, v: x + v * dt, x_t, velocity)
    return x_tp1


def gradient(output, x, create_graph: bool = False):
    tensors, pyspec = optree.tree_flatten(
        x, is_leaf=lambda x: isinstance(x, torch.Tensor)
    )
    grad_outputs = [torch.ones_like(output).detach() for _ in tensors]
    grads = torch.autograd.grad(
        output,
        tensors,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
    )
    return optree.tree_unflatten(pyspec, grads)


class ODESolver:
    def __init__(
        self, multi_conditioning_mode: Optional[Literal["round_robin", "average"]] = None
    ):
        self.multi_conditioning_mode = multi_conditioning_mode

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        raise NotImplementedError

    def solve_iter(self, dynamics_fn, x_init, times, *args, **kwargs):
        x_t = x_init
        for i, (t0, t1) in enumerate(zip(times[:-1], times[1:])):
            dt = t1 - t0

            # assert self.multi_conditioning_mode is not None, "multi_conditioning_mode must be set"
            if self.multi_conditioning_mode == "round_robin":
                # In round-robin sampling, we want to condition on the time step index modulo the batch size.
                batch_size = (
                    x_t["shape"].shape[0] if isinstance(x_t, dict) else x_t.shape[0]
                )

                # All the conditional args that start with that shape should have their data replaced
                # with the appropriate index.
                new_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor) and arg.shape[0] == batch_size:
                        index = i % batch_size
                        arg = arg[index : index + 1].expand_as(arg)
                    new_args.append(arg)
                args = tuple(new_args)

            x_t = self.step(dynamics_fn, x_t, t0, dt, *args, **kwargs)

            if self.multi_conditioning_mode == "average":
                if isinstance(x_t, dict) and "shape" in x_t:
                    # If there's a `shape` field in x_t, we replace all of its entries with the average
                    # along the batch dimension. This serves as a form of multi-image sampling.
                    shape = x_t["shape"]
                    x_t["shape"][:] = torch.mean(shape, dim=0, keepdim=True)
                else:
                    # Otherwise this is the SLat sampling, in which case x_t is a tensor and we similarly
                    # replace it with its average along the batch dimension.
                    x_t[:] = torch.mean(x_t, dim=0, keepdim=True)
            yield x_t, t0

    def solve(self, dynamics_fn, x_init, times, *args, **kwargs):
        for x_t, _ in self.solve_iter(dynamics_fn, x_init, times, *args, **kwargs):
            pass
        return x_t


# https://en.wikipedia.org/wiki/Euler_method
class Euler(ODESolver):
    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        velocity = dynamics_fn(x_t, t, *args, **kwargs)
        x_tp1 = linear_approximation_step(x_t, dt, velocity)
        return x_tp1


# https://arxiv.org/abs/2505.05470
class SDE(ODESolver):
    def __init__(self, **kwargs):
        self.sde_strength = kwargs.pop("sde_strength", 0.1)
        super().__init__(**kwargs)

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        velocity = dynamics_fn(x_t, t, *args, **kwargs)
        sigma = 1 - t
        var_t = sigma / (1 - torch.tensor(sigma).clamp(min=dt))
        std_dev_t = (
            torch.sqrt(variance) * self.sde_strength
        )  # self.sde_strength = alpha

        def compute_mean(x, v):
            drift_term = x * (std_dev_t**2 / (2 * sigma) * dt)
            velocity_term = v * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            return x + drift_term + velocity_term

        prev_sample_mean = tree_tensor_map(compute_mean, x_t, velocity)

        # Generate noise and compute final sample using tree_tensor_map
        def add_noise(mean_val):
            variance_noise = torch.randn_like(mean_val)
            return mean_val + std_dev_t * torch.sqrt(torch.tensor(dt)) * variance_noise

        prev_sample = tree_tensor_map(add_noise, prev_sample_mean)

        return prev_sample


# https://en.wikipedia.org/wiki/Midpoint_method
class Midpoint(ODESolver):
    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        half_dt = 0.5 * dt

        x_mid = Euler.step(self, dynamics_fn, x_t, t, half_dt, *args, **kwargs)

        velocity_mid = dynamics_fn(x_mid, t + half_dt, *args, **kwargs)
        x_tp1 = linear_approximation_step(x_t, dt, velocity_mid)
        return x_tp1


# https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
class RungeKutta4(ODESolver):

    def k1(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        return dynamics_fn(x_t, t, *args, **kwargs)

    def k2(self, dynamics_fn, x_t, t, dt, k1, *args, **kwargs):
        x_k1 = linear_approximation_step(x_t, dt * 0.5, k1)
        return dynamics_fn(x_k1, t + dt * 0.5, *args, **kwargs)

    def k3(self, dynamics_fn, x_t, t, dt, k2, *args, **kwargs):
        x_k2 = linear_approximation_step(x_t, dt * 0.5, k2)
        return dynamics_fn(x_k2, t + dt * 0.5, *args, **kwargs)

    def k4(self, dynamics_fn, x_t, t, dt, k3, *args, **kwargs):
        x_k3 = linear_approximation_step(x_t, dt, k3)
        return dynamics_fn(x_k3, t + dt, *args, **kwargs)

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        k1 = self.k1(dynamics_fn, x_t, t, dt, *args, **kwargs)
        k2 = self.k2(dynamics_fn, x_t, t, dt, k1, *args, **kwargs)
        k3 = self.k3(dynamics_fn, x_t, t, dt, k2, *args, **kwargs)
        k4 = self.k4(dynamics_fn, x_t, t, dt, k3, *args, **kwargs)

        def compute_velocity(k1, k2, k3, k4):
            return (k1 + 2 * k2 + 2 * k3 + k4) / 6

        velocity_k = tree_tensor_map(compute_velocity, k1, k2, k3, k4)
        x_tp1 = linear_approximation_step(x_t, dt, velocity_k)
        return x_tp1
