import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch import distributions as pyd
import math
import time

numpy_to_torch_dtype_dict = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float64"): torch.float32,
    np.dtype("float32"): torch.float32,
    # np.dtype('float64')    : torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}

torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma, reduce=True):
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    if reduce:
        return kl.mean()
    else:
        return kl


def shape_whc_to_cwh(shape):
    if len(shape) == 3:
        return (shape[2], shape[0], shape[1])

    return shape


def shape_cwh_to_whc(shape):
    if len(shape) == 3:
        return (shape[1], shape[2], shape[0])

    return shape


def safe_filesystem_op(func, *args, **kwargs):
    """
    This is to prevent spurious crashes related to saving checkpoints or restoring from checkpoints in a Network
    Filesystem environment (i.e. NGC cloud or SLURM)
    """
    num_attempts = 5
    for attempt in range(num_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            print(
                f"Exception {exc} when trying to execute {func} with args:{args} and kwargs:{kwargs}..."
            )
            wait_sec = 2**attempt
            print(f"Waiting {wait_sec} before trying again...")
            time.sleep(wait_sec)

    raise RuntimeError(
        f"Could not execute {func}, give up after {num_attempts} attempts..."
    )


def safe_save(state, filename):
    return safe_filesystem_op(torch.save, state, filename)


def safe_load(filename):
    return safe_filesystem_op(torch.load, filename)


def save_checkpoint(filename, state):
    print("=> saving checkpoint '{}'".format(filename + ".pth"))
    safe_save(state, filename + ".pth")


def load_checkpoint(filename):
    print("=> loading checkpoint '{}'".format(filename))
    state = safe_load(filename)
    return state


def parameterized_truncated_normal(uniform, mu, sigma, a, b):
    normal = torch.distributions.normal.Normal(0, 1, validate_args=False)

    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    alpha_normal_cdf = normal.cdf(torch.from_numpy(np.array(alpha)))
    p = (
        alpha_normal_cdf
        + (normal.cdf(torch.from_numpy(np.array(beta))) - alpha_normal_cdf) * uniform
    )

    p = p.numpy()
    one = np.array(1, dtype=p.dtype)
    epsilon = np.array(np.finfo(p.dtype).eps, dtype=p.dtype)
    v = np.clip(2 * p - 1, -one + epsilon, one - epsilon)
    x = mu + sigma * np.sqrt(2) * torch.erfinv(torch.from_numpy(v))
    x = torch.clamp(x, a, b)

    return x


def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
    return parameterized_truncated_normal(uniform, mu, sigma, a, b)


def sample_truncated_normal(shape=(), mu=0.0, sigma=1.0, a=-2, b=2):
    return truncated_normal(
        torch.from_numpy(np.random.uniform(0, 1, shape)), mu, sigma, a, b
    )


def variance_scaling_initializer(tensor, mode="fan_in", scale=2.0):
    fan = torch.nn.init._calculate_correct_fan(tensor, mode)
    print(fan, scale)
    sigma = np.sqrt(scale / fan)
    with torch.no_grad():
        tensor[:] = sample_truncated_normal(tensor.size(), sigma=sigma)
        return tensor


def mean_list(val):
    return torch.mean(torch.stack(val))


class AverageMeter(nn.Module):
    def __init__(self, in_shape, max_size):
        super(AverageMeter, self).__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean.fill_(0)

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean.squeeze(0).cpu().numpy()


########################################################################


class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStd, self).__init__()
        print("RunningMeanStd: ", insize)
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0, 2, 3]
            if len(self.insize) == 2:
                self.axis = [0, 2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def _update_mean_var_count_from_moments(
        self, mean, var, count, batch_mean, batch_var, batch_count
    ):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False, mask=None):
        if self.training:
            if mask is not None:
                pass
                # mean, var = torch_ext.get_mean_std_with_masks(input, mask)
            else:
                mean = input.mean(self.axis)  # along channel axis
                var = input.var(self.axis)
            (
                self.running_mean,
                self.running_var,
                self.count,
            ) = self._update_mean_var_count_from_moments(
                self.running_mean,
                self.running_var,
                self.count,
                mean,
                var,
                input.size()[0],
            )

        # change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view(
                    [1, self.insize[0], 1, 1]
                ).expand_as(input)
                current_var = self.running_var.view(
                    [1, self.insize[0], 1, 1]
                ).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(
                    input
                )
                current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(
                    input
                )
            if len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(
                    input
                )
                current_var = self.running_var.view([1, self.insize[0]]).expand_as(
                    input
                )
        else:
            current_mean = self.running_mean
            current_var = self.running_var
        # get output

        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = (
                torch.sqrt(current_var.float() + self.epsilon) * y
                + current_mean.float()
            )
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(
                    current_var.float() + self.epsilon
                )
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self):
        return self.base_dist.entropy()
