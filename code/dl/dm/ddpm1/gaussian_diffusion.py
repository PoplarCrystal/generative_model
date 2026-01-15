"""
Gaussian Diffusion utilities for training and sampling diffusion models.

This code is inspired by improved-diffusion and denoising-diffusion-pytorch.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D tensor for a batch of indices.

    :param arr: the 1-D tensor.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    :param betas: a 1-D tensor of betas for each diffusion timestep.
    :param device: device to run on.
    """

    def __init__(self, betas, device='cpu'):
        self.device = device
        
        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = torch.from_numpy(betas).to(device)
        assert len(self.betas.shape) == 1, "betas must be 1-D"
        assert (self.betas > 0).all() and (self.betas <= 1).all()

        self.num_timesteps = int(self.betas.shape[0])

        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0], device=device)])
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (_extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) *
                noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (_extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                          _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] ==
                posterior_log_variance_clipped.shape[0] == x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """从预测的噪声恢复原始图像 x_0"""
        assert x_t.shape == eps.shape
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps)

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """从原始图像 x_0 计算噪声"""
        return ((_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) /
                _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def p_mean_variance(self, model, x, t, clip_denoised=True):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)
        
        # 模型预测噪声 (epsilon)
        model_output = model(x, t)
        assert model_output.shape == x.shape

        # 从预测的噪声恢复原始图像
        pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)
            # 如果裁剪了 x_start，重新计算噪声
            model_output = self._predict_eps_from_xstart(x, t, pred_xstart)

        # 计算后验分布的均值和方差
        model_mean, posterior_variance, posterior_log_variance = (
            self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        )

        assert (model_mean.shape == posterior_log_variance.shape == pred_xstart.shape == x.shape)
        return {
            "mean": model_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(self, model, x, t, clip_denoised=True):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_t.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, progress=False):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model, shape, noise=noise, clip_denoised=clip_denoised, progress=progress
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True, progress=False):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        device = self.device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(model, img, t, clip_denoised=clip_denoised)
                yield out
                img = out["sample"]

    def training_losses(self, model, x_start, t, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        model_output = model(x_t, t)
        # 目标值是噪声
        target = noise
        assert model_output.shape == target.shape == x_start.shape
        loss = F.mse_loss(model_output, target, reduction='mean')
        return {"loss": loss}

    def _reverse_to_img(self, x):
        """将张量转换为 PIL 图像"""
        x = x * 255
        x = x.clamp(0, 255).to(torch.uint8).cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def sample(self, model, shape, noise=None, clip_denoised=True, progress=True):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param progress: if True, show a tqdm progress bar.
        :return: a list of PIL images.
        """
        samples = self.p_sample_loop(model, shape, noise=noise, clip_denoised=clip_denoised, progress=progress)
        images = [self._reverse_to_img(samples[i]) for i in range(shape[0])]
        return images

    def sample_with_intermediate_steps(self, model, shape, step_interval, noise=None, clip_denoised=True):
        """
        Generate a sample and return intermediate images at specified step intervals.
        This ensures all intermediate images are from the same denoising process.
        
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param step_interval: interval between captured steps (e.g., num_timesteps/16).
        :param noise: if specified, the noise from the encoder to sample.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :return: a tuple (final_images, intermediate_images_list, step_labels)
                 where intermediate_images_list contains images at step_interval, 2*step_interval, ...
        """
        device = self.device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        
        # 计算要捕获的步数
        num_intermediates = int(self.num_timesteps / step_interval)
        steps_to_capture = [int(step_interval * i) for i in range(1, num_intermediates + 1)]
        
        intermediate_images = []
        step_labels = []
        current_step = 0
        
        indices = list(range(self.num_timesteps))[::-1]  # 从 num_timesteps-1 到 0
        
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(model, img, t, clip_denoised=clip_denoised)
                img = out["sample"]
                current_step += 1
                # 如果当前步数在要捕获的列表中，保存图像
                if current_step in steps_to_capture:
                    intermediate_images.append(self._reverse_to_img(img[0]))  # 只保存第一张图像作为代表
                    step_labels.append(f'{current_step} steps')
        # 生成最终图像
        final_images = [self._reverse_to_img(img[i]) for i in range(shape[0])]
        return final_images, intermediate_images, step_labels

