import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
from torchdiffeq import odeint
import random

from models.estimator import Decoder

# modified from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/flow_matching.py
class CFMDecoder(torch.nn.Module):
    def __init__(self, noise_channels, cond_channels, hidden_channels, out_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels, boundary):
        super().__init__()
        self.noise_channels = noise_channels
        self.cond_channels = cond_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.gin_channels = gin_channels
        
        self.eps = 1e-3
        self.delta = 1e-3
        self.num_segments=6
        self.alpha = 1e-5
        self.boundary = boundary
        self.x_to_boundary = 1 - self.boundary
        
        if self.boundary != 0:
            print('trainng consistency model')

        self.estimator = Decoder(noise_channels, cond_channels, hidden_channels, out_channels, filter_channels, p_dropout, n_layers, n_heads, kernel_size, gin_channels)

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, c=None, solver=None, cfg_kwargs=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            c (torch.Tensor, optional): speaker embedding
                shape: (batch_size, gin_channels)
            solver: see https://github.com/rtqichen/torchdiffeq for supported solvers
            cfg_kwargs: used for cfg inference

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        
        # cfg control
        if cfg_kwargs is None:
            estimator = functools.partial(self.estimator, mask=mask, mu=mu, c=c)
        else:
            estimator = functools.partial(self.cfg_wrapper, mask=mask, mu=mu, c=c, cfg_kwargs=cfg_kwargs)
            
        trajectory = odeint(estimator, z, t_span, method=solver, rtol=1e-5, atol=1e-5)
        return trajectory[-1]
    
    @torch.inference_mode()
    def forward_reflow_euler(self, mu, mask, n_timesteps, temperature=1.0, c=None, solver=None, cfg_kwargs=None):
        """The probability flow ODE sampler with simple Euler discretization.
        """
        # Initial sample
        x = torch.randn_like(mu) * temperature
        
        if cfg_kwargs is None:
            estimator = functools.partial(self.estimator, mask=mask, mu=mu, c=c)
        else:
            estimator = functools.partial(self.cfg_wrapper, mask=mask, mu=mu, c=c, cfg_kwargs=cfg_kwargs)
        
        ### Uniform
        dt = 1. / n_timesteps
        eps = self.eps # default: 1e-3
        
        for i in range(n_timesteps):  
            num_t = i / n_timesteps * (1 - eps) + eps
            t = torch.tensor([num_t], device=x.device)
            pred = estimator(t, x)

            x = x + pred * dt
        return x
    
    # cfg inference
    def cfg_wrapper(self, t, x, mask, mu, c, cfg_kwargs):
        fake_speaker = cfg_kwargs['fake_speaker'].repeat(x.size(0), 1)
        fake_content = cfg_kwargs['fake_content'].repeat(x.size(0), 1, x.size(-1))
        cfg_strength = cfg_kwargs['cfg_strength']
        
        cond_output = self.estimator(t, x, mask, mu, c)
        uncond_output = self.estimator(t, x, mask, fake_content, fake_speaker)
        
        output = uncond_output + cfg_strength * (cond_output - uncond_output)
        return output

    def compute_loss(self, x1, mask, mu, c):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            c (torch.Tensor, optional): speaker condition.

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        if random.random() < 0.1:
            self.boundary = 0
        else:
            self.boundary = 0.9
        z0 = torch.randn_like(x1)
        
        t = torch.rand([mu.size(0), 1, 1], device=mu.device, dtype=mu.dtype) * (1 - self.eps) + self.eps
        r = torch.clamp(t + self.delta, max=1.)
        
        xt = t * x1 + (1 - t) * z0
        xr = r * x1 + (1 - r) * z0
        
        segments = torch.linspace(0, 1, self.num_segments + 1, device=x1.device)
        seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1) # .clamp(min=1) prevents the inclusion of 0 in indices.
        segment_ends = segments[seg_indices] # 0.5 or 1, shape:[b, 1, 1] 
        boundary_segment = segment_ends - self.x_to_boundary
        
        x_at_segment_ends = segment_ends * x1 + (1 - segment_ends) * z0
        
        rng_state = torch.get_rng_state()
        rng_state_cuda = torch.cuda.get_rng_state()
        vt = self.estimator(t.squeeze(), xt, mask, mu, c)
        
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(rng_state_cuda) # Shared Dropout Mask
        with torch.no_grad():
            if self.boundary == 0: # when hyperparameter["boundary"] == 0, vr is not needed
                vr = None
            else:
                vr = self.estimator(r.squeeze(), xr, mask, mu, c)
                vr = torch.nan_to_num(vr)
                
        ft = f_euler(t, segment_ends, xt, vt)
        fr = threshold_based_f_euler(r, segment_ends, xr, vr, self.boundary, boundary_segment, x_at_segment_ends)
        
        loss_f = F.mse_loss(ft, fr, reduction="sum") / (torch.sum(mask) * ft.size(1))
        
        loss_v = self._masked_losses_v(vt, vr, self.boundary, boundary_segment, segment_ends, t, mask)
        loss = loss_f + self.alpha * loss_v
        return loss, None
    
    def _masked_losses_v(self, vt, vr, threshold, threshold_segment, segment_ends, t, mask):
      if threshold == 0:
        return 0
    
      less_than_threshold = t < threshold_segment # [b, 1, 1]
      
      far_from_segment_ends = (segment_ends - t) > 1.01 * self.delta
    #   far_from_segment_ends = far_from_segment_ends.unsqueeze(-1).unsqueeze(-1)
      
    #   losses_v = torch.square(vt - vr)
    #   losses_v = less_than_threshold * far_from_segment_ends * losses_v
    #   losses_v = self.reduce_op(losses_v.reshape(losses_v.shape[0], -1), dim=-1)
      
      losses_v = less_than_threshold * far_from_segment_ends * F.mse_loss(vt, vr, reduction="none")
      losses_v = losses_v.sum() / (torch.sum(mask) * vt.size(1))
      return losses_v

def f_euler(t_expand, segment_ends, xt, vt):
      return xt + (segment_ends - t_expand) * vt
  
def threshold_based_f_euler(t_expand, segment_ends, xt, vt, threshold, threshold_segment, x_at_segment_ends):
    if threshold == 0:
        return x_at_segment_ends
    
    less_than_threshold = t_expand < threshold_segment
    
    res = (
    less_than_threshold * f_euler(t_expand, segment_ends, xt, vt)
    + (~less_than_threshold) * x_at_segment_ends
    )
    return res

# modified from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/flow_matching.py
class CFMDecoder_otmf(torch.nn.Module):
    def __init__(self, noise_channels, cond_channels, hidden_channels, out_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels, n_segments):
        super().__init__()
        self.noise_channels = noise_channels
        self.cond_channels = cond_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.gin_channels = gin_channels
        self.n_segments = n_segments
        self.sigma_min = 1e-4

        self.estimator = Decoder(noise_channels, cond_channels, hidden_channels, out_channels, filter_channels, p_dropout, n_layers, n_heads, kernel_size, gin_channels)

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, c=None, solver=None, cfg_kwargs=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            c (torch.Tensor, optional): shape: (batch_size, gin_channels)

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        
        # prepare 
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        
        # cfg control
        if cfg_kwargs is None:
            estimator = functools.partial(self.estimator, mask=mask, mu=mu, c=c)
        else:
            estimator = functools.partial(self.cfg_wrapper, mask=mask, mu=mu, c=c, cfg_kwargs=cfg_kwargs)
            
        trajectory = odeint(estimator, z, t_span, method=solver, rtol=1e-5, atol=1e-5)
        return trajectory[-1]
    
    # cfg inference
    def cfg_wrapper(self, t, x, mask, mu, c, cfg_kwargs):
        fake_speaker = cfg_kwargs['fake_speaker'].repeat(x.size(0), 1)
        fake_content = cfg_kwargs['fake_content'].repeat(x.size(0), 1, x.size(-1))
        cfg_strength = cfg_kwargs['cfg_strength']
        
        cond_output = self.estimator(t, x, mask, mu, c)
        uncond_output = self.estimator(t, x, mask, fake_content, fake_speaker)
        
        output = uncond_output + cfg_strength * (cond_output - uncond_output)
        return output

    def compute_loss(self, x1, mask, mu, c):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            c (torch.Tensor, optional): speaker condition.

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        t = 1 - torch.cos(t * 0.5 * torch.pi)
        # t = torch.sigmoid(torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype))
        
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(self.estimator(t.squeeze(), y, mask, mu, c), u, reduction="sum") / (torch.sum(mask) * u.size(1))
        return loss, y