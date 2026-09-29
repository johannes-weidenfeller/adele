import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from . import register, make
from .utils import get_activation
from .network_utils import get_encoding, get_mlp
from systems.utils import update_module_step
from utils.misc import C
from .utils import generate_ide_fn
from .base import BaseModel


class BaseSDF(BaseModel):
    def setup(self):
        pass
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError
        
    def sample(self, num_samples = 1, *args, **kwargs):
        # Default sample calls forward and returns log-prob 0.0
        # Consistently return shape (num_samples, ...)
        val = self.forward(*args, **kwargs)
        val = val.unsqueeze(0).repeat(num_samples, *([1] * val.dim()))
        log_prob = torch.zeros_like(val)
        return val, log_prob


@register('simple-sdf')
class SimpleSDF(BaseSDF):
    def setup(self):
        self.scale = self.config.get('scale', 1.0)
        self.activation = self.config.get('activation', None)
    
    def forward(self, sdf, *args, **kwargs):
        # Extract first component
        sdf_val = sdf[..., 0]
        
        if self.activation == 'sigmoid':
            sdf_val = 2 * (torch.sigmoid(sdf_val) - 0.5)

        if self.scale != 1.0:
            sdf_val = sdf_val * self.scale
        return sdf_val


@register('additive-sdf')
class AdditiveSDF(BaseSDF):
    """Base + residual SDF: `sdf` is the coarse base (typically stored per-vertex
    on the tetrahedral grid), `sdf_residual` a field evaluated at the same points
    (typically a hashgrid feature) added on top. Gives the geometry a smooth,
    spatially-coherent correction that pure per-vertex storage lacks."""
    def setup(self):
        self.scale = self.config.get('scale', 1.0)
        # residual_scale accepts the usual schedule list
        # [start_step, start_value, end_value, end_step]; keeping it at 0 for the
        # first steps lets a seeded base field (e.g. a depth-TSDF init)
        # survive, since a freshly initialised hashgrid MLP emits O(1) values
        # that would otherwise swamp a base of magnitude ~trunc.
        self._residual_scale_cfg = self.config.get('residual_scale', 1.0)
        self.residual_scale = C(self._residual_scale_cfg, 0, 0)

    def update_step(self, epoch, global_step):
        self.residual_scale = C(self._residual_scale_cfg, global_step, epoch)

    def forward(self, sdf, sdf_residual=None, *args, **kwargs):
        sdf_val = sdf[..., 0]
        if sdf_residual is not None and self.residual_scale != 0.0:
            sdf_val = sdf_val + self.residual_scale * sdf_residual[..., 0]
        if self.scale != 1.0:
            sdf_val = sdf_val * self.scale
        return sdf_val


@register('gaussian-sdf')
class GaussianSDF(BaseSDF):
    def setup(self):
        self.scale = self.config.get('scale', 1.0)
        self.activation = self.config.get('activation', None)
        
    def forward(self, means, log_vars, *args, **kwargs):
        # Expected value is just the mean
        m = means
        if self.activation == 'sigmoid':
            m = 2 * (torch.sigmoid(m) - 0.5)
        return m * self.scale

    def sample(self, num_samples=1, means=None, log_vars=None, seed=None, **kwargs):
        """Reparameterized sample.

        Returns:
            sampled_sdf: (num_samples, V) reparameterized SDF realizations.
            log_prob_sign: (num_samples, V) per-vertex log-probability of the
                drawn sign under N(m_scaled, s_scaled^2). This is the quantity
                consumed by the REINFORCE topology-gradient term — NOT the
                Gaussian density. Specifically:
                    log P(sign(s_v) = eps | mu, sigma)
                        = log Phi(eps * m_scaled / s_scaled)
                with eps the realized sign (treated as constant). The score
                function gradient w.r.t. mu, log_var flows through this.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Apply the same activation/scale transform as forward(), so DMTet
        # sees the same SDF whether sampling or evaluating the mean.
        m = means
        if self.activation == 'sigmoid':
            m = 2 * (torch.sigmoid(means) - 0.5)
        m_scaled = m * self.scale
        s_scaled = torch.exp(0.5 * log_vars) * self.scale

        noise = torch.randn(num_samples, *means.shape, device=means.device, dtype=means.dtype)
        sampled_sdf = m_scaled + s_scaled * noise

        # Per-vertex sign log-probability under the sampled distribution.
        # The sign of the realization is treated as observed (detached); the
        # gradient flows through (m_scaled / s_scaled), and from there to mu
        # and log_var via autograd. The 'scale' factor cancels in the ratio,
        # which is the correct invariance: a global rescaling of the SDF does
        # not change the topology distribution.
        sign = torch.sign(sampled_sdf).detach()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        s_safe = s_scaled.clamp(min=1e-12)
        log_prob_sign = torch.special.log_ndtr(sign * m_scaled / s_safe)

        return sampled_sdf, log_prob_sign


@register('gaussian-mixture-sdf')
class GaussianMixtureSDF(BaseSDF):
    def setup(self):
        self.scale = self.config.get('scale', 1.0)
        
    def _get_params(self, means, log_vars, weights):
        # Apply activations and scaling
        w = F.softmax(weights, dim=-1)  # (..., num_components)
        m = means * self.scale          # (..., num_components)
        
        # log_var = log(var) -> std = exp(0.5 * log_var)
        # Scaling physical distance by S scales variance by S^2
        # log(S^2 * var) = log(S^2) + log(var) = 2*log(S) + log_var
        s = torch.exp(0.5 * log_vars) * self.scale
        
        return w, m, s

    def forward(self, means, log_vars, weights, *args, positions=None, **kwargs):
        w, m, s = self._get_params(means, log_vars, weights)
        
        # Expected value of GMM is sum(w_i * mu_i)
        expected_sdf = (w * m).sum(dim=-1)
        return expected_sdf
        
    def sample(self, num_samples=1, *args, means=None, log_vars=None, weights=None, positions=None, seed=None, **kwargs):
        w, m, s = self._get_params(means, log_vars, weights)
        
        if seed is not None:
            torch.manual_seed(seed)
            
        num_components = w.shape[-1]
        original_shape = w.shape[:-1]
        
        # Flatten for multinomial sampling
        # We want num_samples for each of the N points
        w_flat = w.view(-1, num_components)
        N_flat = w_flat.shape[0]
        
        # 1. Sample mixture component based on categorical distribution
        # Returns (N_flat, num_samples)
        component_indices = torch.multinomial(w_flat, num_samples, replacement=True) 
        
        # 2. Gather the chosen mean and std
        # Repeat means and std for gathering
        m_flat = m.view(-1, num_components)
        s_flat = s.view(-1, num_components)
        
        # chosen values will have shape (N_flat, num_samples)
        chosen_mean = torch.gather(m_flat, 1, component_indices)
        chosen_std = torch.gather(s_flat, 1, component_indices)
        chosen_weight = torch.gather(w_flat, 1, component_indices)
        
        # 3. Sample from normal
        noise = torch.randn_like(chosen_mean)
        sampled_sdf = chosen_mean + chosen_std * noise
        
        # 4. Compute log-prob: log(w) + log(Normal(sampled_sdf | mean, std))
        # log(Normal) = -0.5 * log(2*pi) - log(std) - 0.5 * ((x - mean)/std)^2
        log_w = torch.log(chosen_weight.clamp(min=1e-12))
        log_normal = -0.5 * math.log(2 * math.pi) - torch.log(chosen_std.clamp(min=1e-12)) - 0.5 * (noise ** 2)
        log_prob = log_w + log_normal

        # Reshape back to (num_samples, ...)
        # Currently sampled_sdf is (N_flat, num_samples)
        # We want (num_samples, *original_shape)
        sampled_sdf = sampled_sdf.transpose(0, 1).reshape(num_samples, *original_shape)
        log_prob = log_prob.transpose(0, 1).reshape(num_samples, *original_shape)
        
        return sampled_sdf, log_prob