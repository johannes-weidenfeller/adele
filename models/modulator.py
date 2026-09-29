import torch
import torch.nn as nn
import models

@models.register('modulated-variance')
class ModulatedVariance(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mod_factor = 0.0
        # Configuration parameters
        self.modulate = config.get('modulate', False)
        self.modulate_start = config.get('modulate_start', 0)
        self.modulate_end = config.get('modulate_end', 0)
        self.max_val = config.get('max_val', 10000.0) # Standard max for inv_s
        self.scaling = config.get('scaling', False)
        self.scale_factor = config.get('scale_factor', 1.0)
        self.activation = config.get('activation', 'exp') # 'exp' or 'none' or 'sigmoid'
        self.offset = config.get('offset', 0.0)

    def update_step(self, epoch, global_step):
        if self.modulate:
            if global_step >= self.modulate_start and global_step < self.modulate_end:
                 self.mod_factor = (global_step - self.modulate_start) / (self.modulate_end - self.modulate_start)
            elif global_step >= self.modulate_end:
                 self.mod_factor = 1.0
            else:
                 self.mod_factor = 0.0

    def forward(self, val):

        val = val + self.offset
        
        # Apply activation
        if self.activation == 'exp':
            val = torch.exp(val)
        elif self.activation == 'sigmoid':
            val = torch.sigmoid(val)
        # else: 'none', keep as is

        # Apply modulation (lerp towards max_val)
        if self.modulate:
            val = val * (1.0 - self.mod_factor) + self.mod_factor * self.max_val
        
        # Optional scaling
        if self.scaling:
            val = val / self.scale_factor
            
        return val
