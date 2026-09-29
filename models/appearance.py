import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.utils import get_activation
from models.network_utils import get_encoding, get_mlp
from systems.utils import update_module_step
from models.utils import generate_ide_fn
from render import light
from models.base import BaseModel
import numpy as np
import radfoam
from utils import *
from utils.misc import get_rank
from plyfile import PlyData, PlyElement
import tqdm
from scipy.spatial import cKDTree
import open3d as o3d



@models.register('pbr')
class PBR(BaseModel):
    def setup(self):
        self.learn_light = self.config.get('learn_light', False)
        self.kd_min = self.config.get('kd_min', [0.0, 0.0, 0.0, 0.0])
        self.kd_max = self.config.get('kd_max', [1.0, 1.0, 1.0, 1.0])
        self.ks_min = self.config.get('ks_min', [0.0, 0.25, 0.0])
        self.ks_max = self.config.get('ks_max', [1.0, 1.0, 1.0])
        self.nrm_min = self.config.get('nrm_min', [-1.0, -1.0, 0.0])
        self.nrm_max = self.config.get('nrm_max', [1.0, 1.0, 1.0])
        self.camera_space_light = self.config.get('camera_space_light', False)
        if self.learn_light:
            self.lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=0.5)
        else:
            self.lgt = light.load_env(self.config.environment_map, scale=2.0)
        self.lgt.build_mips()

    def update_step(self, epoch, global_step):
        if self.learn_light:
            with torch.no_grad():
                self.lgt.clamp_(min = 0.0)

    def forward(self, dirs=None, normals=None, positions=None, sdf=None, **kwargs):
        # Match the radtets call interface (features passed as keyword `appearance_params`,
        # like volume-radiance). kd/ks are read directly from the appearance feature field.
        features = kwargs.get('appearance_params')
        kd = torch.sigmoid(features[..., :3])
        ks = torch.sigmoid(features[..., 3:6])
        if self.learn_light:
            self.lgt.build_mips()
        # radtets passes `dirs` = camera->surface (tdirs). shade() computes
        # wo = normalize(view_pos - pos); using view_pos = pos - dirs yields wo = -dirs
        # = the correct surface->camera outgoing direction.
        view_pos = positions - dirs
        # nvdiffrec's shade() expects nvdiffrast image buffers [N,H,W,C], but radtets
        # buffers carry arbitrary leading dims (samples/views/H/W). Flatten to a single
        # [1,1,P,3] "image" for the per-pixel cube-map lookups, then restore the shape.
        lead = positions.shape[:-1]
        P = 1
        for d in lead:
            P *= int(d)
        if P == 0:                       # empty ray chunk (nvdiffrast rejects 0-sized uv)
            return positions.new_zeros(*lead, 3)
        r = lambda x: x.reshape(1, 1, P, x.shape[-1])
        colors = self.lgt.shade(r(positions), r(normals), r(kd), r(ks), r(view_pos), specular=True)
        return colors.reshape(*lead, colors.shape[-1])

@models.register('volume-radiance')
class VolumeRadiance(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3
        self.shading_model = self.config.get('shading_model', 'view_dir')
        self.predicted_tangents = self.config.get('predicted_tangents', False)        
        self.input_normals = self.config.get('input_normals', True)
        self.input_dirs = self.config.get('input_dirs', True)
        self.input_refl_dirs = self.config.get('input_refl_dirs', False)
        self.encode_normals = self.config.get('encode_normals', False)
        self.encode_dirs = self.config.get('encode_dirs', True)
        self.encode_refl_dirs = self.config.get('encode_refl_dirs', False)
        self.n_input_dims = self.config.input_feature_dim

        self.encode_features = self.config.get('encode_features', False)
        if self.encode_features:
            self.feature_mlp = get_mlp(self.n_input_dims, self.n_input_dims, self.config.feature_mlp_config)
            self.n_input_dims += self.n_input_dims

        

        if self.shading_model == 'ref_nerf':
            spherical_harmonics_degree = self.config.get('spherical_harmonics_degree', 3)
            output_dim = 2* (2 ** spherical_harmonics_degree - 1 + spherical_harmonics_degree)
            dir_encoding = generate_ide_fn(self.config.get('spherical_harmonics_degree', 3))
            self.dir_encoding = dir_encoding
            self.n_input_dims += -5 + 1 + output_dim  # -5 for roughness and specular and diffuse rgb, +1 for normal dir cosine angle, +output_dim for reflected view dir encoding
        else:
            if self.input_dirs:
                if self.encode_dirs:
                    dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
                    self.dir_encoding = dir_encoding
                    self.n_input_dims += dir_encoding.n_output_dims
                else:
                    self.n_input_dims += 3

            if self.input_normals:
                if self.encode_normals:
                    normal_encoding = get_encoding(3, self.config.normal_encoding_config)
                    self.normal_encoding = normal_encoding
                    self.n_input_dims += normal_encoding.n_output_dims
                else:
                    self.n_input_dims += 3
            
            if self.input_refl_dirs:
                if self.encode_refl_dirs:
                    refl_dir_encoding = get_encoding(self.n_dir_dims, self.config.refl_dir_encoding_config)
                    self.refl_dir_encoding = refl_dir_encoding
                    self.n_input_dims += refl_dir_encoding.n_output_dims
                else:
                    self.n_input_dims += 3


        # if self.shading_model == 'local_normal_system':
        #     dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        #     normal_encoding = get_encoding(3, self.config.normal_encoding_config)
        #     self.n_input_dims = self.config.input_feature_dim + dir_encoding.n_output_dims + normal_encoding.n_output_dims
        # else:
        #     dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        #     self.n_input_dims = self.config.input_feature_dim + dir_encoding.n_output_dims
        
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    
        self.network = network


    def forward(self, dirs=None, normals=None, positions=None, sdf=None, **kwargs):
        features = kwargs.get('appearance_params')
        if sdf is not None:
            if sdf.dim() == features.dim() - 1:
                sdf = sdf.unsqueeze(-1)
            features = torch.cat([sdf, features], dim=-1)

        if self.encode_features:
            trasnformed_features = self.feature_mlp(features)
            features = torch.cat([features, trasnformed_features], dim=-1)
            
        if self.shading_model == 'local_normal_system':
            if self.predicted_tangents:
                tangent_dirs  = features[..., -3:]
                #Project tangent_dirs to tangent plane using hte normals and then normalize
                tangent_dirs = tangent_dirs - (tangent_dirs * normals).sum(dim=-1, keepdim=True) * normals
                tangent_dirs = tangent_dirs / (torch.norm(tangent_dirs, dim=-1, keepdim=True) + 1e-8)
                bitangent_dirs = torch.cross(normals, tangent_dirs, dim=-1)
                bitangent_dirs = bitangent_dirs / (torch.norm(bitangent_dirs, dim=-1, keepdim=True) + 1e-8)
            else:
                sign = torch.where(normals[..., 2] >= 0, 1.0, -1.0)
                a = -1.0 / (sign + normals[..., 2])
                b = normals[..., 0] * normals[..., 1] * a
                tangent_dirs = torch.stack([
                    1.0 + sign * normals[..., 0] * normals[..., 0] * a,
                    sign * b,
                    -sign * normals[..., 0]
                ], dim=-1)
                bitangent_dirs = torch.stack([
                    b,
                    sign + normals[..., 1] * normals[..., 1] * a,
                    -normals[..., 1]
                ], dim=-1)
                bitangent_dirs = bitangent_dirs / (torch.norm(bitangent_dirs, dim=-1, keepdim=True) + 1e-8)
                tangent_dirs = tangent_dirs / (torch.norm(tangent_dirs, dim=-1, keepdim=True) + 1e-8)


                # up1 = torch.tensor([0.0, 1.0, 0.0], device=normals.device)
                # up2 = torch.tensor([1.0, 0.0, 0.0], device=normals.device)
                # dot = torch.abs(normals @ up1)
                # use_up2 = dot > 0.99
                # up = torch.where(use_up2.unsqueeze(-1), up2, up1)
                # tangent_dirs = torch.cross(up, normals, dim=-1)

            # tangent_dirs = tangent_dirs / (torch.norm(tangent_dirs, dim=-1, keepdim=True) + 1e-8)

  
            # bitangent_dirs = torch.cross(normals, tangent_dirs, dim=-1)
            # bitangent_dirs = bitangent_dirs / (torch.norm(bitangent_dirs, dim=-1, keepdim=True) + 1e-8)
            # Construct a local frame
            R = torch.stack([tangent_dirs, bitangent_dirs, normals], dim=-1)  # [..., 3, 3]
            dirs= torch.matmul(R.transpose(-1, -2), dirs[..., None]).squeeze(-1)

        elif self.shading_model == 'reflected_view_dir':
            dirs = dirs - 2 * (dirs * normals).sum(dim=-1, keepdim=True) * normals

        elif self.shading_model == 'ref_nerf':
            diffuse_color = features[...,-3:]
            
            if 'color_activation' in self.config:
                diffuse_color = get_activation(self.config.color_activation)(diffuse_color)

            specular =  torch.sigmoid(features[...,-4:-3])
            roughness = F.softplus(features[...,-5:-4])
            reflected_dirs = dirs - 2 * (dirs * normals).sum(dim=-1, keepdim=True) * normals
            normal_direction_cosine_angle = torch.clamp((normals * dirs).sum(dim=-1, keepdim=True), -1.0, 1.0)
            dirs_embd = self.dir_encoding(reflected_dirs.reshape(-1, 3), roughness.reshape(-1,1))
            network_inp = torch.cat( [features[...,:-5].reshape(-1, features.shape[-1] - 5), dirs_embd, normal_direction_cosine_angle.reshape(-1,1)] + [arg.reshape(-1, arg.shape[-1]) for arg in args[1:]], dim=-1)
            specular_color = self.network(network_inp).reshape(*features.shape[:-1], self.n_output_dims).float()
            if 'color_activation' in self.config:
                specular_color = get_activation(self.config.color_activation)(specular_color)

            color = diffuse_color * (1 - specular) + specular_color * specular
            color = torch.clamp(color, 0.0, 1.0)
            return color


        refl_dirs = dirs - 2 * (dirs * normals).sum(dim=-1, keepdim=True) * normals
        # ----- Direction embedding -----
        if self.input_dirs:
            if self.encode_dirs:
                dirs_proc = (dirs + 1.) / 2.  # normalize (-1,1)→(0,1)
                dirs_embd = self.dir_encoding(dirs_proc.reshape(-1, self.n_dir_dims))
            else:
                dirs_embd = dirs.reshape(-1, 3)
        else:
            dirs_embd = None

        # ----- Normal embedding -----
        if self.input_normals:
            if self.encode_normals:
                normals_proc = (normals + 1.) / 2.
                normal_embd = self.normal_encoding(normals_proc.reshape(-1, 3))
            else:
                normal_embd = normals.reshape(-1, 3)
        else:
            normal_embd = None

        # ----- Reflected direction embedding -----
        if self.input_refl_dirs:
            if self.encode_refl_dirs:
                refl_dirs_proc = (refl_dirs + 1.) / 2.
                refl_dirs_embd = self.refl_dir_encoding(refl_dirs_proc.reshape(-1, self.n_dir_dims))
            else:
                refl_dirs_embd = refl_dirs.reshape(-1, 3)
        else:
            refl_dirs_embd = None
        
        # Start with features
        embed_list = [dirs_embd, normal_embd, refl_dirs_embd]
        network_inputs = [features.reshape(-1, features.shape[-1])]
        network_inputs += [emb for emb in embed_list if emb is not None]
        network_inp = torch.cat(network_inputs, dim=-1)
        # far background samples (unbounded marching) can hit smooth-SDF regions
        # where grad->0 gives NaN normals; encoding those NaNs crashes the fused
        # MLP kernel with an illegal memory access. Sanitise the network input.
        network_inp = torch.nan_to_num(network_inp, nan=0.0, posinf=0.0, neginf=0.0)
        color = self.network(network_inp).reshape(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def regularizations(self, out):
        return {}


@models.register('volume-color')
class VolumeColor(nn.Module):
    def __init__(self, config):
        super(VolumeColor, self).__init__()
        self.config = config
        self.n_output_dims = 3
        self.n_input_dims = self.config.input_feature_dim
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.network = network
    
    def forward(self, features=None, sdf=None, **kwargs):
        if features is None:
            features = kwargs.get('appearance_params')
        assert features is not None, "appearance_params must be provided to VolumeColor"
        if sdf is not None:
            if sdf.dim() == features.dim() - 1:
                sdf = sdf.unsqueeze(-1)
            features = torch.cat([sdf, features], dim=-1)
        network_inp = features.reshape(-1, features.shape[-1])
        color = self.network(network_inp).reshape(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def regularizations(self, out):
        return {}
    

def _positional_encoding(x, freqs):
    """sin/cos positional encoding with powers-of-two frequencies (IMLS-style)."""
    bands = (2.0 ** torch.arange(freqs, device=x.device, dtype=x.dtype))
    pts = (x[..., None] * bands).reshape(*x.shape[:-1], freqs * x.shape[-1])
    return torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)


@models.register('imls-splatting')
class IMLSSplatting(nn.Module):
    """Reflectance model from IMLS-Splatting (sin/cos PE variant).

    Pipeline:
        spatial_mlp(feature [+PE])         -> (diffuse[3], tint[3], bottleneck[B])
        bottleneck += sin(bottleneck * f)  if btn_freq enabled
        directional_mlp(bottleneck, n.v, viewdir [+PE], refdir [+PE]) -> specular[3]
        rgb = sigmoid(diffuse + tint * specular)

    Config:
        view_pe: -1 disables viewdir/refdir input; 0 = raw only; >0 = raw + sin/cos PE.
        fea_pe:  0 disables feature PE; >0 appends sin/cos PE to spatial MLP input.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_output_dims = 3

        self.n_input_dims = self.config.input_feature_dim
        self.bottleneck_dim = self.config.get('bottleneck_dim', 128)
        self.view_pe = self.config.get('view_pe', 3)
        self.fea_pe = self.config.get('fea_pe', 0)

        btn_freq = self.config.get('btn_freq', None)
        if btn_freq is not None and len(btn_freq) >= 2:
            f = torch.linspace(np.log2(btn_freq[0]), np.log2(btn_freq[1]),
                               self.bottleneck_dim)
            self.register_buffer('btn_freq', torch.exp2(f))
        else:
            self.btn_freq = None

        spatial_in_dim = self.n_input_dims
        if self.fea_pe > 0:
            spatial_in_dim += self.n_input_dims * self.fea_pe * 2
        spatial_out_dim = 3 + 3 + self.bottleneck_dim
        self.spatial_mlp = get_mlp(spatial_in_dim, spatial_out_dim,
                                   self.config.spatial_mlp_config)

        dir_in_dim = self.bottleneck_dim + 1  # bottleneck + n.v
        if self.view_pe > -1:
            dir_in_dim += 6                   # raw viewdir + raw refdir
        if self.view_pe > 0:
            dir_in_dim += 6 * self.view_pe * 2  # PE on viewdir + refdir
        self.directional_mlp = get_mlp(dir_in_dim, 3,
                                       self.config.directional_mlp_config)

    def forward(self, dirs=None, normals=None, positions=None, sdf=None, **kwargs):
        features = kwargs.get('appearance_params')
        if sdf is not None:
            if sdf.dim() == features.dim() - 1:
                sdf = sdf.unsqueeze(-1)
            features = torch.cat([sdf, features], dim=-1)

        feat_flat = features.reshape(-1, features.shape[-1])
        spa_in = [feat_flat]
        if self.fea_pe > 0:
            spa_in.append(_positional_encoding(feat_flat, self.fea_pe))
        spatial_out = self.spatial_mlp(torch.cat(spa_in, dim=-1)).float()
        diffuse, tint, bottleneck = torch.split(
            spatial_out, [3, 3, self.bottleneck_dim], dim=-1)

        if self.btn_freq is not None:
            bottleneck = bottleneck + torch.sin(bottleneck * self.btn_freq[None])

        dirs_flat = dirs.reshape(-1, 3)
        normals_flat = normals.reshape(-1, 3)
        # IMLS reference: refdir = reflect(-viewdir, normal); cos = (-viewdir).normal
        out_dir = -dirs_flat
        refl = 2.0 * (out_dir * normals_flat).sum(-1, keepdim=True) * normals_flat - out_dir
        n_dot_v = (out_dir * normals_flat).sum(-1, keepdim=True)

        dir_in = [bottleneck, n_dot_v]
        if self.view_pe > -1:
            dir_in += [dirs_flat, refl]
        if self.view_pe > 0:
            dir_in += [_positional_encoding(dirs_flat, self.view_pe),
                       _positional_encoding(refl, self.view_pe)]
        specular = self.directional_mlp(torch.cat(dir_in, dim=-1)).float()

        rgb = torch.sigmoid(diffuse + tint * specular)
        return rgb.reshape(*features.shape[:-1], 3)

    def regularizations(self, out):
        return {}


@models.register('uni-sdf')
class UniSDF(nn.Module):
    def __init__(self, config):
        super(UniSDF, self).__init__()
        self.config = config
        self.predict_weight = self.config.get('predict_weight', True)
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3
        self.encode_normals = self.config.get('encode_normals', False)
        self.encode_dirs = self.config.get('encode_dirs', False)
        self.encode_refl_dirs = self.config.get('encode_refl_dirs', False)
        self.encode_positions = self.config.get('encode_positions', False)

        self.n_input_dims_w = self.config.input_feature_dim
        self.n_input_dims_cam = self.config.input_feature_dim
        self.n_input_dims_ref = self.config.input_feature_dim

        if self.encode_dirs:
            dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
            self.dir_encoding = dir_encoding
            self.n_input_dims_cam += dir_encoding.n_output_dims
        else:
            self.n_input_dims_cam += 3

        if self.encode_refl_dirs:
            refl_dir_encoding = get_encoding(self.n_dir_dims, self.config.refl_dir_encoding_config)
            self.refl_dir_encoding = refl_dir_encoding
            self.n_input_dims_ref += refl_dir_encoding.n_output_dims
        else:
            self.n_input_dims_ref += 3
        
        if self.encode_normals:
            normal_encoding = get_encoding(3, self.config.normal_encoding_config)
            self.normal_encoding = normal_encoding
            self.n_input_dims_cam += normal_encoding.n_output_dims
            self.n_input_dims_ref += normal_encoding.n_output_dims
            self.n_input_dims_w += normal_encoding.n_output_dims
        else:
            self.n_input_dims_cam += 3
            self.n_input_dims_ref += 3
            self.n_input_dims_w += 3

        if self.encode_positions:
            position_encoding = get_encoding(3, self.config.position_encoding_config)
            self.position_encoding = position_encoding
            self.n_input_dims_cam += position_encoding.n_output_dims
            self.n_input_dims_ref += position_encoding.n_output_dims
            self.n_input_dims_w += position_encoding.n_output_dims
        else:
            self.n_input_dims_cam += 3
            self.n_input_dims_ref += 3
            self.n_input_dims_w += 3
        
        if self.predict_weight:
            self.weight_network = get_mlp(self.n_input_dims_w, 1, self.config.weight_network_config)    
        self.cam_network = get_mlp(self.n_input_dims_cam, self.n_output_dims, self.config.cam_network_config)    
        self.ref_network = get_mlp(self.n_input_dims_ref, self.n_output_dims, self.config.ref_network_config)


    def forward(self, dirs=None, normals=None, positions=None, sdf=None, **kwargs):
        features = kwargs.get('appearance_params')
        if sdf is not None:
            if sdf.dim() == features.dim() - 1:
                sdf = sdf.unsqueeze(-1)
            features = torch.cat([sdf, features], dim=-1)


        refl_dirs = dirs - 2 * (dirs * normals).sum(dim=-1, keepdim=True) * normals

        # ----- Direction embedding -----
        if self.encode_dirs:
            dirs_proc = (dirs + 1.) / 2.  # normalize (-1,1)→(0,1)
            dirs_embd = self.dir_encoding(dirs_proc.reshape(-1, self.n_dir_dims))
        else:
            dirs_embd = dirs.reshape(-1, 3)

        # ----- Normal embedding -----
        if self.encode_normals:
            normals_proc = (normals + 1.) / 2.
            normal_embd = self.normal_encoding(normals_proc.reshape(-1, 3))
        else:
            normal_embd = normals.reshape(-1, 3)

        # ----- Reflected direction embedding -----
        if self.encode_refl_dirs:
            refl_dirs_proc = (refl_dirs + 1.) / 2.
            refl_dirs_embd = self.refl_dir_encoding(refl_dirs_proc.reshape(-1, self.n_dir_dims))
        else:
            refl_dirs_embd = refl_dirs.reshape(-1, 3)

        # ----- Position embedding -----
        if self.encode_positions:
            positions_proc = (positions + 1.) / 2.
            position_embd = self.position_encoding(positions_proc.reshape(-1, 3))
        else:
            position_embd = positions.reshape(-1, 3)

        # Start with features
        network_inputs_w = torch.cat([features.reshape(-1, features.shape[-1]), position_embd, normal_embd], dim=-1)
        network_inputs_cam = torch.cat([features.reshape(-1, features.shape[-1]), dirs_embd, normal_embd, position_embd], dim=-1)
        network_inputs_ref = torch.cat([features.reshape(-1, features.shape[-1]), refl_dirs_embd, normal_embd, position_embd], dim=-1)

        cam_color = self.cam_network(network_inputs_cam).reshape(*features.shape[:-1], self.n_output_dims).float()
        ref_color = self.ref_network(network_inputs_ref).reshape(*features.shape[:-1], self.n_output_dims).float()

        if self.predict_weight:
            weight = self.weight_network(network_inputs_w).reshape(*features.shape[:-1], 1).float()
            color = cam_color * (1 - torch.sigmoid(weight)) + ref_color * torch.sigmoid(weight)
        else:
            color = torch.sigmoid(cam_color + ref_color)
        return color

    def regularizations(self, out):
        return {}
