import gc
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import tinycudann as tcnn
from nerfacc import ContractionType
from utils.misc import get_rank
import math
import numpy as np
import open3d as o3d
import trimesh

class MarchingCubeHelper(nn.Module):
    def __init__(self, resolution, use_torch=True):
        super().__init__()
        self.resolution = resolution
        self.use_torch = use_torch
        self.points_range = (0, 1)
        if self.use_torch:
            import torchmcubes
            self.mc_func = torchmcubes.marching_cubes
        else:
            import mcubes
            self.mc_func = mcubes.marching_cubes
        self.verts = None

    def grid_vertices(self):
        if self.verts is None:
            x, y, z = torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution)
            x, y, z = torch.meshgrid(x, y, z, indexing='ij')
            verts = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1).reshape(-1, 3)
            self.verts = verts
        return self.verts

    def forward(self, level, threshold=0.):
        level = level.float().view(self.resolution, self.resolution, self.resolution)
        if self.use_torch:
            verts, faces = self.mc_func(level.to(get_rank()), threshold)
            verts, faces = verts.cpu(), faces.cpu().long()
        else:
            verts, faces = self.mc_func(-level.numpy(), threshold) # transform to numpy
            verts, faces = torch.from_numpy(verts.astype(np.float32)), torch.from_numpy(faces.astype(np.int64)) # transform back to pytorch
        verts = verts / (self.resolution - 1.)
        return {
            'v_pos': verts,
            't_pos_idx': faces
        }

def contract_to_unisphere(x, radius, contraction_type):
    if contraction_type == ContractionType.AABB:
        x = scale_anything(x, (-radius, radius), (0, 1))
    elif contraction_type == ContractionType.UN_BOUNDED_SPHERE:
        x = scale_anything(x, (-radius, radius), (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        raise NotImplementedError
    return x

def chunk_batch(func, chunk_size, move_to_cpu, *args, **kwargs):
    B = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    out = defaultdict(list)
    out_type = None
    for i in range(0, B, chunk_size):
        out_chunk = func(*[arg[i:i+chunk_size] if isinstance(arg, torch.Tensor) else arg for arg in args], **kwargs)
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(f'Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}.')
            exit(1)
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            v = v.cpu() if move_to_cpu else v
            out[k].append(v)
    
    if out_type is None:
        return

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    if out_type is torch.Tensor:
        return out[0]
    elif out_type in [tuple, list]:
        return out_type([out[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

trunc_exp = _TruncExp.apply


def get_activation(name):
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == 'none':
        return lambda x: x
    elif name.startswith('scale'):
        scale_factor = float(name[5:])
        return lambda x: x.clamp(0., scale_factor) / scale_factor
    elif name.startswith('clamp'):
        clamp_max = float(name[5:])
        return lambda x: x.clamp(0., clamp_max)
    elif name.startswith('mul'):
        mul_factor = float(name[3:])
        return lambda x: x * mul_factor
    elif name == 'lin2srgb':
        return lambda x: torch.where(x > 0.0031308, torch.pow(torch.clamp(x, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*x).clamp(0., 1.)
    elif name == 'trunc_exp':
        return trunc_exp
    elif name.startswith('+') or name.startswith('-'):
        return lambda x: x + float(name)
    elif name == 'sigmoid':
        return lambda x: torch.sigmoid(x)
    elif name == 'tanh':
        return lambda x: torch.tanh(x)
    else:
        return getattr(F, name)
 

def dot(x, y):
    return torch.sum(x*y, -1, keepdim=True)


def reflect(x, n):
    return 2 * dot(x, n) * n - x


def scale_anything(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = [dat.min(), dat.max()]
    dat = (dat  - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()

def inverse_softplus(x, beta, scale=1):
    # log(exp(scale*x)-1)/scale
    out = x / scale
    mask = x * beta < 20 * scale
    out[mask] = torch.log(torch.exp(beta * out[mask]) - 1 + 1e-10) / beta
    return out


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(-1, img1.shape[-1]).mean(0, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def get_expon_lr_func(
    lr_init,
    lr_final,
    warmup_steps=0,
    max_steps=1_000,
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if warmup_steps and step < warmup_steps:
            return lr_init * step / warmup_steps
        elif step > max_steps:
            return 0
        t = np.clip((step - warmup_steps) / (max_steps - warmup_steps), 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return log_lerp

    return helper


def get_cosine_lr_func(
    lr_init,
    lr_final,
    warmup_steps=0,
    max_steps=10_000,
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if warmup_steps and step < warmup_steps:
            return lr_init * step / warmup_steps
        elif step > max_steps:
            return 0.0
        lr_cos = lr_final + 0.5 * (lr_init - lr_final) * (
            1
            + np.cos(np.pi * (step - warmup_steps) / (max_steps - warmup_steps))
        )
        return lr_cos

    return helper


def point_cloud_distance(positions, pcd):
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    #Cull point cloud to bounding box of positions
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=positions.min(axis=0)-0.1, max_bound=positions.max(axis=0)+0.1)
    pcd = pcd.crop(bbox)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
    mesh.compute_vertex_normals()
    #vertices_to_keep = densities > np.quantile(densities, 0.01)
    #mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])
    mesh_tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
    sdf = mesh_tri.nearest.signed_distance(positions)
    return sdf


def linear_schedule(step, start_value, end_value, start_step, end_step):
    if step <= start_step:
        return start_value
    elif step >= end_step:
        return end_value
    else:
        t = (step - start_step) / (end_step - start_step)
        return start_value + t * (end_value - start_value)
    
def logarithmic_schedule(step, start_value, end_value, start_step, end_step):
    if step <= start_step:
        return start_value
    elif step >= end_step:
        return end_value
    else:
        # Normalize step to [0, 1]
        t = (step - start_step) / (end_step - start_step)
        # Interpolate in log-space
        log_start = math.log(start_value)
        log_end = math.log(end_value)
        log_value = log_start + t * (log_end - log_start)
        return math.exp(log_value)
    
def exponential_schedule(step, start_value, end_value, start_step, end_step):
    if step <= start_step:
        return start_value
    elif step >= end_step:
        return end_value
    else:
        # Compute factor
        factor = (end_value / start_value) ** (1 / (end_step - start_step))
        return start_value * (factor ** (step - start_step))
    
import math

def sigmoid_schedule(step, start_value, end_value, start_step, end_step, k = 6.0):
    if step <= start_step:
        return start_value
    elif step >= end_step:
        return end_value
    else:
        # Normalize step to [0, 1]
        t = (step - start_step) / (end_step - start_step)
        # Apply sigmoid interpolation with normalization
        sigmoid_0 = 1 / (1 + math.exp(k / 2))   # value at t=0
        sigmoid_1 = 1 / (1 + math.exp(-k / 2))  # value at t=1
        sigmoid_t = 1 / (1 + math.exp(-k * (t - 0.5)))

        # Normalize to exactly [0, 1]
        normalized_t = (sigmoid_t - sigmoid_0) / (sigmoid_1 - sigmoid_0)

        return start_value + normalized_t * (end_value - start_value)

def get_schedule_function(config):
    name = config['name']
    start_value = config['start_value']
    end_value = config['end_value']
    start_step = config['start_step']
    end_step = config['end_step']
    print(f"Creating schedule function: {name} from {start_value} to {end_value} between steps {start_step} and {end_step}")
    if name == 'linear':
        return lambda step: linear_schedule(step, start_value, end_value, start_step, end_step)
    elif name == 'logarithmic':
        return lambda step: logarithmic_schedule(step, start_value, end_value, start_step, end_step)
    elif name == 'exponential':
        return lambda step: exponential_schedule(step, start_value, end_value, start_step, end_step)
    elif name == 'sigmoid':
        return lambda step: sigmoid_schedule(step, start_value, end_value, start_step, end_step)
    else:
        raise ValueError(f"Unknown schedule name: {name}")
    

def get_contraction_type(name):
    name = name.lower()
    if name == 'aabb':
        return ContractionType.AABB
    elif name == 'un_bounded_sphere':
        return ContractionType.UN_BOUNDED_SPHERE
    else:
        raise ValueError(f"Unknown contraction type: {name}")
    

def contract(
    x: torch.Tensor,
    roi: torch.Tensor,
    type: ContractionType = ContractionType.AABB)-> torch.Tensor:
    """Contract the space into [0, 1]^3.
    Args:
        x (torch.Tensor): Un-contracted points.
        roi (torch.Tensor): Region of interest.
        type (ContractionType): Contraction type.
    Returns:
        torch.Tensor: Contracted points ([0, 1]^3).
    """
    roi_min = roi[0:3]
    roi_max = roi[3:6]
    if type == ContractionType.AABB:
        return (x - roi_min) / (roi_max - roi_min)
    elif type == ContractionType.UN_BOUNDED_SPHERE:
        points = (x - roi_min) / (roi_max - roi_min)
        # [0, 1]^3 -> [-1, 1]^3
        points = points * 2.0 - 1.0
        norm = torch.norm(points, dim=-1, keepdim=True)
        # Contract points outside the unit sphere
        mask = norm > 1.0 #shape (num_points, 1)
        contraction_factor = (2.0*norm - 1.0) / (norm*norm)
        contracted = contraction_factor * points
        points = torch.where(mask, contracted, points)
        # [-1, 1]^3 -> [0.25, 0.75]^3
        points = points * 0.25 + 0.5
        return points
    
def contract_inv(  
    x: torch.Tensor,
    roi: torch.Tensor,
    type: ContractionType = ContractionType.AABB,
) -> torch.Tensor:
    """Recover the space from [0, 1]^3 by inverse contraction.
    Args:
        x (torch.Tensor): Contracted points ([0, 1]^3).
        roi (torch.Tensor): Region of interest.
        type (ContractionType): Contraction type.
    Returns:
        torch.Tensor: Un-contracted points.
    """
    roi_min = roi[0:3]
    roi_max = roi[3:6]
    if type == ContractionType.AABB:
        return x * (roi_max - roi_min) + roi_min
    elif type == ContractionType.UN_BOUNDED_SPHERE:
        points = (x - 0.5) * 4.0
        norm = torch.norm(points, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-9, max=2.0-1e-9)
        # Recover points outside the unit sphere
        mask = norm > 1.0 #shape (num_points, 1)
        inv_contraction_factor = 1.0/(2.0 - norm)
        recovered = inv_contraction_factor * points/norm
        points = torch.where(mask, recovered, points)
        # [-1, 1]^3 -> [0, 1]^3
        points = (points + 1.0) / 2.0
        return points * (roi_max - roi_min) + roi_min


def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2**i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    # Convert list into a numpy array.
    ml_array = np.array(ml_list).T
    return ml_array

def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    return np.prod(a - np.arange(k)) / np.math.factorial(k)


def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.

    Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
    (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

    Args:
      l: associated Legendre polynomial degree.
      m: associated Legendre polynomial order.
      k: power of cos(theta).

    Returns:
      A float, the coefficient of the term corresponding to the inputs.
    """
    return ((-1)**m * 2**l * np.math.factorial(l) / np.math.factorial(k) /
            np.math.factorial(l - k - m) *
            generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))


def sph_harm_coeff(l, m, k):
    """Compute spherical harmonic coefficients."""
    return (np.sqrt(
        (2.0 * l + 1.0) * np.math.factorial(l - m) /
        (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))

def generate_ide_fn(deg_view):
    """Generate integrated directional encoding (IDE) function.

    This function returns a function that computes the integrated directional
    encoding from Equations 6-8 of arxiv.org/abs/2112.03907.

    Args:
      deg_view: number of spherical harmonics degrees to use.

    Returns:
      A function for evaluating integrated directional encoding.

    Raises:
      ValueError: if deg_view is larger than 5.
    """
    if deg_view > 5:
        print('WARNING: Only deg_view of at most 5 is numerically stable.')
    #   raise ValueError('Only deg_view of at most 5 is numerically stable.')

    ml_array = get_ml_array(deg_view)
    l_max = 2**(deg_view - 1)

    # Create a matrix corresponding to ml_array holding all coefficients, which,
    # when multiplied (from the right) by the z coordinate Vandermonde matrix,
    # results in the z component of the encoding.
    mat = torch.zeros((l_max + 1, ml_array.shape[1]))
    for i, (m, l) in enumerate(ml_array.T):
        for k in range(l - m + 1):
            mat[k, i] = sph_harm_coeff(l, m, k)

    def integrated_dir_enc_fn(xyz, kappa_inv):
        """Function returning integrated directional encoding (IDE).

        Args:
        xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
        kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
            Mises-Fisher distribution.

        Returns:
        An array with the resulting IDE.
        """

        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        # Compute z Vandermonde matrix.
        vmz = torch.cat([z**i for i in range(mat.shape[0])], axis=-1)

        # Compute x+iy Vandermonde matrix.
        vmxy = torch.cat([(x + 1j * y)**m for m in ml_array[0, :]], axis=-1)

        # Get spherical harmonics.
        sph_harms = vmxy * torch.matmul(vmz, mat.to(xyz.device))

        # Apply attenuation function using the von Mises-Fisher distribution
        # concentration parameter, kappa.
        sigma = 0.5 * ml_array[1, :] * (ml_array[1, :] + 1)
        ide = sph_harms * torch.exp(-torch.tensor(sigma, device=xyz.device) * kappa_inv)
        
        # Split into real and imaginary parts and return
        return torch.concatenate([torch.real(ide), torch.imag(ide)], axis=-1).float()

    return integrated_dir_enc_fn