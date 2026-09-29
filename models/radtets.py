import math
import models
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from render import renderutils as ru
from render import util
from models.utils import *
from nerfacc import render_weight_from_alpha
from models.base import BaseModel
from models.utils import chunk_batch
from render import light, mesh
from nerfacc import ContractionType, OccupancyGrid, ray_marching, accumulate_along_rays
import torch.nn as nn
from geometry.dmtet import DMTet
from systems.utils import update_module_step
from pysdf import SDF
import trimesh
from utils.misc import C, get_rank
from .image_appearance import ImageAppearanceNetwork
from .grid_manager import GridManager
import math
import numpy as np
import radfoam
from nerfacc import contract, contract_inv

class NormalTexture(nn.Module):
    """Volumetric normal map: a small dedicated hashgrid + zero-init head that
    outputs tangent-space offsets (tx, ty), composed onto the geometric normal
    as normalize(tx*T + ty*B + N) — the [tx, ty, 1] convention of a rendering
    normal texture. Identity at init (head zeroed). Perturbs only the shading
    normal fed to the appearance model; geometry and its regularizers keep the
    geometric normal."""

    def __init__(self, config, roi):
        super().__init__()
        from models.network_utils import get_encoding, get_mlp
        self.config = config
        self.register_buffer('roi_min', roi[:3].clone())
        self.register_buffer('roi_size', (roi[3:] - roi[:3]).clone())
        self.gain = float(config.get('gain', 1.0))
        self.encoding = get_encoding(3, config.encoding_config)
        self.head = get_mlp(self.encoding.n_output_dims, 2, config.mlp_network_config)
        # zero the final layer -> zero offsets -> shading normal == geometric at start
        last = None
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                last = m
        if last is not None:
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def _offsets(self, positions):
        x = ((positions - self.roi_min) / self.roi_size).clamp(0.0, 1.0)
        o = self.head(self.encoding(x.reshape(-1, 3))).reshape(*positions.shape[:-1], 2).float()
        return self.gain * torch.tanh(o)

    def deviation_penalty(self, positions):
        """Mean (1 - cos(shading, geometric)) = 1 - 1/sqrt(1 + tx^2 + ty^2).
        Depends ONLY on the texture field at `positions` (not the geometric
        normals), so it is computed on DETACHED positions in a standalone
        forward — decoupled from the render graph (which, in the volumetric
        path, carries a create_graph=True SDF gradient and would otherwise
        cause a double-backward)."""
        txy = self._offsets(positions)
        cos = 1.0 / torch.sqrt(1.0 + (txy ** 2).sum(-1))
        return (1.0 - cos).mean()

    def forward(self, positions, normals):
        txy = self._offsets(positions)
        n = F.normalize(normals, p=2, dim=-1)
        # branchless orthonormal basis (Duff et al.), same construction as appearance.py
        sign = torch.where(n[..., 2] >= 0, 1.0, -1.0)
        a = -1.0 / (sign + n[..., 2])
        b = n[..., 0] * n[..., 1] * a
        t = torch.stack([1.0 + sign * n[..., 0] ** 2 * a, sign * b, -sign * n[..., 0]], dim=-1)
        bt = torch.stack([b, sign + n[..., 1] ** 2 * a, -n[..., 1]], dim=-1)
        out = n + txy[..., :1] * t + txy[..., 1:2] * bt
        return F.normalize(out, p=2, dim=-1)


@models.register('radtets')
class RadTetsModel(BaseModel):

    def setup(self):        
        #Set region of interest and initialize contraction
        self.register_buffer('offset', torch.tensor(self.config.offset, dtype=torch.float32))
        self.register_buffer('radius', torch.tensor(self.config.radius, dtype=torch.float32))
        roi = [
            self.config.offset[0] - self.config.radius,
            self.config.offset[1] - self.config.radius,
            self.config.offset[2] - self.config.radius,
            self.config.offset[0] + self.config.radius,
            self.config.offset[1] + self.config.radius,
            self.config.offset[2] + self.config.radius,
        ]
        self.register_buffer('scene_aabb', torch.as_tensor(roi, dtype=torch.float32))
        self.contraction_type = get_contraction_type(self.config.get('contraction_type', 'aabb'))

        #Initialize per-image appearance embeddings
        if self.config.get('appearance_embeddings', False):
            std = 1e-4
            self.appearance_embedding_network = AppearanceNetwork(3+64, 3).cuda()
            self._appearance_embeddings = nn.Parameter(torch.empty(2048, 64).cuda())
            self._appearance_embeddings.data.normal_(0, std)

        # Set up Models
        self.appearance_model = models.make(self.config.appearance.model.name, self.config.appearance.model)
        self.sdf_model = models.make(self.config.sdf.model.name, self.config.sdf.model)
        self.variance_model = models.make(self.config.variance.model.name, self.config.variance.model)

        # Optional learned volumetric normal texture (shading normals only)
        nt_cfg = self.config.get('normal_texture', None)
        if nt_cfg is not None and nt_cfg.get('enabled', False):
            self.normal_texture = NormalTexture(nt_cfg, torch.as_tensor(roi, dtype=torch.float32))
        else:
            self.normal_texture = None

        # Optional periodic reset of the appearance MLP. Snapshot taken right
        # after construction so we restore to the *exact* random init this
        # training run started with — reproducible, regardless of init scheme.
        # Memory cost is negligible (an MLP is tiny vs. the hashgrid).
        self._appearance_reset_interval = int(self.config.appearance.get('reset_every_n_steps', 0) or 0)
        if self._appearance_reset_interval > 0:
            self._appearance_init_state = {
                k: v.detach().clone() for k, v in self.appearance_model.state_dict().items()
            }
        else:
            self._appearance_init_state = None

        # Initialize grids and features
        self.feature_registry = self.config.get('features', {})
        _hr = float((self.config.get('hashgrid', {}) or {}).get('radius', 1.0))
        _gs = float(self.config.grid.get('scale', 1.0))
        if abs(_hr - _gs) > 1e-6:
            print(f"[radtets] WARNING: hashgrid.radius={_hr} != grid.scale={_gs} — "
                  f"mesh branch (tet grid) and SDF field span DIFFERENT world sizes; "
                  f"meshes/renders will be mis-scaled unless intentional.", flush=True)
        self.tetrahedral_grid = models.make("tetrahedral_grid", self.config.grid)

        num_hashgrid_features = 0
        self.hashgrid_layout = {} # name -> (offset, dim)
        hashgrid_grad_scales = []
        for name, feat in self.feature_registry.items():
            storage = feat.get('storage', 'hashgrid')
            dim = feat['dim']
            
            if storage == 'global':
                init_cfg = feat.get('init', {})
                mode = init_cfg.get('type', 'constant')
                args = init_cfg.get('args', {})
                if mode == 'constant':
                    val = torch.full((dim,), args.get('value', 0.0))
                elif mode == 'random':
                    val = torch.randn(dim) * args.get('std', 0.1) + args.get('mean', 0.0)
                else:
                    val = torch.zeros(dim)
                self.register_parameter(name, nn.Parameter(val.cuda()))
            
            elif storage == 'tetrahedral_grid':
                self.tetrahedral_grid.initialize_feature(name, feat)
            
            elif storage == 'hashgrid':
                self.hashgrid_layout[name] = (num_hashgrid_features, dim)
                num_hashgrid_features += dim
                grad_scale = feat.get('grad_scale', 1.0)
                hashgrid_grad_scales.extend([grad_scale] * dim)
            
            else:
                raise ValueError(f"Unknown storage type '{storage}' for feature '{name}'")

        if num_hashgrid_features > 0:
            hashgrid_config = self.config.hashgrid
            hashgrid_config['feature_dim'] = num_hashgrid_features
            hashgrid_config['grad_scales'] = hashgrid_grad_scales
            self.hashgrid = models.make(hashgrid_config.get('name', 'hashgrid'), hashgrid_config)
        else:
            self.hashgrid = None
        self.recompute_grid_features = True

        #Create occupancy grid
        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=128,
                contraction_type=get_contraction_type(self.config.get('contraction_type', 'aabb'))
            )

        #Grid Manager
        self.grid_manager = GridManager(
            self.config.grid_manager, 
            self.tetrahedral_grid,
            get_viewport_mask_fn=self.get_viewport_mask,
            get_contribution_mask_fn=self.get_mesh_contribution_mask,
            get_densify_probs_fn=self.get_densification_probs
        )

        # Other variables
        self.all_mvp = None
        self.all_campos = None
        self.image_resolution = None

        #Rendering parameters
        self.num_mesh_offset_samples = self.config.mesh_offset_sampling.num_samples[1] if isinstance(self.config.mesh_offset_sampling.num_samples, list) else self.config.mesh_offset_sampling.num_samples
        self.marching_tets = DMTet()
        self.randomized = self.config.randomized
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
        self.backface_culling = self.config.get('backface_culling', False)

        #Background
        self.learned_background = self.config.background.learned
        if self.learned_background:
            self.background_network = models.make(self.config.background.model.name, self.config.background.model)
        else:
            if (self.config.background.default_color == 'white'):
                self.default_background_color = torch.ones(3, dtype=torch.float32, device=self.rank)
            elif (self.config.background.default_color == 'black'):
                self.default_background_color = torch.zeros(3, dtype=torch.float32, device=self.rank)
            else:
                raise ValueError("default_background_color must be 'white' or 'black'")

    def pretrain_sdf_double_sphere(self):
        """Fit the (hashgrid-stored) SDF to the analytic double-sphere
        sdf(x) = min(|x| - r_fg, R_bg - |x|): a foreground object seed inside an
        inward-facing background shell. Used for unbounded scenes where the SDF
        lives on a contracted hashgrid and cannot be shaped by feature init.
        Config (model.sdf_pretrain): steps, r_fg, r_bg, lr, batch, contracted_max."""
        cfg = self.config.get('sdf_pretrain', None)
        if not cfg or self.hashgrid is None:
            return
        steps = int(cfg.get('steps', 1500))
        r_fg = float(cfg.get('r_fg', 1.0))
        r_bg = float(cfg.get('r_bg', 8.0))
        lr = float(cfg.get('lr', 0.01))
        bs = int(cfg.get('batch', 65536))
        cmax = float(cfg.get('contracted_max', 0.999))
        radius = float(self.config.radius)
        opt = torch.optim.Adam(self.hashgrid.parameters(), lr=lr)
        for it in range(steps):
            # sample uniformly in CONTRACTED space so far-field gets equal capacity
            d = torch.randn(bs, 3, device=self.rank)
            d = d / d.norm(dim=-1, keepdim=True)
            rc = cmax * torch.rand(bs, 1, device=self.rank) ** (1.0 / 3.0)  # [-1,1] contracted ball
            xc = d * rc
            m = xc.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            # invert the hashgrid's contraction: |x|<=1 identity, else (2m-1)/m^2 fwd
            #   forward: y = x * (2|x|-1)/|x|^2  =>  |y| = 2 - 1/|x|  =>  |x| = 1/(2-|y|)
            inv_scale = torch.where(m > 1.0, 1.0 / ((2.0 - m).clamp_min(1e-3) * m), torch.ones_like(m))
            xw = xc * inv_scale * radius                                   # world coords
            n = xw.norm(dim=-1, keepdim=True)
            target = torch.minimum(n - r_fg, r_bg - n)
            hg = self.hashgrid(xw, contract_input=True)
            sdf_inputs = self._collect_features(self.config.sdf.input_parameters, xw, hg)
            sdf = self.sdf_model(**sdf_inputs)
            loss = F.l1_loss(sdf.squeeze(-1), target.squeeze(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 300 == 0 or it == steps - 1:
                print(f"[sdf_pretrain it{it}] loss={float(loss):.4f}")
        del opt
        torch.cuda.empty_cache()

    def dump_radial_sdf(self, path, rmax=60.0, nr=400, ndir=256):
        """Radial SDF profile of the initialised field: for many directions,
        query the SDF along world radius [0, rmax] and report mean/std + sign
        crossings. Decisive check for double-sphere init correctness."""
        import numpy as np
        with torch.no_grad():
            rs = torch.linspace(0.02, rmax, nr, device=self.rank)
            d = torch.randn(ndir, 3, device=self.rank); d = d / d.norm(dim=-1, keepdim=True)
            prof_mean = []; prof_std = []
            for r in rs:
                x = d * r
                hg = self.hashgrid(x, contract_input=True) if self.hashgrid is not None else None
                sdf_inputs = self._collect_features(self.config.sdf.input_parameters, x, hg)
                sdf = self.sdf_model(**sdf_inputs).squeeze(-1)
                prof_mean.append(float(sdf.mean())); prof_std.append(float(sdf.std()))
            rs = rs.cpu().numpy(); mean = np.array(prof_mean); std = np.array(prof_std)
            cr = [float((rs[i]+rs[i+1])/2) for i in range(len(rs)-1) if (mean[i] < 0) != (mean[i+1] < 0)]
            lines = ["r,sdf_mean,sdf_std"] + [f"{rs[i]:.3f},{mean[i]:.4f},{std[i]:.4f}" for i in range(len(rs))]
            open(path,'w').write("\n".join(lines))
            print(f"[radial_sdf] mean-sign crossings at world r: {[round(c,2) for c in cr]}")
            print(f"[radial_sdf] std at r~12: {std[np.argmin(np.abs(rs-12))]:.3f}  at r~30: {std[np.argmin(np.abs(rs-30))]:.3f}  at r~50: {std[np.argmin(np.abs(rs-50))]:.3f}")
            # per-direction crossing count (spurious far crossings show up here)
            allx = (d[:,None,:]*torch.linspace(0.02,rmax,nr,device=self.rank)[None,:,None]).reshape(-1,3)
            hg = self.hashgrid(allx, contract_input=True) if self.hashgrid is not None else None
            si = self._collect_features(self.config.sdf.input_parameters, allx, hg)
            allsdf = self.sdf_model(**si).squeeze(-1).reshape(ndir, nr)
            ncross = ((allsdf[:,1:]<0)!=(allsdf[:,:-1]<0)).sum(-1).float()
            print(f"[radial_sdf] per-ray crossings: mean={float(ncross.mean()):.2f} max={int(ncross.max())} (want 2)")

    def register_optimizer(self, optim):
        self.tetrahedral_grid.register_optimizer(optim)
        # Stash on self too so periodic resets (e.g. appearance MLP) can clear
        # Adam momentum without grabbing the trainer.
        self.optimizer = optim

    def _appearance_dirs(self, d):
        """Optionally perturb the view directions fed to the appearance model
        (model.viewdir_noise_deg, mesh branch, TRAINING only). Deliberate,
        controlled version of the accidental regularizer the jitter ray bugs
        produced: weakening the usable view-dir conditioning pushes appearance
        variation into geometry/diffuse instead of the view-dependent MLP.
        Per-pixel isotropic Gaussian on the direction, sigma = tan(noise_deg),
        renormalized. Validation/test render with clean dirs."""
        deg = float(self.config.get('viewdir_noise_deg', 0.0))
        if deg <= 0.0 or not self.training or d is None:
            return d
        sigma = math.tan(math.radians(deg))
        return F.normalize(d + sigma * torch.randn_like(d), dim=-1)

    def _shading_normals(self, positions, normals):
        """Apply the learned volumetric normal texture (if enabled) to get the
        shading normal for appearance evaluation. No-op when disabled."""
        if self.normal_texture is None or normals is None:
            return normals
        return self.normal_texture(positions, normals)

    def reset_appearance_model(self):
        """Restore appearance MLP weights to their initial values and zero
        Adam momentum for those parameters.

        Implementation notes:
          * `load_state_dict` (default `assign=False`) does an in-place
            `param.data.copy_` per tensor — Parameter object identity is
            preserved, which is required because the optimizer's
            `param_groups` reference these tensors directly.
          * We zero `exp_avg` / `exp_avg_sq` (and `max_exp_avg_sq` for AMSGrad,
            `step` for completeness) on every Parameter that belongs to
            `appearance_model`. Without this the accumulated momentum would
            instantly drag the freshly-reset weights back toward whatever they
            had drifted to, defeating the reset.
        """
        if self._appearance_init_state is None:
            return
        self.appearance_model.load_state_dict(self._appearance_init_state)
        if getattr(self, 'optimizer', None) is None:
            return
        ap_param_ids = {id(p) for p in self.appearance_model.parameters()}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if id(p) not in ap_param_ids:
                    continue
                state = self.optimizer.state.get(p, None)
                if state is None:
                    continue
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'].zero_()
                if 'max_exp_avg_sq' in state:
                    state['max_exp_avg_sq'].zero_()
                if 'step' in state:
                    s = state['step']
                    if torch.is_tensor(s):
                        s.zero_()
                    else:
                        state['step'] = 0

    def get_appearance_embedding(self, idx):
        return self._appearance_embeddings[idx]

    def get_feature(self, name, x, precomputed_hashgrid_features = None):
        feat = self.feature_registry.get(name)
        if feat is None:
            return None
        
        storage = feat.get('storage', 'hashgrid')
        dim = feat['dim']
        
        if storage == 'global':
            val = getattr(self, name)
            if x is not None:
                #Expand to shape of x in ALL but trailign dimension
                shape = x.shape[:-1] + (dim,)
                return val.expand(shape)
            else:
                return val
        elif storage == 'hashgrid':
            if precomputed_hashgrid_features is None:
                hashgrid_features = self.get_hashgrid_features(x)
            else:
                hashgrid_features = precomputed_hashgrid_features
            offset, _ = self.hashgrid_layout[name]
            return hashgrid_features[..., offset:offset+dim]
        elif storage == 'tetrahedral_grid':
            if x is not None:
                return self.tetrahedral_grid.interpolate(x, name)
            else:
                return getattr(self.tetrahedral_grid, name)
        return None

    def get_hashgrid_features(self, x):
        # with_grad= was dropped from HashGrid.forward; every other call site
        # already uses the two-arg form.
        hashgrid_features = self.hashgrid(x, contract_input=True)
        hashgrid_features_dict = {}
        for name, (offset, dim) in self.hashgrid_layout.items():
            hashgrid_features_dict[name] = hashgrid_features[..., offset:offset+dim]
        return hashgrid_features_dict

    def _collect_features(self, mappings, x, hashgrid_features=None):
        results = {}
        for item in mappings:
            stored_name, arg_name = (item, item) if isinstance(item, str) else list(item.items())[0]
            results[arg_name] = self.get_feature(stored_name, x, hashgrid_features)
        return results

    def _collect_grid_features(self, mappings):
        results = {}
        for item in mappings:
            stored_name, arg_name = (item, item) if isinstance(item, str) else list(item.items())[0]
            results[arg_name] = self.get_grid_feature(stored_name)
        return results

    def get_inv_s(self, x=None, hashgrid_features=None):
        variance_inputs = self._collect_features(self.config.variance.input_parameters, x, hashgrid_features)
        return self.variance_model(**variance_inputs)

    def sample_offset(self, gb_pos, view_pos, t_dirs, num_offset_samples, cos, inv_s):
        dists = (gb_pos - view_pos)
        dists_norm = torch.norm(dists, dim=-1, keepdim=True).clamp(min=1e-12)

        with torch.no_grad():
            t_ends = torch.linspace(0, 1, num_offset_samples + 1, device=gb_pos.device)[1:-1]
            perturbation = torch.rand((*t_dirs.shape[:-1], num_offset_samples-1), device=gb_pos.device)
            t_ends = t_ends[None, None, None, :] + (perturbation - 0.5) * (0.5 / num_offset_samples)
            t_ends = torch.cat([torch.zeros_like(t_ends[..., :1]), t_ends, torch.ones_like(t_ends[..., :1])], dim=-1)
            t_mids = 0.5*(t_ends[...,:-1] + t_ends[...,1:])
            t_ends = t_ends[...,1:-1]
            inv_t_ends = torch.logit(t_ends)/inv_s
            inv_t_mids = torch.logit(t_mids)/inv_s
            inv_t_ends = inv_t_ends  / cos + dists_norm
            inv_t_mids = inv_t_mids / cos + dists_norm

        sample_dist_offsets = inv_t_mids - dists_norm.detach()
        sampled_gb_pos = view_pos.unsqueeze(-2) + t_dirs.unsqueeze(-2) * (dists_norm + sample_dist_offsets).unsqueeze(-1)
        
        t_ends_sig = (inv_t_ends - dists_norm) * cos
        t_ends_sig = torch.sigmoid(t_ends_sig * inv_s)
        t_ends_sig = torch.cat([torch.zeros((*t_ends_sig.shape[:-1], 1), device=t_ends_sig.device), t_ends_sig, torch.ones((*t_ends_sig.shape[:-1], 1), device=t_ends_sig.device)], dim=-1)
        weights = t_ends_sig[..., 1:] - t_ends_sig[..., :-1]
        
        return sampled_gb_pos, weights

    def _interpolate(self, attr, rast, attr_idx, rast_db=None):
        return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

    @property
    def glctx(self):
        if not hasattr(self, '_glctx'):
            self._glctx = dr.RasterizeCudaContext()
        return self._glctx
    #-------------------------------------
    # Update methods
    #-------------------------------------

    def update_step(self, epoch, global_step):

        # Must be set before update_occ_grid() below, which evaluates get_alpha()
        # and reads self.cos_anneal_ratio (otherwise crashes at the first step).
        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)

        #Update submodules
        update_module_step(self.tetrahedral_grid, epoch, global_step)
        update_module_step(self.sdf_model, epoch, global_step)
        update_module_step(self.appearance_model, epoch, global_step)
        update_module_step(self.grid_manager, epoch, global_step)
        update_module_step(self.variance_model, epoch, global_step)
        if (self.hashgrid is not None):
            update_module_step(self.hashgrid, epoch, global_step)
        if self.training:
            if self.config.grid_prune:
                self.update_occ_grid(global_step)
            # Periodic appearance MLP reset. Only fires during training, never
            # at step 0 (init already gave us those weights), and never if no
            # interval was configured.
            if (self._appearance_reset_interval > 0
                and global_step > 0
                and global_step % self._appearance_reset_interval == 0):
                self.reset_appearance_model()
                print(f"[radtets] reset appearance MLP at step {global_step}")

        #Update rendering parameters and tracking variables
        self.spp = C(self.config.get('spp', 1), global_step, epoch)
        self.num_mesh_offset_samples = int(C(self.config.mesh_offset_sampling.num_samples, global_step, epoch))
        self.recompute_grid_features = True


    def update_occ_grid(self, global_step):
        #Update occupancy grid
        def occ_eval_fn(x):
            if self.hashgrid is not None:
                hashgrid_features = self.hashgrid(x, contract_input=True)
            else:
                hashgrid_features = None
            
            sdf_inputs = self._collect_features(self.config.sdf.input_parameters, x, hashgrid_features)
            sdf = self.sdf_model(**sdf_inputs)
            
            inv_s = self.get_inv_s(hashgrid_features=hashgrid_features)
            
            # Use get_alpha with dummy normal/dirs
            dirs = torch.zeros_like(x)
            dirs[..., 2] = -1.0
            normals = torch.zeros_like(x)
            normals[..., 2] = 1.0
            dists = torch.full_like(sdf, self.render_step_size)
            
            return self.get_alpha(sdf, inv_s, normals, dirs, dists)

        self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn, occ_thre=self.config.get('grid_prune_occ_thre', 0.01))
        

    #-------------------------------------
    # Helper methods for pruning and densification
    #-------------------------------------

    def _prune_view_indices(self):
        """View indices used for visibility/contribution pruning. In sparse-view
        training (model.train_view_ids set) restrict pruning to the training views
        so it is consistent with densification + the supervised loss (otherwise
        points seen only by non-training cameras are never pruned). Returns None
        for dense runs -> use all views (unchanged behaviour)."""
        vids = self.config.get('train_view_ids', None)
        if not vids:
            return None
        return torch.as_tensor(list(vids), dtype=torch.long, device=self.all_mvp.device)

    @torch.no_grad()
    def get_viewport_mask(self):
        points = self.tetrahedral_grid.primal_points_uncontracted
        mvp = self.all_mvp
        vidx = self._prune_view_indices()
        if vidx is not None:
            mvp = mvp[vidx]
        v_pos_clip = ru.xfm_points(points[None, ...].clone(), mvp[:,:,:]) # (N, V, 4)
        v_pos_clip = v_pos_clip[..., :3] / v_pos_clip[..., 3:4]
        valid_points_mask = v_pos_clip.abs().max(dim=-1).values <= 1.0
        valid_points_mask = valid_points_mask.any(dim=0) # (V,)
        return valid_points_mask

    def get_mesh_contribution_mask(self):
        """
        Returns a boolean mask for primal points that contribute to any rendered mesh image.
        A point is considered contributing if the gradient of the rendered image w.r.t. that point
        is non-zero in any view.
        """
        device = self.tetrahedral_grid.primal_points.device
        accumulated_grad = torch.zeros_like(self.tetrahedral_grid.primal_points, device=device)
        original_pp = self.tetrahedral_grid.primal_points

        vidx = self._prune_view_indices()
        view_iter = range(self.all_mvp.shape[0]) if vidx is None else vidx.tolist()

        for i in view_iter:
            mvp = self.all_mvp[i]
            campos = self.all_campos[i]
            primal_points_copy = original_pp.detach().clone().requires_grad_(True)
            # Swap the attribute (prefer assignment over .data)
            replaced_with_param = False
            if isinstance(original_pp, torch.nn.Parameter):
                self.tetrahedral_grid.primal_points = torch.nn.Parameter(primal_points_copy)
                self.tetrahedral_grid.recompute_primal_points_uncontracted = True
                replaced_with_param = True
            else:
                # plain tensor attribute
                self.tetrahedral_grid.primal_points = primal_points_copy

            try:
                # Ensure gradient tracking is enabled for the rendering op
                with torch.enable_grad():
                    batch = self.render_mesh(
                        mvp=mvp[None, ...],
                        view_pos=campos[None, ...],
                        resolution=self.image_resolution,
                        background=torch.zeros((1, self.image_resolution[0], self.image_resolution[1], 3), device=mvp.device),
                        spp=1,
                        sample_sdf=False,
                    )

                    image = batch['rgb']
                    # sanity checks
                    if not image.requires_grad:
                        # image does not depend on the points -> skip
                        # This may indicate render_mesh is using no_grad/detach internally
                        # or the points are out of the view / unused.
                        # We continue to next view.
                        continue

                    image_scalar = image.sum()

                    grads = torch.autograd.grad(
                        outputs=image_scalar,
                        inputs=self.tetrahedral_grid.primal_points,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True
                    )[0]  # grads can be None if allow_unused

                    if grads is not None:
                        accumulated_grad += grads.abs()

            finally:
                # Restore original primal_points exactly as it was
                self.tetrahedral_grid.primal_points = original_pp

        grad_sum = accumulated_grad.sum(dim=-1)
        return (grad_sum != 0)

    def _densify_min_size_mask(self):
        """Keep-mask over tets for densification: 0 for tets already at/below the
        minimum size so subdivision never manufactures tinier (sliver-prone) tets.
        Threshold from grid_manager.densification: `min_tet_edge` (absolute, model
        frame) and/or `min_tet_edge_factor` (x median max-edge, scale-adaptive);
        the larger of the two applies. Returns None if neither is set."""
        dcfg = self.config.grid_manager.densification
        abs_thr = float(dcfg.get('min_tet_edge', 0.0))
        fac = float(dcfg.get('min_tet_edge_factor', 0.0))
        if abs_thr <= 0 and fac <= 0:
            return None
        tp = self.tetrahedral_grid.primal_points_uncontracted[self.tetrahedral_grid.indices]  # [T,4,3]
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        max_edge = torch.stack([(tp[:, a] - tp[:, b]).norm(dim=-1) for a, b in pairs], -1).max(-1).values
        thr = abs_thr
        if fac > 0:
            thr = max(thr, fac * max_edge.median().item())
        keep = (max_edge >= thr).float()
        print(f"[densify] min-tet-size mask: keeping {int(keep.sum())}/{keep.numel()} tets (thr={thr:.4f})", flush=True)
        return keep

    def get_densification_probs(self, num_new_points):
        densify_by_gradient = self.config.get('densify_by_gradient', True)
        size_mask = self._densify_min_size_mask()
        if densify_by_gradient:
            #Get per point average gradient magnitude
            point_grads = self.tetrahedral_grid.point_grad_accum
            tet_grads = point_grads[self.tetrahedral_grid.indices].mean(dim=-1)
            tet_occ = tet_grads
            if self.config.get('weight_densification_by_distance', False):
                #Weight by distance to center
                tet_pts = self.tetrahedral_grid.primal_points_uncontracted[self.tetrahedral_grid.indices]
                tet_centers = tet_pts.mean(dim=1)
                center_dists = torch.norm(tet_centers, dim=-1)
                dist_weights = torch.where(center_dists > 4.0*self.config.radius, 0.0, 1.0)
                tet_occ = tet_occ * dist_weights
            if size_mask is not None:
                tet_occ = tet_occ * size_mask
            return tet_occ, num_new_points
        else:
            tet_sdf = self.grid_sdf[self.tetrahedral_grid.indices]
            tet_occ = ((tet_sdf.min(dim=-1).values < 0.0) & (tet_sdf.max(dim=-1).values > 0.0))
            tet_occ = tet_occ.to(torch.float32)
            if size_mask is not None:
                tet_occ = tet_occ * size_mask
            num_to_sample = min(num_new_points, int(tet_occ.sum().item()))
            return tet_occ, num_to_sample

    #-------------------------------------
    # Grid methods
    #-------------------------------------

    def get_grid_feature(self, name):
        feat = self.feature_registry.get(name)
        if feat is None:
            return None
        
        storage = feat.get('storage', 'hashgrid')
        dim = feat['dim']
        
        if storage == 'global':
            val = getattr(self, name)
            N = self.tetrahedral_grid.primal_points.shape[0]
            return val.unsqueeze(0).expand(N, dim)
        elif storage == 'tetrahedral_grid':
            return getattr(self.tetrahedral_grid, name)
        elif storage == 'hashgrid':
            if getattr(self, 'recompute_grid_features', True) or not hasattr(self, '_grid_features'):
                self._grid_features = self.hashgrid(
                    self.tetrahedral_grid.primal_points, 
                    contract_input=False
                )
                self.recompute_grid_features = False
            offset, _ = self.hashgrid_layout[name]
            return self._grid_features[..., offset:offset+dim]
        return None

    @property
    def grid_sdf(self):
        sdf_inputs = self._collect_grid_features(self.config.sdf.input_parameters)
        return self.sdf_model(**sdf_inputs)

    def sample_grid_sdf(self, num_samples=1, seed=None):
        sdf_inputs = self._collect_grid_features(self.config.sdf.input_parameters)
        return self.sdf_model.sample(num_samples=num_samples, **sdf_inputs, seed = seed)
    
    #-------------------------------------
    # Mesh-related methods
    #-------------------------------------
    def _mesh_extraction_indices(self):
        """Tetrahedra to feed marching_tets. Optionally drops oversized tets
        (config.mesh_extraction.filter_large_tets) to suppress the huge spurious
        triangles that Delaunay produces across empty space in large-scale
        scenes. `self.tetrahedral_grid.indices` is left intact (tracer/densify
        keep the full triangulation); this only affects mesh extraction."""
        indices = self.tetrahedral_grid.indices
        me = self.config.get('mesh_extraction', None)
        if me is None or not me.get('filter_large_tets', False):
            return indices
        keep = self.tetrahedral_grid.large_tet_keep_mask(
            me.get('tet_max_edge_factor', None), me.get('tet_max_edge_abs', None))
        return indices[keep]

    def getMesh(self):
        sdf = self.grid_sdf
        # Run DM tet to get a base mesh
        grid_features = self.tetrahedral_grid.features
        if self.config.appearance.get("interpolate_mesh", False):
            for name in self.config.appearance.input_parameters:
                #If the feature is stored on hashgrid add it to the mesh features
                if self.feature_registry[name]['storage'] == 'hashgrid':
                    grid_features[name] = self.get_grid_feature(name)
        tet_indices = self._mesh_extraction_indices()
        verts, faces, uvs, uv_idx, vert_features, face_tet_idx = self.marching_tets(self.tetrahedral_grid.primal_points_uncontracted, sdf, tet_indices, grid_features)
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, v_feature=vert_features)
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        # Stash for the (rare) caller that wants topology bookkeeping; not
        # required for the deterministic render path.
        imesh.face_tet_idx = face_tet_idx
        return imesh

    def sampleMesh(self, num_samples=1, seed=None):
        sdf_samples, log_prob_sign = self.sample_grid_sdf(num_samples=num_samples, seed=seed)
        meshes = []
        for i in range(num_samples):
            sdf = sdf_samples[i]
            # Run DM tet to get a base mesh
            features = self.tetrahedral_grid.features
            tet_indices = self._mesh_extraction_indices()
            verts, faces, uvs, uv_idx, vert_features, face_tet_idx = self.marching_tets(self.tetrahedral_grid.primal_points_uncontracted, sdf, tet_indices, features)
            imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, v_feature=vert_features)
            imesh = mesh.auto_normals(imesh)
            imesh = mesh.compute_tangents(imesh)
            # Topology bookkeeping for the REINFORCE term: face_tet_idx maps
            # rasterized triangle -> source tet, log_prob_sign[v] = log P of
            # this sample's sign at vertex v under N(mu_v, sigma_v^2).
            imesh.face_tet_idx = face_tet_idx
            imesh.log_prob_sign = log_prob_sign[i]
            meshes.append(imesh)
        return meshes, log_prob_sign

    def merge_buffers(self, buffers, probs=None):
        if len(buffers) == 1:
            return buffers[0]
        
        # Simple averaging for MC integration of the expectation E[I]
        results = {}
        for key in buffers[0].keys():
            if torch.is_tensor(buffers[0][key]):
                results[key] = torch.stack([b[key] for b in buffers], dim=0).mean(dim=0)
            else:
                results[key] = buffers[0][key]
        return results

    #-------------------------------------
    # Saving/Loading functions
    #-------------------------------------
    @torch.no_grad()
    def save_grid(self, filename, scale = 1.0, transform = torch.eye(4)):
        additional_fields = {'quality': self.grid_sdf.detach().cpu().numpy()}
        self.tetrahedral_grid.export(filename, additional_fields = additional_fields, scale = scale, transform = transform)

    @torch.no_grad()
    def save_mesh(self, mesh_path, scale = 1.0, transform = torch.eye(4)):
        mesh = self.getMesh()
        mesh.transform(scale = scale, transform = transform)
        mesh.export(mesh_path)

    #-------------------------------------
    # Rendering functions
    #-------------------------------------
    def render_mesh_depth(
            self,
            view_pos,
            mvp,
            resolution,
            spp=1):  
        view_pos = view_pos[:, None, None, :] if len(view_pos.shape) == 2 else view_pos
        opt_mesh = self.getMesh()
        v_pos_clip = ru.xfm_points(opt_mesh.v_pos[None, ...], mvp)
        if (v_pos_clip.size()[1] != 0):
            full_res = [resolution[0]*spp, resolution[1]*spp]
            with dr.DepthPeeler(self.glctx, v_pos_clip, opt_mesh.t_pos_idx.int(), full_res) as peeler:
                rast, rast_deriv = peeler.rasterize_next_layer()
                gb_pos, _ = self._interpolate(opt_mesh.v_pos[None, ...], rast, opt_mesh.t_pos_idx.int())

        dists = (gb_pos - view_pos) # (1, N, M, 3)
        depth = torch.norm(dists, dim=-1, keepdim=True).clamp(min=1e-12) # (1, N, M, 1)
        alpha = (rast[..., -1:] > 0).float() 
        depth = torch.lerp(torch.full_like(depth, 1e6), depth, alpha)
        return depth
    

    def _compute_pixel_probabilities_fast(self, mvp, resolution, view_pos, log_probs_samples):
        # Ultra-fast 2D projected points splatting strategy.
        # This accurately models the "points belonging to tetrahedra intersected by rays"
        # by scattering each vertex's log_probs into a screen-space neighborhood.
        num_samples = log_probs_samples.shape[0]
        B = mvp.shape[0]
        H, W = resolution

        pts_3d = self.tetrahedral_grid.primal_points_uncontracted
        pts_hom = torch.cat([pts_3d, torch.ones_like(pts_3d[:, :1])], dim=-1)
        pts_clip = (mvp @ pts_hom.t()).transpose(1, 2)
        pts_w = pts_clip[..., 3:4]
        pts_ndc = pts_clip[..., :2] / (pts_w + 1e-6)
        
        # NDC to Pixel Coordinates
        pts_px = (pts_ndc + 1.0) * 0.5 * torch.tensor([W, H], device=mvp.device)
        
        pixel_log_probs = torch.zeros((num_samples, B, H, W, 1), device=mvp.device)
        
        # neighborhood coverage: controls how "wide" the tetrahedron proxy is in pixels.
        # radius = 1 corresponds to a 3x3 footprint per vertex, easily filling most tetrahedral interiors.
        radius = 1 
        
        for b in range(B):
            px = pts_px[b, :, 0]
            py = pts_px[b, :, 1]
            w = pts_w[b, :, 0]
            
            # Keep vertices theoretically in view
            valid = (w > 0.01) & (px >= -W) & (px < 2*W) & (py >= -H) & (py < 2*H)
            
            px_int = torch.round(px[valid]).long()
            py_int = torch.round(py[valid]).long()
            
            valid_log_probs = log_probs_samples[:, valid] # (num_samples, V_valid)
            
            pixel_log_probs_flat = torch.zeros((num_samples, H * W), device=mvp.device)
            
            # Splat the vertices into image pixels
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx = px_int + dx
                    ny = py_int + dy
                    
                    # Prevent out-of-bounds writes
                    mask = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
                    
                    if not mask.any():
                        continue
                        
                    flat_idx = ny[mask] * W + nx[mask]
                    log_p = valid_log_probs[:, mask]
                    
                    # Scatter_add log probabilities seamlessly
                    # Note: Since explicit unique mapping logic ensures uniqueness spatially, 
                    # one vertex contributes precisely ONCE to each distinct pixel in its footprint.
                    pixel_log_probs_flat.scatter_add_(1, flat_idx.unsqueeze(0).expand(num_samples, -1), log_p)
                    
            pixel_log_probs[:, b, :, :, 0] = pixel_log_probs_flat.view(num_samples, H, W)
            
        return pixel_log_probs

    def _layer_opacity(self, opt_mesh, rast, gb_pos, hashgrid_features):
        """Per-pixel opacity alpha in [0,1] for one depth-peeled layer.

        opacity_source='tetrahedral_grid': interpolate the per-vertex opacity
        feature across the triangle (dr.interpolate) -- the "blended vertex
        quality". 'hashgrid': query the opacity feature at the surface point.
        An optional self.opacity_model head maps the feature to a logit; else the
        (dim-1) feature IS the logit. Returns (N,H,W,1) BEFORE the hit mask."""
        src = self.config.mesh_transparency.get('opacity_source', 'tetrahedral_grid')
        if src == 'tetrahedral_grid':
            if 'opacity' not in opt_mesh.v_feature:
                return (rast[..., -1:] > 0).float()
            logit = self._interpolate(opt_mesh.v_feature['opacity'][None, ...], rast, opt_mesh.t_pos_idx.int())[0]
        else:
            logit = self.get_feature('opacity', gb_pos, hashgrid_features)
        if getattr(self, 'opacity_model', None) is not None:
            logit = self.opacity_model(logit)
        return torch.sigmoid(logit)

    def _shade_layer(self, opt_mesh, rast, view_pos, t_dirs):
        """Shade a single rasterized layer (one offset sample). Mirrors the
        interpolate -> hashgrid -> appearance block of render_mesh_instance."""
        faces = opt_mesh.t_pos_idx.int()
        gb_pos, _ = self._interpolate(opt_mesh.v_pos[None, ...], rast, faces)
        v0 = opt_mesh.v_pos[opt_mesh.t_pos_idx[:, 0], :]
        v1 = opt_mesh.v_pos[opt_mesh.t_pos_idx[:, 1], :]
        v2 = opt_mesh.v_pos[opt_mesh.t_pos_idx[:, 2], :]
        fn = util.safe_normalize(torch.cross(v1 - v0, v2 - v0))
        fni = torch.arange(0, fn.shape[0], dtype=torch.int64, device='cuda')[:, None].repeat(1, 3)
        gb_geom, _ = self._interpolate(fn[None, ...], rast, fni.int())
        gb_geom = util.safe_normalize(gb_geom)
        gb_normal, _ = self._interpolate(opt_mesh.v_nrm[None, ...], rast, opt_mesh.t_nrm_idx.int())
        gb_tangent, _ = self._interpolate(opt_mesh.v_tng[None, ...], rast, opt_mesh.t_tng_idx.int())
        if self.config.normals.get('smooth_mesh_normals', False):
            render_normals = ru.prepare_shading_normal(gb_pos, view_pos, None, gb_normal, gb_tangent, gb_geom,
                                                       two_sided_shading=(not self.backface_culling), opengl=True)
        else:
            render_normals = gb_geom
        hashgrid_features = self.hashgrid(gb_pos) if self.hashgrid is not None else None
        tdirs = t_dirs if t_dirs is not None else F.normalize((gb_pos - view_pos).detach(), dim=-1)
        if self.config.appearance.get("interpolate_mesh", False):
            # grid-stored appearance: interpolate vertex features via the
            # rasterizer instead of positional queries (which would need the
            # point-in-tet tracer) -- mirrors render_mesh_instance
            appearance_inputs = {}
            for name in self.config.appearance.input_parameters:
                appearance_inputs[name] = self._interpolate(opt_mesh.v_feature[name][None, ...], rast, faces)[0]
        else:
            appearance_inputs = self._collect_features(self.config.appearance.input_parameters, gb_pos, hashgrid_features)
        z_shape = (*gb_pos.shape[:-1], 1)
        colors = self.appearance_model(dirs=self._appearance_dirs(tdirs),
                                       normals=self._shading_normals(gb_pos, render_normals),
                                       positions=gb_pos,
                                       sdf=torch.zeros(z_shape, device=gb_pos.device), **appearance_inputs)
        depth = torch.norm(gb_pos - view_pos, dim=-1, keepdim=True).clamp(min=1e-12)
        hit = (rast[..., -1:] > 0).float()
        alpha = self._layer_opacity(opt_mesh, rast, gb_pos, hashgrid_features) * hit
        return {'color': colors, 'alpha': alpha, 'depth': depth, 'gb_pos': gb_pos,
                'normal': gb_normal, 'geom': gb_geom, 'tangent': gb_tangent, 'hit': hit}

    def render_mesh_layered(self, opt_mesh, mvp, resolution, view_pos, t_dirs,
                            num_offset_samples, spp, background):
        """Depth-peeled translucent render: `num_layers` layers, front-to-back
        alpha compositing WITH the background. Returns composited color / opacity
        / expected-depth (+ front-layer normals/gb_pos), same keys as the opaque
        path so systems/losses are unchanged. num_layers=1, alpha==1 reproduces
        render_mesh_instance exactly."""
        cfg = self.config.mesh_transparency
        L = int(cfg.get('num_layers', 4))
        depth_mode = cfg.get('depth_mode', 'expected')
        full_res = [resolution[0] * spp, resolution[1] * spp]
        if background is None:
            bg = self.get_background_color(t_dirs)
        elif spp > 1:
            bg = util.scale_img_nhwc(background, full_res, mag='nearest', min='nearest')
        else:
            bg = background
        N = bg.shape[0]
        z = lambda c: torch.zeros(N, resolution[0], resolution[1], c, device='cuda')
        buffers = {'rgb': bg, 'opacity': z(1), 'depth': z(1), 'normal': z(3),
                   'geometric_normal': z(3), 'tangent': z(3), 'gb_pos': z(3)}
        v_pos_clip = ru.xfm_points(opt_mesh.v_pos[None, ...], mvp)
        if v_pos_clip.size()[1] == 0:
            return buffers
        faces = opt_mesh.t_pos_idx.int()
        Hs, Ws = full_res
        T = torch.ones(N, Hs, Ws, 1, device='cuda')
        C = torch.zeros(N, Hs, Ws, 3, device='cuda')
        D = torch.zeros(N, Hs, Ws, 1, device='cuda')
        O = torch.zeros(N, Hs, Ws, 1, device='cuda')
        bg_full = bg if spp == 1 else util.scale_img_nhwc(bg, full_res, mag='nearest', min='nearest')
        front = None
        with dr.DepthPeeler(self.glctx, v_pos_clip, faces, full_res) as peeler:
            for i in range(L):
                rast, _ = peeler.rasterize_next_layer()
                s = self._shade_layer(opt_mesh, rast, view_pos, t_dirs)
                color = dr.antialias(s['color'].contiguous(), rast, v_pos_clip, faces)
                alpha = dr.antialias(s['alpha'].contiguous(), rast, v_pos_clip, faces).clamp(0.0, 1.0)
                w = T * alpha
                C = C + w * color
                D = D + w * s['depth']
                O = O + w
                T = T * (1.0 - alpha)
                if i == 0:
                    front = s
        C = C + T * bg_full                                  # residual transmittance sees the bg
        depth_out = D / O.clamp(min=1e-6) if depth_mode == 'expected' else front['depth']
        out = {'rgb': C, 'opacity': O, 'depth': depth_out, 'normal': front['normal'],
               'geometric_normal': front['geom'], 'tangent': front['tangent'], 'gb_pos': front['gb_pos']}
        if spp > 1:
            for k in out:
                out[k] = util.avg_pool_nhwc(out[k], spp)
        buffers.update(out)
        return buffers

    def render_mesh(
            self,
            mvp,
            view_pos,
            resolution,
            rays = None,
            background = None,
            num_offset_samples = None,
            spp=1, 
            sample_sdf = False,
            num_samples = 1):
        
        view_pos = view_pos[:, None, None, :] if len(view_pos.shape) == 2 else view_pos
        t_dirs = rays[..., 3:].view(mvp.shape[0], resolution[0], resolution[1], 3) if rays is not None else None
        t_dirs = F.normalize(t_dirs, dim=-1) if t_dirs is not None else None

        if (sample_sdf):
            opt_meshes, _ = self.sampleMesh(num_samples=num_samples)

            # 1. Render all meshes (each mesh carries face_tet_idx + log_prob_sign,
            #    so render_mesh_instance fills 'topology_log_p' per pixel).
            buffers_list = []
            for opt_mesh in opt_meshes:
                buffers_list.append(self.render_mesh_instance(opt_mesh, mvp, resolution, view_pos, t_dirs, num_offset_samples, spp, background))

            # Stack results: (num_samples, ...)
            stacked_buffers = {}
            for key in buffers_list[0].keys():
                if torch.is_tensor(buffers_list[0][key]):
                    stacked_buffers[key] = torch.stack([b[key] for b in buffers_list], dim=0)
                else:
                    stacked_buffers[key] = [b[key] for b in buffers_list]

            # Per-pixel topology log-prob, shape (N, B, H, W, 1). One scalar per
            # (sample, pixel) — log P(tau_{Tv(p,i)} | mu, sigma). Consumed by the
            # REINFORCE+LOO surrogate in the system; pathwise gradients flow
            # through stacked_buffers['rgb'] etc. as today.
            pixel_log_p = stacked_buffers.pop('topology_log_p',
                                              torch.zeros((num_samples, mvp.shape[0], resolution[0], resolution[1], 1),
                                                          device=mvp.device))

            return stacked_buffers, pixel_log_p
        else:
            opt_mesh = self.getMesh()
            if self.config.get('mesh_transparency', {}).get('enabled', False):
                buffers = self.render_mesh_layered(opt_mesh, mvp, resolution, view_pos, t_dirs, num_offset_samples, spp, background)
            else:
                buffers = self.render_mesh_instance(opt_mesh, mvp, resolution, view_pos, t_dirs, num_offset_samples, spp, background)
            return buffers

    def render_mesh_instance(
        self, 
        opt_mesh, 
        mvp, 
        resolution, 
        view_pos, 
        t_dirs, 
        num_offset_samples, 
        spp, 
        background,
        features_to_interpolate = []):

        v_pos_clip = ru.xfm_points(opt_mesh.v_pos[None, ...], mvp)
        # Initialize default values
        full_res = [resolution[0]*spp, resolution[1]*spp]
        if background is None:
            background_instance = self.get_background_color(t_dirs)
        elif spp > 1:
            background_instance = util.scale_img_nhwc(background, full_res, mag='nearest', min='nearest')
        else:
            background_instance = background

        # Initialize buffers with zero/background values. Use the batch size N
        # from background_instance (= number of views) so that when the mesh
        # rasterizes NO geometry (degenerate/collapsed mesh, v_pos_clip empty)
        # the returned buffers still have a consistent N. Otherwise opacity etc.
        # default to N=1 while the batch is N=train_num_images, which crashes the
        # per-pixel losses (e.g. mask `expand_as`). With consistent N the mask
        # loss instead sees all-zero opacity and pushes the surface back.
        N = background_instance.shape[0]
        buffers = {
            'rgb': background_instance,
            'opacity': torch.zeros(N, resolution[0], resolution[1], 1, device='cuda'),
            'depth': torch.zeros(N, resolution[0], resolution[1], 1, device='cuda'),
            'normal': torch.zeros(N, resolution[0], resolution[1], 3, device='cuda'),
            'geometric_normal': torch.zeros(N, resolution[0], resolution[1], 3, device='cuda'),
            'tangent': torch.zeros(N, resolution[0], resolution[1], 3, device='cuda'),
        }
        # Topology bookkeeping: only populated when the mesh carries it (i.e.
        # came from sampleMesh with a probabilistic SDF). Background pixels
        # contribute log_p = 0 to the score-function term.
        track_topology = hasattr(opt_mesh, 'face_tet_idx') and hasattr(opt_mesh, 'log_prob_sign')
        if track_topology:
            buffers['topology_log_p'] = torch.zeros(N, resolution[0], resolution[1], 1, device='cuda')

        if (v_pos_clip.size()[1] != 0):
            rast, rast_deriv = dr.rasterize(self.glctx, v_pos_clip, opt_mesh.t_pos_idx.int(), full_res)
            gb_pos, _ = self._interpolate(opt_mesh.v_pos[None, ...], rast, opt_mesh.t_pos_idx.int())
            
            # Compute geometric normals
            v0 = opt_mesh.v_pos[opt_mesh.t_pos_idx[:, 0], :]
            v1 = opt_mesh.v_pos[opt_mesh.t_pos_idx[:, 1], :]
            v2 = opt_mesh.v_pos[opt_mesh.t_pos_idx[:, 2], :]
            face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0))
            face_normal_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
            gb_geometric_normal, _ = self._interpolate(face_normals[None, ...], rast, face_normal_indices.int())
            gb_geometric_normal = util.safe_normalize(gb_geometric_normal)

            # Interpolate vertex normals, tangents, and texture coords (currently not used, only used in 2D textured case in the future)
            assert opt_mesh.v_nrm is not None and opt_mesh.v_tng is not None and opt_mesh.v_tex is not None
            gb_normal, _ = self._interpolate(opt_mesh.v_nrm[None, ...], rast, opt_mesh.t_nrm_idx.int())
            gb_tangent, _ = self._interpolate(opt_mesh.v_tng[None, ...], rast, opt_mesh.t_tng_idx.int())
            gb_texc, gb_texc_deriv = self._interpolate(opt_mesh.v_tex[None, ...], rast, opt_mesh.t_tex_idx.int(), rast_db=rast_deriv)
            
            # Prepare shading normal (handles two-sided shading, local frame, etc.)
            #render_normals = gb_normal if self.config.normals.get('smooth_mesh_normals', False) else gb_geometric_normal
            if self.config.normals.get('smooth_mesh_normals', False):
                render_normals = ru.prepare_shading_normal(gb_pos, view_pos, None, gb_normal, gb_tangent, gb_geometric_normal, two_sided_shading=(not self.backface_culling), opengl=True)
            else:
                render_normals = gb_geometric_normal

            # Compute hashgrid features (deferred to the branches: with
            # appearance.shade_chunk the shading path re-encodes per chunk, so
            # the eager full-size call would only size tcnn's arena for nothing)
            _chunk = int(self.config.appearance.get('shade_chunk', 0) or 0)
            hashgrid_features = None

            # Fallback for ray directions if not provided
            if t_dirs is None:
                t_dirs = F.normalize((gb_pos - view_pos).detach(), dim=-1)

            # Offset sampling
            num_offset_samples = self.num_mesh_offset_samples if num_offset_samples is None else num_offset_samples
            if self.config.mesh_offset_sampling.get('vary_sample_number', False):
                num_offset_samples = torch.randint(1, num_offset_samples + 1, (1,)).item()
            
            if num_offset_samples > 1:
                cos = (t_dirs * gb_geometric_normal).sum(-1, keepdim=True).abs().clamp(min=self.config.mesh_offset_sampling.get('cos_min_clamp', 0.5)) if self.config.mesh_offset_sampling.get('cosine_scaling', True) else 1.0
                if self.hashgrid is not None:
                    hashgrid_features = self.hashgrid(gb_pos)
                inv_s = self.get_inv_s(gb_pos, hashgrid_features)
                gb_pos_eval, weights = self.sample_offset(gb_pos, view_pos, t_dirs, num_offset_samples, cos, inv_s)
                if self.hashgrid is not None and _chunk == 0:
                    hashgrid_features = self.hashgrid(gb_pos_eval)
            else:
                gb_pos_eval = gb_pos
                weights = None
                if self.hashgrid is not None and _chunk == 0:
                    hashgrid_features = self.hashgrid(gb_pos)

            # Appearance model
            if self.config.appearance.get("interpolate_mesh", False) and num_offset_samples == 1:
                appearance_inputs = {}
                for name in self.config.appearance.input_parameters:
                    appearance_inputs[name] = self._interpolate(opt_mesh.v_feature[name][None, ...], rast, opt_mesh.t_pos_idx.int())[0]
            else:
                appearance_inputs = {} if _chunk > 0 else self._collect_features(self.config.appearance.input_parameters, gb_pos_eval, hashgrid_features)
            t_dirs_eval = t_dirs.unsqueeze(-2).expand(*gb_pos_eval.shape) if (t_dirs is not None and num_offset_samples > 1) else t_dirs
            render_normals_eval = render_normals.unsqueeze(-2).expand(*gb_pos_eval.shape) if (render_normals is not None and num_offset_samples > 1) else render_normals
            z_shape = (*gb_pos_eval.shape[:-1], 1)

            #Interpolate additional features
            mesh_features = {}
            for feat_name in features_to_interpolate:
                feat_val = self._interpolate(opt_mesh.v_feature[feat_name][None, ...], rast, opt_mesh.t_pos_idx.int())[0]
                mesh_features[feat_name] = feat_val
                    

            # Compute colors
            if _chunk > 0:
                # Chunk the shading path (hashgrid re-encode + appearance MLP)
                # so tcnn's workspace arena and the held activations are capped
                # at chunk size instead of the full point count.
                _lead = gb_pos_eval.shape[:-1]
                _pos = gb_pos_eval.reshape(-1, gb_pos_eval.shape[-1])
                _dir = self._appearance_dirs(t_dirs_eval).reshape(_pos.shape[0], -1)
                _nrm = self._shading_normals(gb_pos_eval, render_normals_eval).reshape(_pos.shape[0], -1)
                def _shade_chunk_fn(_pc, _dc, _nc):
                    _hf = self.hashgrid(_pc) if self.hashgrid is not None else None
                    _ai = self._collect_features(self.config.appearance.input_parameters, _pc, _hf)
                    return self.appearance_model(
                        dirs=_dc, normals=_nc, positions=_pc,
                        sdf=torch.zeros((_pc.shape[0], 1), device=_pc.device), **_ai)
                _ckpt = bool(self.config.appearance.get('shade_chunk_checkpoint', False))
                _outs = []
                for _s in range(0, _pos.shape[0], _chunk):
                    _e = min(_s + _chunk, _pos.shape[0])
                    if _ckpt and torch.is_grad_enabled():
                        # free each chunk's activations after forward; recompute
                        # them during backward (exact, ~2x shading forward cost)
                        _outs.append(torch.utils.checkpoint.checkpoint(
                            _shade_chunk_fn, _pos[_s:_e], _dir[_s:_e], _nrm[_s:_e],
                            use_reentrant=False))
                    else:
                        _outs.append(_shade_chunk_fn(_pos[_s:_e], _dir[_s:_e], _nrm[_s:_e]))
                colors = torch.cat(_outs, 0).reshape(*_lead, -1)
            else:
                colors = self.appearance_model(dirs=self._appearance_dirs(t_dirs_eval), normals=self._shading_normals(gb_pos_eval, render_normals_eval), positions=gb_pos_eval, sdf=torch.zeros(z_shape, device=gb_pos_eval.device), **appearance_inputs)
            shaded_col = (colors * weights[..., None]).sum(dim=-2) if num_offset_samples > 1 else colors
            opacity = (rast[..., -1:] > 0).float()
            if self.backface_culling:
                # nvdiffrast has no native backface-cull flag — mask alpha post-rasterization
                # using the geometric face normal vs. the camera-to-surface direction.
                view_dir_to_surface = F.normalize(gb_pos - view_pos, dim=-1)
                front_facing = ((view_dir_to_surface * gb_geometric_normal).sum(-1, keepdim=True) < 0).float()
                opacity = opacity * front_facing

            render_buffers = {
                'rgb'    : shaded_col,
                'gb_pos' : gb_pos,
                'depth':  torch.norm(gb_pos - view_pos, dim=-1, keepdim=True).clamp(min=1e-12),
                'normal'    : gb_normal,
                'geometric_normal': gb_geometric_normal,
                'tangent'   : gb_tangent,
                'opacity'   : opacity,
            }
            render_buffers.update(mesh_features)

            # Post-processing (antialiasing, pooling)
            for key in render_buffers:
                render_buffers[key] = dr.antialias(render_buffers[key].contiguous(), rast, v_pos_clip, opt_mesh.t_pos_idx.int())
                if spp > 1:
                    render_buffers[key] = util.avg_pool_nhwc(render_buffers[key], spp)

            # Merge results and blend background
            for key, val in render_buffers.items():
                if key == 'rgb':
                    buffers['rgb'] = torch.lerp(background_instance, val, render_buffers['opacity'])
                elif key == 'opacity':
                    buffers[key] = val
                else:
                    buffers[key] = torch.lerp(torch.zeros_like(val), val, render_buffers['opacity'])


            # Per-pixel topology log-probability (REINFORCE term). Computed
            # OUTSIDE the antialias loop on purpose: each visible pixel maps
            # to exactly one source tet, and we sum the log-prob of that
            # tet's four vertex signs. Antialiasing would smear log-probs
            # across silhouettes between different tets, so we keep this as
            # a hard per-pixel quantity. Background pixels keep log_p = 0.
            if track_topology:
                tri_id_1based = rast[..., 3].long()           # (B, H, W)
                valid = tri_id_1based > 0
                tri_idx = (tri_id_1based - 1).clamp(min=0)    # (B, H, W) safe indexer
                tet_idx = opt_mesh.face_tet_idx.long()[tri_idx]
                tet_v4 = self.tetrahedral_grid.indices.long()[tet_idx]   # (B, H, W, 4)
                log_p_per_vert = opt_mesh.log_prob_sign[tet_v4]          # (B, H, W, 4, D)
                pixel_log_p = log_p_per_vert.sum(dim=(-2, -1), keepdim=True).squeeze(-1)   # (B, H, W, 1)
                pixel_log_p = torch.where(valid.unsqueeze(-1), pixel_log_p, torch.zeros_like(pixel_log_p))
                if spp > 1:
                    pixel_log_p = util.avg_pool_nhwc(pixel_log_p, spp)
                buffers['topology_log_p'] = pixel_log_p

        return buffers

    def get_alpha(self, sdf, inv_s, normal, dirs, dists):
        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha

    
    def sample_points_occ_grid(self, rays_o, rays_d):

        with torch.no_grad():
            #max_intersected_triangles = min(256, 1 << int(self.avg_intersected_triangles + 15).bit_length())
            if self.config.contraction_type == 'aabb':
                ray_indices, t_starts, t_ends = ray_marching(
                    rays_o, rays_d,
                    scene_aabb=self.scene_aabb,
                    grid=self.occupancy_grid if self.config.grid_prune else None,
                    alpha_fn=None,
                    near_plane=None, far_plane=None,
                    render_step_size=self.render_step_size,
                    stratified=self.randomized,
                    cone_angle=0.0,
                    alpha_thre=0.0
                )
            else:
                ray_indices, t_starts, t_ends = ray_marching(
                    rays_o, rays_d,
                    #scene_aabb=self.scene_aabb,
                    grid=self.occupancy_grid if self.config.grid_prune else None,
                    alpha_fn=None,
                    near_plane=None, far_plane=self.config.get('unbounded_far_plane', 1000.0),
                    # cone marching: step grows with distance so far-field rays
                    # stay tractable; base step near the cameras. NOTE: /10 here
                    # produced ~6k samples/ray -> OOM in the full-image validation
                    # render after densification; unbounded_step_div (default 2)
                    # keeps it tractable.
                    render_step_size=self.render_step_size / self.config.get('unbounded_step_div', 2.0),
                    stratified=self.randomized,
                    cone_angle=self.config.get('cone_angle', 0.004),
                    alpha_thre=0.0
                )

            
        return ray_indices, t_starts, t_ends

    def get_background_color(self, dirs):
        if self.learned_background:
            if self.config.get('background_angle_input', False):
                #Map viewing direction into [0,1]^2
                theta = torch.acos(dirs[...,1]) / math.pi  # [0, pi] -> [0,1]
                phi = (torch.atan2(dirs[...,2], dirs[...,0]) + math.pi) / (2*math.pi)  # [-pi, pi] -> [0,1]
                bg_input = torch.stack([theta, phi], dim=-1)
                _, background = self.background_network(bg_input, scale_output = False, with_grad = False)
            else:
                background = self.background_network(dirs)
        else:
            background = self.default_background_color.to(dirs.device)[None,:].expand(*dirs.shape)
        return background
            

    def render_rays(self, rays, background = None):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
        rays_d = F.normalize(rays_d, dim=-1) # Normalize rays_d as requested
        ray_indices, t_starts, t_ends = self.sample_points_occ_grid(rays_o, rays_d)
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        dists = t_ends - t_starts

        # Enable gradient tracking for positions
        with torch.enable_grad():
            positions.requires_grad_(True)
            
            if self.hashgrid is not None:
                hashgrid_features = self.hashgrid(positions)
            else:
                hashgrid_features = None
            
            sdf_inputs = self._collect_features(self.config.sdf.input_parameters, positions, hashgrid_features)
            sdf = self.sdf_model(**sdf_inputs)
            
            sdf_grad = torch.autograd.grad(
                sdf, positions, grad_outputs=torch.ones_like(sdf),
                create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True
            )[0]

        geometric_normals = F.normalize(sdf_grad, p=2, dim=-1)
        normals = geometric_normals

        appearance_inputs = self._collect_features(self.config.appearance.input_parameters, positions, hashgrid_features)
        # shading normal (perturbed by the learned normal texture) only feeds
        # the appearance model; get_alpha below keeps the geometric normal
        colors = self.appearance_model(dirs=t_dirs, normals=self._shading_normals(positions, normals), positions=positions, sdf=sdf.unsqueeze(-1), **appearance_inputs)
        
        variance_inputs = self._collect_features(self.config.variance.input_parameters, positions, hashgrid_features)
        inv_s = self.variance_model(**variance_inputs)

        #Get colors 
        alpha = self.get_alpha(sdf, inv_s, normals, t_dirs, dists)[...,None]
        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=colors, n_rays=n_rays)
        comp_normal = accumulate_along_rays(weights, ray_indices, values=normals, n_rays=n_rays)
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)

        out = {
            'comp_rgb_fg': comp_rgb,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(midpoints)], dtype=torch.int32, device=rays.device)
        }

        if self.training:
            out.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad,
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': dists.view(-1),
                'ray_indices': ray_indices.view(-1)                
            })

        if background is None:
            background = self.get_background_color(rays_d)

        return {
            **out,
            'comp_rgb_bg': background,
            'comp_rgb_full': out['comp_rgb_fg'] + background * (1.0 - out['opacity'])
        }
    
    #-------------------------------------
    # Regularizations
    #-------------------------------------
    def mesh_angle_loss(self):
        mesh = self.getMesh()
        angles = mesh.triangle_cosine_angles()
        loss = ((angles - 0.5) ** 2).mean()
        return loss

    # -------------------------------------------------------------------------
    # Mesh-quality regularizers (fight sliver triangles from marching-tets).
    # All operate on the differentiable extracted mesh / grid tets, so their
    # gradient flows back through marching_tets into the SDF (and primal_points
    # for the tet term), reshaping geometry toward better-conditioned elements.
    # Pass a pre-extracted `mesh` to avoid re-running getMesh() per term.
    # -------------------------------------------------------------------------
    @staticmethod
    def _tri_geom(mesh):
        f = mesh.t_pos_idx.long()
        v0, v1, v2 = mesh.v_pos[f[:, 0]], mesh.v_pos[f[:, 1]], mesh.v_pos[f[:, 2]]
        e0, e1, e2 = v1 - v0, v2 - v1, v0 - v2
        l2 = (e0 * e0).sum(-1) + (e1 * e1).sum(-1) + (e2 * e2).sum(-1)   # sum of squared edge lengths [F]
        area = 0.5 * torch.linalg.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)  # [F]
        return e0, e1, e2, l2, area

    def mesh_triangle_quality_loss(self, mesh=None):
        """Sliverness of extracted triangles. Q = 4*sqrt(3)*Area / (sum sq edge len),
        which is 1 for an equilateral triangle and -> 0 for a sliver/needle. Penalise
        (1 - Q): pushes each triangle toward well-conditioned, directly targeting slivers."""
        if mesh is None:
            mesh = self.getMesh()
        _, _, _, l2, area = self._tri_geom(mesh)
        Q = (4.0 * math.sqrt(3.0) * area) / (l2 + 1e-12)
        return (1.0 - Q.clamp(max=1.0)).mean()

    def mesh_edge_size_loss(self, mesh=None):
        """Size of extracted triangles: soft-cap edges longer than `edge_size_factor`
        x the (detached) median edge length. Scale-relative, so it only fights the
        oversized triangles without shrinking the whole mesh."""
        if mesh is None:
            mesh = self.getMesh()
        e0, e1, e2, _, _ = self._tri_geom(mesh)
        el = torch.cat([e0.norm(dim=-1), e1.norm(dim=-1), e2.norm(dim=-1)], 0)
        factor = float(self.config.get('mesh_reg', {}).get('edge_size_factor', 3.0))
        thr = el.detach().median() * factor
        return F.relu(el - thr).pow(2).mean()

    def mesh_small_area_loss(self, mesh=None):
        """One-sided, LOCALLY-relative small-triangle penalty (anti-sliver dust):
        penalise triangles whose area is below `small_area_tau` x the mean area
        of their local neighbourhood (vertex-1-ring average, DETACHED). One-sided
        -> normal/large triangles feel nothing and refinement stays possible when
        a whole neighbourhood refines together; local reference -> spatially
        varying triangle density is untouched; detached reference -> gradient
        only grows the runt triangle, never shrinks its neighbours."""
        if mesh is None:
            mesh = self.getMesh()
        _, _, _, _, area = self._tri_geom(mesh)                    # (F,)
        Fi = mesh.t_pos_idx.long()                                 # (F,3)
        nv = mesh.v_pos.shape[0]
        dev = area.device
        vsum = torch.zeros(nv, device=dev).index_add_(
            0, Fi.reshape(-1), area[:, None].expand(-1, 3).reshape(-1))
        vcnt = torch.zeros(nv, device=dev).index_add_(
            0, Fi.reshape(-1), torch.ones_like(area[:, None].expand(-1, 3).reshape(-1)))
        vmean = vsum / vcnt.clamp_min(1.0)
        a_local = vmean[Fi].mean(dim=1).detach()                   # (F,) local scale
        tau = float(self.config.get('mesh_reg', {}).get('small_area_tau', 0.2))
        r = area / (tau * a_local + 1e-14)
        return F.relu(1.0 - r).pow(2).mean()

    def mesh_laplacian_loss(self, mesh=None):
        """Uniform-Laplacian smoothing on the extracted mesh: penalise each vertex's
        offset from the centroid of its 1-ring. Smooths the surface and, as a side
        effect, evens out triangle shapes."""
        if mesh is None:
            mesh = self.getMesh()
        v, f = mesh.v_pos, mesh.t_pos_idx.long()
        src = torch.cat([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
        dst = torch.cat([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
        N = v.shape[0]
        deg = torch.zeros(N, device=v.device, dtype=v.dtype).index_add_(0, src, torch.ones_like(src, dtype=v.dtype))
        nbr_sum = torch.zeros_like(v).index_add_(0, src, v[dst])
        lap = v - nbr_sum / deg.clamp_min(1.0).unsqueeze(-1)
        return (lap * lap).sum(-1).mean()

    def tet_quality_loss(self):
        """Regularity of the grid tetrahedra themselves (their shape drives where the
        marching-tets crossings land). Q_tet = C * V^(2/3) / (sum sq edge len), 1 for a
        regular tet, -> 0 for a sliver tet. Gradient flows to primal_points."""
        g = self.tetrahedral_grid
        p = g.primal_points_uncontracted
        tv = p[g.indices]                                   # [T,4,3]
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        l2 = sum(((tv[:, a] - tv[:, b]) ** 2).sum(-1) for a, b in pairs)   # [T]
        a = tv[:, 1] - tv[:, 0]; b = tv[:, 2] - tv[:, 0]; c = tv[:, 3] - tv[:, 0]
        vol = (torch.linalg.cross(a, b, dim=-1) * c).sum(-1).abs() / 6.0   # [T]
        C = (6.0 * math.sqrt(2.0)) ** (2.0 / 3.0) * 6.0                    # regular-tet normaliser -> Q=1
        Q = C * vol.clamp_min(1e-9) ** (2.0 / 3.0) / (l2 + 1e-12)
        return (1.0 - Q.clamp(max=1.0)).mean()

    def sdf_range_regularization(self):
        sdf = self.grid_sdf
        # penalize values outside of [-0.1, 0.1]
        loss = F.relu(torch.abs(sdf) - 0.1).pow(2).mean()
        return loss
    
    #-------------------------------------
    # Pytorch Lightning methods
    #-------------------------------------

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()

    @torch.no_grad()
    def export(self, export_config):
        pass
        # mesh = self.isosurface()
        # if export_config.export_vertex_color:
        #     _, sdf_grad, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), with_grad=True, with_feature=True)
        #     normal = F.normalize(sdf_grad, p=2, dim=-1)
        #     rgb = self.texture(feature, -normal, normal) # set the viewing directions to the normal to get "albedo"
        #     mesh['v_rgb'] = rgb.cpu()
        # return mesh