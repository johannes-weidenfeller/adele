import models
from models.base import BaseModel
import torch
import radfoam
import torch
import numpy as np
from torch import nn
import radfoam
from models.utils import *
from utils.misc import get_rank
from plyfile import PlyData, PlyElement
import tqdm
import torch
from scipy.spatial import cKDTree
import open3d as o3d


def triangulation_with_retry(points, tries=12, wait=25):
    """radfoam.Triangulation makes one large raw cudaMalloc. On this shared box
    a neighbour process can land on the same GPU between our process start and
    this call (the picker's free-memory snapshot is stale within seconds), so
    the allocation transiently fails with 'out of memory' even though the GPU
    was empty at launch. Our process HOLDS its other allocations, so retrying
    in-process wins as soon as a window opens -- unlike process-level retries,
    which pay full startup and re-enter the same race."""
    import time
    for i in range(tries):
        try:
            return radfoam.Triangulation(points)
        except RuntimeError as e:
            if 'out of memory' not in str(e) or i == tries - 1:
                raise
            print(f"[grid] Triangulation OOM (attempt {i + 1}/{tries}); "
                  f"neighbour-allocation race, retrying in {wait}s", flush=True)
            torch.cuda.empty_cache()
            time.sleep(wait)


def filter_random_points_knn(points, random_outside_points, threshold=0.01):
    """
    Efficiently filter random points that are within `threshold`
    of any input point using PyTorch3D KNN.
    """
    # Find nearest neighbor distance from each random point to the input points
    knn = knn_points(random_outside_points[None], points[None], K=1)
    # knn.dists shape: [1, M, 1] — squared distances
    min_dists = knn.dists.squeeze()
    keep_mask = min_dists > threshold ** 2
    return random_outside_points[keep_mask]

@models.register('tetrahedral_grid')
class TetrahedralGrid(BaseModel):
    def setup(self):

        with torch.cuda.device(get_rank()):

            self.feature_names = []
            self.feature_limits = {}
            self.normalize_feature = {}
            self.feature_grad_accum = {}
            # Stored copies of each feature's spec so we can re-initialize it
            # in-place at a fixed cadence (see `reset_every_n_steps`). Keeping
            # the original Parameter object alive on reset is critical: the
            # main optimizer holds a reference into `param_groups`, and any
            # densification/pruning code in this class has built up state
            # against that exact tensor — replacing it would silently break.
            self.feature_configs = {}
            self.feature_reset_intervals = {}
            
            self.scale = torch.tensor(self.config.scale, dtype=torch.float32, device='cuda')
            self.offset = torch.tensor(self.config.offset, dtype=torch.float32, device='cuda') 
            self.roi = torch.tensor(
                [
                    self.offset[0] - self.scale,
                    self.offset[1] - self.scale,
                    self.offset[2] - self.scale,
                    self.offset[0] + self.scale,
                    self.offset[1] + self.scale,
                    self.offset[2] + self.scale,
                ],
                dtype=torch.float32,
                device='cuda'
            )
            self.contraction_type = get_contraction_type(self.config.get('contraction_type', 'aabb'))

            #Initialize points
            self._tetrahedra_tracer = None
            self.recompute_primal_points_uncontracted = True
            if self.config.point_initialization == "random":
                self.initialize_points_random()
            elif self.config.point_initialization == "grid":
                self.initialize_points_from_grid()
            elif self.config.point_initialization == "points":
                self.initialize_points_from_pcd(file_path = self.config.initial_points_file)
            elif self.config.point_initialization == "seed_spawn":
                self.initialize_points_from_seeds(file_path = self.config.initial_points_file)
            self.update_triangulation(rebuild=False)
            self.point_grad_accum = torch.zeros(size = (self.primal_points.shape[0],), dtype = torch.float32, requires_grad = False, device = 'cuda')

            #Initialize features
            for name, feature_dict in self.config.get("features" , {}).items():
                self.initialize_feature(name, feature_dict)
            
            

    @property
    def tetrahedra_tracer(self):
        if self._tetrahedra_tracer is None:
            from tetranerf import cpp
            self._tetrahedra_tracer = cpp.TetrahedraTracer(self.primal_points.device)
            self._tetrahedra_tracer.load_tetrahedra(self.primal_points_uncontracted, self.indices.to(torch.int32))
        return self._tetrahedra_tracer
            

    def update_step(self, epoch, global_step):
        with torch.no_grad():
            self.recompute_primal_points_uncontracted = True
            if self.contraction_type == ContractionType.UN_BOUNDED_SPHERE:
                #Clamp primal points to 0.999 norm sphere
                norms = torch.norm(self.primal_points - 0.5, dim=-1)
                mask = norms > 0.4995
                self.primal_points[mask] = 0.5 + 0.4995 * (self.primal_points[mask] - 0.5) / norms[mask].unsqueeze(-1)

            if self._tetrahedra_tracer is not None:
                self._tetrahedra_tracer.update_points(self.primal_points_uncontracted)

            for fname in self.feature_names:
                data = getattr(self, fname)
                fmin, fmax = self.feature_limits[fname]
                if fmin is not None or fmax is not None:
                    data.clamp_(min=fmin, max=fmax)

            # Periodic feature reset. Triggered at multiples of the per-feature
            # interval, but never at step 0 (initialization already did that).
            if global_step > 0:
                for fname, interval in self.feature_reset_intervals.items():
                    if interval and (global_step % interval == 0):
                        self.reset_feature(fname)
                        print(f"[tetrahedral_grid] reset feature '{fname}' at step {global_step}")

    @property
    def features(self):
        features = {}
        for fname in self.feature_names:
            features[fname] = getattr(self, fname)
        return features
    
    @property
    def primal_points_uncontracted(self):
        if self.recompute_primal_points_uncontracted:
            with torch.enable_grad():
                self._primal_points_uncontracted = contract_inv(self.primal_points, self.roi, self.contraction_type)
            self.recompute_primal_points_uncontracted = False
        return self._primal_points_uncontracted

    #-------------------------------------
    # Point and feature initialization methods
    #-------------------------------------
    def initialize_feature(self, name, feature_dict):
        dim = feature_dict["dim"]
        
        # New initialization format using "init" dictionary
        init_config = feature_dict.get("init", {})
        mode = init_config.get("type", feature_dict.get("initialization", "constant"))
        
        if mode == "constant":
            init_val = init_config.get("value", feature_dict.get("initial_value", 0.0))
            setattr(self, name, nn.Parameter(torch.full((self.primal_points.shape[0], dim), init_val, dtype = torch.float32, device = 'cuda')))
        elif mode == "sphere":
            # Signed-distance-to-a-sphere init (negative inside, positive outside),
            # matching the network's sphere_init. Zero-crossing = sphere of `radius`
            # centred at `center`, evaluated in the *uncontracted* (model) frame.
            radius = float(init_config.get("radius", feature_dict.get("initial_value", 0.5)))
            center = init_config.get("center", [0.0, 0.0, 0.0])
            c = torch.as_tensor(center, dtype=torch.float32, device='cuda')
            p = self.primal_points_uncontracted
            d = torch.norm(p - c, dim=-1, keepdim=True) - radius   # (N,1)
            if dim > 1:
                d = d.expand(-1, dim).contiguous()
            setattr(self, name, nn.Parameter(d.to(dtype=torch.float32, device='cuda')))
        elif mode == "double_sphere":
            # union of a foreground sphere and an inward-facing background shell:
            # sdf(x) = min(|x| - r_fg, R_bg - |x|). Zero-crossings at r_fg (object
            # seed, outward normal) and R_bg (sky shell, normal facing the cameras).
            r_fg = float(init_config.get("radius", 1.0))
            r_bg = float(init_config.get("bg_radius", 8.0))
            center = init_config.get("center", [0.0, 0.0, 0.0])
            c = torch.as_tensor(center, dtype=torch.float32, device='cuda')
            p_unc = self.primal_points_uncontracted
            n = torch.norm(p_unc - c, dim=-1, keepdim=True)
            d = torch.minimum(n - r_fg, r_bg - n)
            if dim > 1:
                d = d.expand(-1, dim).contiguous()
            setattr(self, name, nn.Parameter(d.to(dtype=torch.float32, device='cuda')))
        elif mode == "depth_tsdf":
            # Coarse SDF from VGGT (or any) depth maps, fused DIRECTLY at the grid
            # points -- no mesh extraction, no pysdf. Robust replacement for the
            # 'mesh_sdf' init, which baked a non-watertight TSDF mesh via pysdf
            # (undefined inside/outside -> corrupted signs). See
            # scripts/vggt_tsdf_cache.py for the cache format. Sign convention
            # matches 'sphere'/'double_sphere': negative inside, positive in free
            # space.
            cache_path = init_config.get("cache", init_config.get("path", None))
            if cache_path is None:
                raise ValueError("depth_tsdf init requires 'cache' (path to the depth cache .npz).")
            d = self._tsdf_from_depth_cache(
                cache_path,
                trunc=float(init_config.get("trunc", 0.05)),
                conf_percentile=init_config.get("conf_percentile", None),
                empty_fill=init_config.get("empty_fill", None),
                chunk=int(init_config.get("chunk", 200000)),
            )
            if dim > 1:
                d = d.expand(-1, dim).contiguous()
            setattr(self, name, nn.Parameter(d.to(dtype=torch.float32, device='cuda')))
        elif mode == "uniform":
            init_val = init_config.get("mean", feature_dict.get("initial_value", 0.0))
            init_scale = init_config.get("std", feature_dict.get("initial_scale", 1.0))
            setattr(self, name, nn.Parameter((torch.rand((self.primal_points.shape[0], dim), dtype = torch.float32, device = 'cuda') - 0.5) * init_scale + init_val))
        elif mode == "gaussian" or mode == "random":
            init_val = init_config.get("mean", feature_dict.get("initial_value", 0.0))
            init_scale = init_config.get("std", feature_dict.get("initial_scale", 1.0))
            setattr(self, name, nn.Parameter(torch.randn((self.primal_points.shape[0], dim), dtype = torch.float32, device = 'cuda') * init_scale + init_val))
        elif mode == 'mesh_sdf':
            mesh_path = init_config.get("path", feature_dict.get("mesh_file", None))
            if mesh_path is None:
                raise ValueError("Mesh path must be provided for 'mesh_sdf' initialization.")
            self.initialize_sdf_feature_from_mesh(mesh_path, name=name)
        elif mode == 'tensor':
            init_tensor = init_config.get("value", feature_dict.get("initial_value", None))
            setattr(self, name, nn.Parameter(init_tensor.to(device='cuda', dtype=torch.float32)))
        elif mode == 'function':
            init_func = init_config.get("func", feature_dict.get("initial_function", None))
            if init_func is None:
                raise ValueError("Initial function must be provided for 'function' initialization.")
            if init_config.get("contract_points", feature_dict.get("contract_points", False)):
                setattr(self, name, nn.Parameter(init_func(self.primal_points)))
            else:
                setattr(self, name, nn.Parameter(init_func(self.primal_points_uncontracted)))
        else:
            raise ValueError("Unknown feature initialization: {}".format(mode))
        
        self.feature_names.append(name)
        self.feature_limits[name] = [feature_dict.get("min", None), feature_dict.get("max", None)]
        self.normalize_feature[name] = feature_dict.get("normalize", False)
        self.feature_grad_accum[name] = torch.zeros(size = (self.primal_points.shape[0], dim), dtype = torch.float32, requires_grad = False, device = 'cuda')
        # Cache the spec for periodic re-initialization. `reset_every_n_steps`
        # is optional; if absent or 0/None, the feature is never reset.
        self.feature_configs[name] = feature_dict
        reset_int = feature_dict.get('reset_every_n_steps', None)
        self.feature_reset_intervals[name] = int(reset_int) if reset_int else None

    def reset_feature(self, name):
        """Re-initialize feature `name` in-place using its stored config.

        In-place is mandatory: we must keep the same `nn.Parameter` object so
        the main optimizer's `param_groups` and any references held by the
        densification/pruning code in this class stay valid. We also zero the
        Adam momentum buffers for this parameter — otherwise the accumulated
        first/second moments would immediately push the freshly-reset values
        back toward whatever they had drifted into. The grad accumulator is
        also cleared so densification heuristics don't see stale gradient
        magnitudes from before the reset.

        Stochastic init modes (uniform / gaussian) draw fresh samples; this
        matches the spirit of "reinitialize as if from scratch" rather than
        replaying the exact original numbers. Shape-dependent inits (mesh_sdf,
        function) re-evaluate against current `primal_points`, so this works
        correctly even if densification/pruning has changed N since startup.
        """
        feature_dict = self.feature_configs[name]
        init_config = feature_dict.get("init", {})
        mode = init_config.get("type", feature_dict.get("initialization", "constant"))
        dim = feature_dict["dim"]
        param = getattr(self, name)
        N = self.primal_points.shape[0]
        device = param.device

        with torch.no_grad():
            if mode == "constant":
                init_val = init_config.get("value", feature_dict.get("initial_value", 0.0))
                param.data.fill_(float(init_val))
            elif mode == "uniform":
                init_val = init_config.get("mean", feature_dict.get("initial_value", 0.0))
                init_scale = init_config.get("std", feature_dict.get("initial_scale", 1.0))
                param.data.copy_((torch.rand((N, dim), dtype=torch.float32, device=device) - 0.5) * init_scale + init_val)
            elif mode in ("gaussian", "random"):
                init_val = init_config.get("mean", feature_dict.get("initial_value", 0.0))
                init_scale = init_config.get("std", feature_dict.get("initial_scale", 1.0))
                param.data.copy_(torch.randn((N, dim), dtype=torch.float32, device=device) * init_scale + init_val)
            elif mode == "mesh_sdf":
                mesh_path = init_config.get("path", feature_dict.get("mesh_file", None))
                if mesh_path is None:
                    raise ValueError("Mesh path must be provided for 'mesh_sdf' reset.")
                import trimesh
                from pysdf import SDF
                mesh = trimesh.load(mesh_path)
                f = SDF(mesh.vertices, mesh.faces)
                pts = self.primal_points_uncontracted.detach().cpu().numpy()
                vals = -f(pts)
                param.data.copy_(torch.tensor(vals, device=device, dtype=torch.float32).unsqueeze(-1))
            elif mode == "tensor":
                init_tensor = init_config.get("value", feature_dict.get("initial_value", None))
                if init_tensor is None:
                    raise ValueError("Initial tensor must be provided for 'tensor' reset.")
                param.data.copy_(init_tensor.to(device=device, dtype=torch.float32))
            elif mode == "function":
                init_func = init_config.get("func", feature_dict.get("initial_function", None))
                if init_func is None:
                    raise ValueError("Initial function must be provided for 'function' reset.")
                if init_config.get("contract_points", feature_dict.get("contract_points", False)):
                    param.data.copy_(init_func(self.primal_points))
                else:
                    param.data.copy_(init_func(self.primal_points_uncontracted))
            else:
                raise ValueError(f"Unknown feature initialization for reset: {mode}")

            # Clear Adam momentum + step count for this Parameter.
            if getattr(self, 'optimizer', None) is not None:
                stored_state = self.optimizer.state.get(param, None)
                if stored_state is not None:
                    if 'exp_avg' in stored_state:
                        stored_state['exp_avg'].zero_()
                    if 'exp_avg_sq' in stored_state:
                        stored_state['exp_avg_sq'].zero_()
                    if 'max_exp_avg_sq' in stored_state:
                        stored_state['max_exp_avg_sq'].zero_()
                    if 'step' in stored_state:
                        # newer torch stores `step` as a tensor; older as an int.
                        s = stored_state['step']
                        if torch.is_tensor(s):
                            s.zero_()
                        else:
                            stored_state['step'] = 0

            # Stale gradient-magnitude EMA used by densification heuristics.
            if name in self.feature_grad_accum:
                self.feature_grad_accum[name].zero_()

    def _tsdf_from_depth_cache(self, cache_path, trunc=0.05, conf_percentile=None,
                               empty_fill=None, chunk=200000):
        """Fuse a TSDF directly at the grid vertices (primal_points_uncontracted,
        model frame) from cached depth maps + cameras.

        The cache (see scripts/vggt_tsdf_cache.py) stores the depth maps,
        intrinsics and extrinsics in the depth-estimator's NATIVE frame, plus a
        single similarity  model = s * R @ x_native + t  that maps that frame to
        the dataset's model frame (verified there against the dataset cameras).
        We map each grid point INTO the native frame and do all projection/depth
        math there -- so no camera or depth ever needs re-scaling (minimal frame
        surface). The fused TSDF (native units) is scaled by s at the end to get
        model-frame signed distances.

        TSDF convention (standard): for grid point x seen in view i with observed
        depth d and point depth z,  raw = d - z  (>0 in front of the surface =
        free space = OUTSIDE, matching the sphere-init sign). Contributions with
        raw < -trunc are occluded (behind the surface) and carry NO weight; the
        rest are clamped to +/-trunc and confidence-weighted. Grid points seen by
        no view fall back to `empty_fill` (default +trunc, i.e. free space)."""
        z = np.load(cache_path)
        dev = 'cuda'
        # nan/inf sanitize: a NaN depth would poison a grid point's SDF via
        # 0*NaN in the accumulation, and a NaN in conf would make the quantile
        # threshold NaN -> every point 'unseen' -> silent all-empty init. 0 is a
        # safe sentinel (killed by the d_obs>1e-6 / c_obs>=thr gates).
        depths = torch.nan_to_num(torch.as_tensor(z['depths'], dtype=torch.float32, device=dev),
                                  nan=0.0, posinf=0.0, neginf=0.0)             # (V,H,W)
        conf = torch.nan_to_num(torch.as_tensor(z['conf'], dtype=torch.float32, device=dev),
                                nan=0.0, posinf=0.0, neginf=0.0)               # (V,H,W)
        K = torch.as_tensor(z['intrinsics'], dtype=torch.float32, device=dev)    # (V,3,3)
        w2c = torch.as_tensor(z['w2c'], dtype=torch.float32, device=dev)         # (V,4,4) native world->cam
        s = float(z['sim_s']); R = torch.as_tensor(z['sim_R'], dtype=torch.float32, device=dev)  # (3,3)
        t = torch.as_tensor(z['sim_t'], dtype=torch.float32, device=dev)         # (3,)
        V, H, W = depths.shape

        # confidence gate: a per-scene threshold (percentile over valid depths)
        if conf_percentile is None:
            conf_percentile = float(z['conf_percentile']) if 'conf_percentile' in z.files else 0.0
        if conf_percentile and conf_percentile > 0:
            # sample-with-replacement for the quantile estimate: avoids a full
            # int64 permutation of the entire depth stack (V*H*W can be >1e8)
            flat = conf.reshape(-1)
            k = min(2_000_000, flat.numel())
            idx = torch.randint(0, flat.numel(), (k,), device=dev)
            thr = torch.quantile(flat[idx], float(conf_percentile) / 100.0)
        else:
            thr = torch.tensor(-1e30, device=dev)

        trunc_native = float(trunc) / s            # model-unit trunc -> native units
        # empty_fill: a number, or 'visual_hull' to decide unseen points by the
        # object masks shipped in the cache (interior -> -trunc, else +trunc).
        hull_masks = None
        if isinstance(empty_fill, str) and empty_fill.lower() == 'visual_hull':
            assert 'masks' in z.files, \
                "empty_fill='visual_hull' needs 'masks' in the depth cache"
            hull_masks = torch.as_tensor(z['masks'], device=dev)
            empty_fill = None
        efill = (float(trunc) if empty_fill is None else float(empty_fill))     # model units

        p_model = self.primal_points_uncontracted.detach().to(dev)              # (M,3) model frame
        M = p_model.shape[0]
        out = torch.empty(M, 1, dtype=torch.float32, device=dev)

        Rt = R.t().contiguous()
        for a in range(0, M, chunk):
            pm = p_model[a:a + chunk]                                            # (m,3)
            # model -> native:  x_native = ((pm - t)/s) @ R   (inverse of  s * x@R.T + t)
            pn = ((pm - t) / s) @ R                                             # (m,3)
            num = torch.zeros(pn.shape[0], device=dev)
            den = torch.zeros(pn.shape[0], device=dev)
            pn_h = torch.cat([pn, torch.ones_like(pn[:, :1])], dim=-1)          # (m,4)
            for i in range(V):
                cam = pn_h @ w2c[i].t()                                         # (m,4) -> world->cam
                zc = cam[:, 2]
                front = zc > 1e-6
                uv = (cam[:, :3] @ K[i].t())
                u = uv[:, 0] / zc.clamp_min(1e-6)
                v = uv[:, 1] / zc.clamp_min(1e-6)
                ui = u.round().long(); vi = v.round().long()
                inb = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
                if not inb.any():
                    continue
                ui = ui.clamp(0, W - 1); vi = vi.clamp(0, H - 1)
                flat = vi * W + ui
                d_obs = depths[i].reshape(-1)[flat]                            # (m,)
                c_obs = conf[i].reshape(-1)[flat]
                valid = inb & (d_obs > 1e-6) & (c_obs >= thr)
                raw = d_obs - zc                                               # native units
                w = valid.float() * c_obs * (raw > -trunc_native).float()
                tsdf = raw.clamp(-trunc_native, trunc_native)
                num = num + w * tsdf
                den = den + w
            seen = den > 0
            fused = torch.where(seen, num / den.clamp_min(1e-12) * s,          # native->model scale
                                torch.full_like(num, efill))
            if hull_masks is not None:
                # Points behind the surface get NO weight (the raw < -trunc gate),
                # so the whole deep interior would fall back to efill = free space
                # and marching tets would emit a spurious inner shell. Decide those
                # by the visual hull instead: inside every view's mask -> interior.
                inside = torch.ones(pn.shape[0], dtype=torch.bool, device=dev)
                for i in range(V):
                    cam = pn_h @ w2c[i].t()
                    zc = cam[:, 2]
                    uv = (cam[:, :3] @ K[i].t())
                    u = (uv[:, 0] / zc.clamp_min(1e-6)).round().long()
                    v = (uv[:, 1] / zc.clamp_min(1e-6)).round().long()
                    inb = (zc > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                    m = torch.zeros_like(inside)
                    idx = inb.nonzero(as_tuple=True)[0]
                    if idx.numel():
                        m[idx] = hull_masks[i].reshape(-1)[v[idx] * W + u[idx]] > 0
                    # outside the frustum a view cannot veto membership
                    inside &= (m | ~inb)
                fused = torch.where(~seen & inside, torch.full_like(fused, -efill), fused)
            out[a:a + chunk, 0] = fused
        seen_frac = 100.0 * float((out.abs() < efill - 1e-9).float().mean())
        print(f"[depth_tsdf] {M} grid pts, {V} views, trunc={trunc} (model) -> "
              f"{seen_frac:.1f}% within truncation band, "
              f"neg(inside)={100.0*float((out<0).float().mean()):.1f}%")
        return out

    def initialize_sdf_feature_from_mesh(self, mesh_path, name='sdf'):
        import trimesh
        from pysdf import SDF
        mesh = trimesh.load(mesh_path)
        f = SDF(mesh.vertices, mesh.faces)
        def sdf_func(x):
            input = x.cpu().detach().numpy()
            output = -f(input) # Use negative SDF for inside/outside convention if needed
            return torch.tensor(output, device='cuda', dtype=torch.float32).unsqueeze(-1)
        
        val = sdf_func(self.primal_points_uncontracted)
        setattr(self, name, nn.Parameter(val))


    def initialize_points(self, primal_points):
        # contraction = get_contraction_type(self.config.get('contraction_type', 'aabb'))

        # roi = torch.tensor(roi, dtype=torch.float32, device=primal_points.device)
        # device = primal_points.device

        # primal_points = contract_inv(primal_points, roi, contraction)
        # #After contraction we need to move primal points to device
        # primal_points = primal_points.to(device)

        # primal_points.clamp_(min=-1e8,max=1e8)
        primal_points_uncontracted = contract_inv(primal_points, self.roi, self.contraction_type)
        self.triangulation = triangulation_with_retry(primal_points_uncontracted.contiguous())
        perm = self.triangulation.permutation().to(torch.long)
        primal_points = primal_points[perm]
        self.primal_points = nn.Parameter(primal_points)
        for fname in self.feature_names:
            data = getattr(self, fname)
            data = data[perm]
            setattr(self, fname, nn.Parameter(data))
        self.faces = None

    def initialize_points_from_grid(self):
        tets = np.load('data/tets/{}_tets.npz'.format(self.config.grid_resolution))
        grid_verts = torch.tensor(tets['vertices'], dtype = torch.float32, device = 'cuda')
        primal_points = grid_verts + 1/100.0*(torch.rand_like(grid_verts) - 0.5) + 0.5
        if self.contraction_type == ContractionType.UN_BOUNDED_SPHERE:
            primal_points = primal_points[torch.norm(primal_points - 0.5, dim=-1) <= 0.4995]
        self.initialize_points(primal_points)

    def initialize_points_random(self):
        if self.contraction_type == ContractionType.UN_BOUNDED_SPHERE:
            # Sample uniformly in the CONTRACTED sphere. max_init_radius caps the
            # contracted radius (in [-1,1]); the contraction blows the far edge up
            # to astronomic world radii (0.999 -> world r~1500), which is useless
            # geometry and, near the mlp=1 boundary, noise-dominated. 0.92 -> world
            # r~19: comfortably past a background shell at world r~8 with margin for
            # training to expand it, without the far garbage.
            num_points = self.config.num_init_points
            device = 'cuda'
            dirs = torch.randn(num_points, 3, device=device)
            dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
            rmax = float(self.config.get('max_init_radius', 0.999))
            radii = rmax*(torch.rand(num_points, 1, device=device) ** (1/3))
            primal_points = dirs * radii
            primal_points = 0.5*primal_points + 0.5
        else:
            primal_points = torch.rand(self.config.num_init_points, 3, device = 'cuda')
        self.initialize_points(primal_points)

    def initialize_points_from_seeds(self, file_path):
        """SfM-seeded init: load a SPARSE seed cloud (COLMAP-triangulated,
        pre-normalised to the model frame), spawn a dense cluster of grid points
        in the *vicinity* of each seed, and add a coarse background shell so the
        Delaunay domain is enclosed. Unlike initialize_points_from_pcd this does
        NOT set the SDF from the ply — the SDF is left to the feature registry
        (typically a random/gaussian init), so triangles appear near the seeds.

        Config keys (all optional, under model.grid):
          num_seed_points        cap on seeds after subsampling (default 40000)
          seed_max_norm          drop seeds with |x|>this in model frame (64.0 --
                                 only cuts SfM junk-at-infinity; real far
                                 background structure is kept and spawned with
                                 distance-scaled jitter)
          num_neighbors_per_seed K jittered points spawned per seed (8)
          seed_spawn_radius      half-width of the per-seed jitter box, model frame (0.02)
          num_background_points  coarse points over the ROI for the outside (20000)
          background_min_dist    keep bg points at least this far from any seed (0.05)
        """
        import open3d as o3d
        cfg = self.config
        pcd = o3d.io.read_point_cloud(file_path)
        # Uniformize seed density FIRST: SfM clouds are wildly non-uniform
        # (feature-rich spots carry micro-clusters of near-coincident tracks).
        # radfoam's Delaunay spatial hash has a fixed per-bucket capacity and
        # dies with an illegal access when too many points share a bucket --
        # whether it triggers is a coin flip on the bbox-normalized grid, and
        # the crash poisons the CUDA context (no retry possible), so density
        # must be bounded up front. 2cm spacing < the 3cm spawn jitter, so the
        # spawn structure is unaffected. Verified across torch seeds 42/0/7.
        vox_seed = float(cfg.get('seed_voxel_size', 0.02))
        if vox_seed > 0:
            pcd = pcd.voxel_down_sample(vox_seed)
        seeds = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
        seeds = seeds[torch.norm(seeds, dim=-1) <= float(cfg.get('seed_max_norm', 64.0))]
        max_seeds = int(cfg.get('num_seed_points', 40000))
        if seeds.shape[0] > max_seeds:
            seeds = seeds[torch.randperm(seeds.shape[0], device='cuda')[:max_seeds]]

        parts = [seeds]
        K = int(cfg.get('num_neighbors_per_seed', 8))
        r = float(cfg.get('seed_spawn_radius', 0.02))
        if K > 0 and r > 0:
            rep = seeds.repeat_interleave(K, dim=0)
            # distance-scaled jitter: far seeds (kept for background structure)
            # get a spawn box that grows linearly with |x|, matching the
            # contraction's tangential footprint -- a fixed 3cm box at r=50
            # would collapse all K spawns into near-coincident points (both
            # useless as tets and a density hazard for the Delaunay hash)
            jit_scale = rep.norm(dim=-1, keepdim=True).clamp_min(1.0)
            parts.append(rep + (torch.rand_like(rep) - 0.5) * (2.0 * r) * jit_scale)

        num_bg = int(cfg.get('num_background_points', 20000))
        if num_bg > 0:
            if self.contraction_type == ContractionType.UN_BOUNDED_SPHERE:
                dirs = torch.randn(num_bg, 3, device='cuda'); dirs = dirs / dirs.norm(dim=-1, keepdim=True)
                radii = 0.99 * (torch.rand(num_bg, 1, device='cuda') ** (1.0 / 3.0))
                bg_c = 0.5 * dirs * radii + 0.5
            else:
                bg_c = torch.rand(num_bg, 3, device='cuda')
            bg = contract_inv(bg_c, self.roi, self.contraction_type)
            thr = float(cfg.get('background_min_dist', 0.05))
            if thr > 0 and seeds.shape[0] > 0:
                kd = cKDTree(seeds.detach().cpu().numpy())
                nb = kd.query_ball_point(bg.detach().cpu().numpy(), thr, workers=-1)
                keep = torch.from_numpy(np.array([len(n) == 0 for n in nb])).to('cuda')
                bg = bg[keep]
            parts.append(bg)

        points = torch.cat(parts, dim=0)
        points = points[torch.isfinite(points).all(dim=-1)]                    # drop any non-finite
        # Voxel-dedup: SfM seed clouds carry sub-mm micro-clusters (duplicate
        # tracks). Together with the far background shell (r up to ~100) this
        # spans a huge dynamic range with near-coincident points, which
        # overflows radfoam's fixed-resolution Delaunay spatial hash
        # (sorted_map illegal access). Collapse points closer than one voxel
        # (< the seed-spawn jitter, so the spawn structure is preserved).
        vox = float(cfg.get('seed_dedup_voxel', 0.01))
        if vox > 0:
            keys = torch.round(points / vox).to(torch.int64)
            keys = keys - keys.min(dim=0).values
            lin = (keys[:, 0] * 4194304 + keys[:, 1]) * 4194304 + keys[:, 2]
            uniq, first = torch.unique(lin, return_inverse=True)
            keep = torch.zeros(uniq.shape[0], dtype=torch.int64, device=points.device)
            keep.scatter_(0, first, torch.arange(points.shape[0], device=points.device))
            points = points[keep]
        points = points + 1 / 1000.0 * (torch.rand_like(points) - 0.5)        # break coplanar degeneracies
        print(f"[seed_spawn] seeds={seeds.shape[0]} +neighbours(K={K},r={r}) +bg "
              f"-> {points.shape[0]} grid points (voxel-deduped @ {vox})")
        import os as _os
        if _os.environ.get('SEED_DUMP'):
            torch.cuda.synchronize()
            print(f"[seed_dump] dtype={points.dtype} contig={points.is_contiguous()} "
                  f"finite={torch.isfinite(points).all().item()} rmax={points.norm(dim=-1).max().item():.3f} "
                  f"aabb_min={points.min(0).values.tolist()} aabb_max={points.max(0).values.tolist()}", flush=True)
            torch.save(points.detach().cpu(), '/tmp/real_seed_points.pt')
            print("[seed_dump] saved /tmp/real_seed_points.pt", flush=True)

        self.triangulation = triangulation_with_retry(points.contiguous())
        perm = self.triangulation.permutation().to(torch.long)
        contracted_points = contract(points, self.roi, self.contraction_type)
        self.primal_points = nn.Parameter(contracted_points[perm])
        self.faces = None

    def initialize_points_from_pcd(self, file_path):
        #Read point cloud file with open3d
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(file_path)
        points_np = np.asarray(pcd.points)
        points = torch.tensor(points_np, dtype=torch.float32, device='cuda')

        #Filter points with norm too large
        norm = torch.norm(points, dim=-1)
        mask = norm <= 4.0
        points = points[mask]

        points = points + 1/1000.0*(torch.rand_like(points) - 0.5)
        num_points = self.config.num_init_points

        if self.contraction_type == ContractionType.UN_BOUNDED_SPHERE:
            # Sample uniformly in sphere
            device = 'cuda'
            dirs = torch.randn(num_points, 3, device=device)
            dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
            radii = 0.99*(torch.rand(num_points, 1, device=device) ** (1/3))
            random_outside_points_contracted = dirs * radii
            random_outside_points_contracted = 0.5*random_outside_points_contracted + 0.5
        else:
            random_outside_points_contracted = torch.rand(num_points, 3, device = 'cuda')

        random_outside_points = contract_inv(random_outside_points_contracted, self.roi, self.contraction_type)
        threshold = 0.18
        kdtree = cKDTree(points_np)
        neighbor_indices = kdtree.query_ball_point(random_outside_points.cpu().numpy(), threshold, workers=-1)
        keep_mask = np.array([len(n) == 0 for n in neighbor_indices])
        keep_mask = torch.from_numpy(keep_mask).to('cuda')
        filtered_random_points = random_outside_points[keep_mask]
        filtered_random_points_contracted = random_outside_points_contracted[keep_mask]
        num_random = filtered_random_points.shape[0]
        points = torch.cat([points, filtered_random_points], dim=0)

        self.triangulation = triangulation_with_retry(points.contiguous())
        perm = self.triangulation.permutation().to(torch.long)
        contracted_points = contract(points, self.roi, self.contraction_type)
        primal_points = contracted_points[perm]
        self.primal_points = nn.Parameter(primal_points)
        self.faces = None
        color = np.asarray(pcd.colors)
        colors = torch.tensor(color, dtype=torch.float32, device='cuda')
        colors = colors[mask]
        sdf = colors[:,0:1]*2.0-1.0  #Assume red channel encodes sdf in [0,1]

        #new_norm = 2*torch.norm(filtered_random_points_contracted - 0.5, dim=-1, keepdim=True)
        #new_sdf = torch.where(new_norm < 0.9, 1.0, 20.0*(0.95 - new_norm))
        new_sdf = torch.ones((num_random,1), dtype=torch.float32, device='cuda')

        sdf = torch.cat([sdf, new_sdf], dim=0)
        sdf = 0.1*sdf
        self.initialize_feature('sdf', {
            'dim': 1,
            'initialization' : 'tensor',
            'initial_value': sdf[perm]
        })
        # self.update_triangulation(rebuild=False)

        # for i in range(2):
        #     tet_means = self.primal_points[self.indices,:].mean(dim=-2)
        #     sample_mask = (tet_means - 0.5).abs().max(dim=1).values < 0.5
        #     num_new_points = sample_mask.sum()
        #     sampled_barycentric_coords = torch.full((num_new_points, 4), 1.0/4.0, device=self.primal_points.device) + torch.randn((num_new_points, 4), device=self.primal_points.device)*0.01
        #     sampled_barycentric_coords /= sampled_barycentric_coords.sum(dim=-1, keepdim=True)
        #     sampled_points = (sampled_barycentric_coords.unsqueeze(-1)*self.primal_points_uncontracted[self.indices[sample_mask,:],:]).sum(dim=-2)
        #     sampled_points = contract(sampled_points, self.roi, self.contraction_type)
        #     #Concat new points
        #     self.primal_points = nn.Parameter(torch.cat([self.primal_points, sampled_points], dim=0))
        #     sampled_sdf = (sampled_barycentric_coords.unsqueeze(-1)*self.sdf[self.indices[sample_mask,:],:]).sum(dim=-2)
        #     self.sdf = nn.Parameter(torch.cat([self.sdf, sampled_sdf], dim=0))  
        #     self.update_triangulation(incremental=False)


    def old_initialize_points_from_pcd(self):
        points = self.config.initial_points
        points = points.to(self.device)
        num_random = 5_000
        random = 2*self.config.scale * (torch.rand(self.config.num_init_points, 3, device = 'cuda') - 0.5) + self.config.offset

        num_samples = int(0.5 * points.shape[0])
        print(f"Starting with {num_samples} points from {points.shape[0]} input points")
        points_idx = torch.randint(0, points.shape[0], (num_samples,))
        samp_points = points[points_idx]
        samp_points += 2*self.config.scale/50.0*(torch.rand_like(samp_points) - 0.5)

        primal_points = torch.cat([samp_points, random], dim=0)
        torch.cuda.empty_cache()
        self.initialize_points(primal_points)


    #-------------------------------------
    # Triangulation methods
    #-------------------------------------

    @torch.no_grad()
    def rearrange_tets(self, tet_fx4, pos):
        # 1) Gather positions for each corner of each tetrahedron.
        p0 = pos[tet_fx4[:, 0]]  # [n_tets, 3]
        p1 = pos[tet_fx4[:, 1]]
        p2 = pos[tet_fx4[:, 2]]
        p3 = pos[tet_fx4[:, 3]]

        # 2) Compute normal for the face defined by corners (1,2,3).
        v12 = p2 - p1  # [n_tets, 3]
        v13 = p3 - p1  # [n_tets, 3]
        n = torch.cross(v12, v13, dim=-1)  # [n_tets, 3]

        # 3) Vector from corner1 to corner0, then dot with face normal
        v10 = p0 - p1  # [n_tets, 3]
        d = (n * v10).sum(dim=-1)  # dot product -> shape [n_tets]

        # 4) Check sign
        outside_mask = d > 0  # counter-clockwise
        # inside_mask = d < 0  # clockwise
        # coplanar_mask = d == 0  # rarely exactly zero in floating-point

        # 5) DMTet class assumes CLOCKWISE arrangement of vertices (1,2,3) w.r.t. vertex 0.
        #    We need to flip the order of vertices (1,2,3) for the tets that are counter-clockwise.
        tet_fx4[outside_mask] = tet_fx4[outside_mask][:, [0, 3, 2, 1]]

        return tet_fx4

    def update_triangulation(self, rebuild=True, incremental=False):

        if not self.primal_points_uncontracted.isfinite().all():
            raise RuntimeError("NaN in points")

        needs_permute = False
        perturbation = 1e-6
        del_points = self.primal_points_uncontracted
        failures = 0
        while rebuild:
            if failures > 25:
                raise RuntimeError("aborted triangulation after 25 attempts")
            try:
                needs_permute = self.triangulation.rebuild(
                    del_points, incremental=incremental
                )
                break
            except radfoam.TriangulationFailedError as e:
                print("caught: ", e)
                perturbation *= 2
                failures += 1
                incremental = False
                with torch.no_grad():
                    del_points = (
                        self.primal_points_uncontracted
                        + perturbation * torch.randn_like(self.primal_points_uncontracted)
                    )

        if failures > 5:
            with torch.no_grad():
                self.primal_points.copy_(contract(del_points, self.roi, self.contraction_type))

        if needs_permute:
            perm = self.triangulation.permutation().to(torch.long)
            self.permute_points(perm)

        # self.aabb_tree = radfoam.build_aabb_tree(self.primal_points)
        #Update indices and edges
        # self.indices = self.triangulation.tets().long()
        self.indices = self.rearrange_tets(
            self.triangulation.tets().long(), self.primal_points_uncontracted
        )
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = self.indices.device)
            all_edges = self.indices[:,edges].reshape(-1,2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

        self.point_adjacency = self.triangulation.point_adjacency()
        self.point_adjacency_offsets = (
            self.triangulation.point_adjacency_offsets()
        )

        if self._tetrahedra_tracer is not None:
            self._tetrahedra_tracer.load_tetrahedra(self.primal_points_uncontracted, self.indices.to(torch.int32))

    def large_tet_keep_mask(self, max_edge_factor=None, max_edge_abs=None):
        """Boolean keep-mask over the tetrahedra (`self.indices`): True = keep.

        In large-scale / unbounded scenes the Delaunay triangulation produces a
        few enormous tetrahedra that span empty space; their zero-crossing yields
        huge spurious triangles. This drops tets whose largest edge exceeds a
        threshold, computed (in the uncontracted/model frame) as
            thr = min( max_edge_factor * median(max_edge), max_edge_abs )
        using whichever of the two knobs are provided. Returns all-True if neither
        is set. The mask is meant to be applied ONLY at the marching_tets call
        (self.indices itself is left intact, so tracer/densification stay valid)."""
        pts = self.primal_points_uncontracted
        tv = pts[self.indices]                                   # [T,4,3]
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        edges = torch.stack([tv[:, a] - tv[:, b] for a, b in pairs], dim=1)  # [T,6,3]
        max_edge = edges.norm(dim=-1).amax(dim=1)                # [T]
        thr = None
        if max_edge_factor is not None:
            thr = float(max_edge_factor) * max_edge.median()
        if max_edge_abs is not None:
            a = max_edge.new_tensor(float(max_edge_abs))
            thr = a if thr is None else torch.minimum(thr, a)
        if thr is None:
            return torch.ones(self.indices.shape[0], dtype=torch.bool, device=pts.device)
        return max_edge <= thr

    #-------------------------------------
    # Permutation, Pruning and Densification
    #-------------------------------------

    def register_optimizer(self, optimizer):
        self.optimizer = optimizer
        if self.config.point_initialization == "points":
            for i in range(1):
                tet_means = self.primal_points[self.indices,:].mean(dim=-2)
                sample_mask = (tet_means - 0.5).abs().max(dim=1).values < 0.5
                num_new_points = sample_mask.sum()
                probs = sample_mask.float()
                self.sample_new_points(num_new_points = num_new_points, probs = probs, replacement = False)

    def permute_points(self, permutation):
        optimizable_tensors = {}
        param_names_to_update = {"primal_points"}
        param_names_to_update.update(x for x in self.feature_names)

        for name, param in self.named_parameters():
            if name not in param_names_to_update:
                continue

            matching_groups = [g for g in self.optimizer.param_groups if g["params"][0].data_ptr() == param.data_ptr()]
            if not matching_groups:
                continue

            group = matching_groups[0]
            old_param = group["params"][0]
            stored_state = self.optimizer.state.get(old_param, None)

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][permutation]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][permutation]
                del self.optimizer.state[old_param]
                new_param = nn.Parameter((old_param[permutation].requires_grad_(True)))
                group["params"][0] = new_param
                self.optimizer.state[new_param] = stored_state
                optimizable_tensors[name] = new_param
            else:
                new_param = nn.Parameter((old_param[permutation].requires_grad_(True)))
                group["params"][0] = new_param
                optimizable_tensors[name] = new_param

        self.primal_points = optimizable_tensors.get("primal_points", self.primal_points)
        self.recompute_primal_points_uncontracted = True
        for fname in self.feature_names:
            if fname in optimizable_tensors:
                setattr(self, fname, optimizable_tensors[fname])
        self.point_grad_accum = self.point_grad_accum[permutation]
        for fname in self.feature_grad_accum.keys():
            self.feature_grad_accum[fname] = self.feature_grad_accum[fname][permutation]
       
    def prune_optimizer(self, mask):
        optimizable_tensors = {}
        param_names_to_update = {"primal_points"}
        param_names_to_update.update(self.feature_names)

        for name, param in self.named_parameters():

            if name not in param_names_to_update:
                continue

            matching_groups = [g for g in self.optimizer.param_groups if g["params"][0].data_ptr() == param.data_ptr()]
            if not matching_groups:
                optimizable_tensors[name] = nn.Parameter(param[mask])
                continue
            group = matching_groups[0]
            stored_state = self.optimizer.state.get(group["params"][0], None)
    
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[name] = group["params"][0]

            
        return optimizable_tensors
    
    def propagate_mask(self, mask):
        #Propagate mask to neighbors
        point_adjacency = self.point_adjacency
        point_adjacency_offsets = self.point_adjacency_offsets

        neighbor_mask = mask.long()[point_adjacency.long()]
        nsum_mask = torch.cumsum(neighbor_mask, dim=0)
        nsum_mask = torch.cat(
            [torch.zeros_like(nsum_mask[:1]), nsum_mask],
            dim=0
        )
        offsets = point_adjacency_offsets.long()
        num_masked_neighbors = nsum_mask[offsets[1:]] - nsum_mask[offsets[:-1]]
        propagated_mask = (num_masked_neighbors > 0) | mask
        return propagated_mask

    def prune_points(self, prune_mask):
        # Rebuild-time micro-cluster guard: points DRIFT into sub-voxel clusters
        # during optimisation (the densify-time guard only sees newly added
        # points), and the full Delaunay rebuild below overflows its sorted_map
        # buckets on such clusters (illegal access, cuh:109). Extend the prune
        # mask so at most one point survives per fine voxel of contracted space.
        with torch.no_grad():
            vox = float(self.config.get('dedup_voxel', 2.0 / 2048.0))
            p = self.primal_points
            lo = p.min(dim=0).values - vox
            occ = ((p - lo) / vox).long()
            dims = occ.max(dim=0).values + 2
            lin = (occ[:, 0] * dims[1] + occ[:, 1]) * dims[2] + occ[:, 2]
            # keep the first point per voxel among the ones NOT already pruned;
            # points already marked for pruning must not "reserve" a voxel.
            lin_sort, order = torch.sort(lin + prune_mask.long() * (lin.max() + 1))
            first = torch.ones_like(lin, dtype=torch.bool)
            first[1:] = lin_sort[1:] != lin_sort[:-1]
            dup = torch.zeros_like(first)
            dup[order] = ~first
            n_dup = int((dup & ~prune_mask).sum())
            if n_dup > 0:
                print(f"[prune] micro-cluster guard pruning {n_dup} voxel-duplicate points")
            prune_mask = prune_mask | dup
        valid_points_mask = ~prune_mask
        optimizable_tensors = self.prune_optimizer(valid_points_mask)
        self.point_grad_accum = self.point_grad_accum[valid_points_mask]
        self.primal_points = optimizable_tensors["primal_points"]
        self.recompute_primal_points_uncontracted = True
        for fname in self.feature_names:
            self.feature_grad_accum[fname] = self.feature_grad_accum[fname][valid_points_mask]
            if fname in optimizable_tensors:
                setattr(self, fname, optimizable_tensors[fname])
        
        self._tetrahedra_tracer = None
        print(f"Number of pruned points: {prune_mask.sum()}")
        self.update_triangulation(incremental=False)


    def cat_tensors_to_optimizer(self, new_params):
        optimizable_tensors = {}
        param_names_to_update = {"primal_points"}
        param_names_to_update.update(new_params.keys())

        for name, param in self.named_parameters():
            if name not in param_names_to_update:
                continue

            matching_groups = [g for g in self.optimizer.param_groups if g["params"][0].data_ptr() == param.data_ptr()]
            if not matching_groups:
                continue

            group = matching_groups[0]        
            assert len(group["params"]) == 1
            stored_tensor = group["params"][0]
            extension_tensor = new_params[name]
            stored_state = self.optimizer.state.get(
                group["params"][0], None
            )
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (
                        stored_state["exp_avg"],
                        torch.zeros_like(extension_tensor),
                    ),
                    dim=0,
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (
                        stored_state["exp_avg_sq"],
                        torch.zeros_like(extension_tensor),
                    ),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (stored_tensor, extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (stored_tensor, extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[name] = group["params"][0]

        return optimizable_tensors

    def add_points(self, new_params):
        optimizable_tensors = self.cat_tensors_to_optimizer(new_params)
        self.point_grad_accum = torch.cat(
            (self.point_grad_accum, torch.zeros_like(new_params["primal_points"][...,0], requires_grad = False)),
            dim=0,
        )
        self.primal_points = optimizable_tensors["primal_points"]
        self.recompute_primal_points_uncontracted = True
        for fname in self.feature_names:
            self.feature_grad_accum[fname] = torch.cat(
                (self.feature_grad_accum[fname], torch.zeros_like(new_params[fname], requires_grad = False)),
                dim=0
            )
            if fname in optimizable_tensors:
                setattr(self, fname, optimizable_tensors[fname])

    def sample_new_points(self, num_new_points, probs = None, replacement = True, feature_sample_functions = {}):
        if probs is None:
            probs = torch.ones((self.indices.shape[0],), device = self.indices.device)

        if not replacement:
            valid = probs > 0
            max_without_replacement = valid.sum().item()
            num_new_points = min(num_new_points, max_without_replacement)

        # all-zero probs (or a 0-point request) reduce the sample count to 0;
        # multinomial raises on n<=0 -- skip this densification step instead
        # (observed: bear mdP crash at it8860, "cannot sample n_sample <= 0")
        if num_new_points <= 0:
            return

        sampled_inds = torch.multinomial(
            probs,
            num_new_points,
            replacement=replacement,
        )
        sampled_barycentric_coords = torch.full((num_new_points, 4), 1.0/4.0, device=self.primal_points.device) + torch.randn((num_new_points, 4), device=self.primal_points.device)*0.01
        sampled_barycentric_coords /= sampled_barycentric_coords.sum(dim=-1, keepdim=True)
        sampled_points = (sampled_barycentric_coords.unsqueeze(-1)*self.primal_points_uncontracted[self.indices[sampled_inds,:],:]).sum(dim=-2)
        sampled_points = contract(sampled_points, self.roi, self.contraction_type)
        # Micro-cluster guard (same failure mode as the init-time seed dedup):
        # densified points landing in an already-occupied fine voxel pile up
        # over rounds and overflow the Delaunay sorted_map buckets (illegal
        # access at sorted_map.cuh:109). Contracted coords are bounded, so a
        # global fine voxel grid is valid: drop candidates colliding with an
        # existing point or with each other at ~1e-3 resolution.
        with torch.no_grad():
            vox = float(self.config.get('dedup_voxel', 2.0 / 2048.0))
            lo = torch.minimum(self.primal_points.min(dim=0).values,
                               sampled_points.min(dim=0).values) - vox
            occ_old = ((self.primal_points - lo) / vox).long()
            occ_new = ((sampled_points - lo) / vox).long()
            dims = torch.maximum(occ_old.max(dim=0).values, occ_new.max(dim=0).values) + 2
            lin_old = (occ_old[:, 0] * dims[1] + occ_old[:, 1]) * dims[2] + occ_old[:, 2]
            lin_new = (occ_new[:, 0] * dims[1] + occ_new[:, 1]) * dims[2] + occ_new[:, 2]
            order = torch.argsort(lin_new)
            keep_sorted = torch.ones_like(lin_new, dtype=torch.bool)
            keep_sorted[1:] = lin_new[order][1:] != lin_new[order][:-1]  # first per voxel
            keep = torch.zeros_like(keep_sorted)
            keep[order] = keep_sorted
            keep &= ~torch.isin(lin_new, lin_old.unique())
        if int(keep.sum()) < num_new_points:
            print(f"[densify] micro-cluster guard dropped {num_new_points - int(keep.sum())} of {num_new_points} candidates")
        sampled_inds = sampled_inds[keep]
        sampled_barycentric_coords = sampled_barycentric_coords[keep]
        sampled_points = sampled_points[keep]
        num_new_points = int(keep.sum())
        new_params = {
            "primal_points": sampled_points
        }
        for fname in self.feature_names:
            feature = getattr(self, fname)
            if fname in feature_sample_functions:
                sampled_feature = feature_sample_functions[fname](sampled_inds, sampled_barycentric_coords, sampled_points)
            else:
                sampled_feature = (sampled_barycentric_coords.unsqueeze(-1)*feature[self.indices[sampled_inds,:],:]).sum(dim=-2)
            new_params[fname] = sampled_feature

        self.add_points(new_params)
        self.point_grad_accum = torch.zeros_like(
            self.primal_points[..., 0], requires_grad=False
        )
        for fname in self.feature_grad_accum.keys():
            self.feature_grad_accum[fname] = torch.zeros_like(
                getattr(self, fname), requires_grad=False
            )

        self.update_triangulation(incremental=False)
        print(f"Number of new points: {num_new_points}")
        print(f"Number of total points: {self.primal_points.shape[0]}")

    #TODO: FIX PRUNE AND DENSIFY 
    def prune_and_densify(
        self, sdf, upsample_factor=1.2
    ):
        
        with torch.no_grad():
            points = self.primal_points
            point_adjacency = self.point_adjacency
            point_adjacency_offsets = self.point_adjacency_offsets

            ################### Farthest neighbor ###################
            farthest_neighbor, cell_radius = radfoam.farthest_neighbor(
                points,
                point_adjacency,
                point_adjacency_offsets,
            )
            farthest_neighbor = farthest_neighbor.long()

            ######################## Pruning ########################
            sdf_pos_mask = sdf > 0.0
            neighbor_pos_mask = sdf_pos_mask.long()[point_adjacency.long()]
            n_sum_pos = torch.cumsum(neighbor_pos_mask, dim=0)
            nsum_pos = torch.cat(
                [torch.zeros_like(n_sum_pos[:1]), n_sum_pos],
                dim=0
            )
            offsets = point_adjacency_offsets.long()
            num_pos_neighbors = nsum_pos[offsets[1:]] - nsum_pos[offsets[:-1]] 
            # _point_adjacency_offsets = torch.cat(
            #     [point_adjacency_offsets, torch.zeros_like(point_adjacency_offsets[:1])], dim=0
            # ).long()
            num_neighbors = offsets[1:] - offsets[:-1]
            diff_sign_mask = ~(num_pos_neighbors == sdf_pos_mask * num_neighbors)

            #Use diff sign mask to find points with neighbors that have all the same sign as well
            neighbor_diff_sign_mask = diff_sign_mask[point_adjacency.long()]
            nsum_diff_sign = torch.cumsum(neighbor_diff_sign_mask, dim=0)
            nsum_diff_sign = torch.cat(
                [torch.zeros_like(nsum_diff_sign[:1]), nsum_diff_sign],
                dim=0
            )
            num_neighbors_diff_sign = nsum_diff_sign[offsets[1:]] - nsum_diff_sign[offsets[:-1]]
            prune_mask = (num_neighbors_diff_sign == 0) & ~diff_sign_mask
            self.prune_points(prune_mask)
            self.update_triangulation(incremental=False)

            #contrib_mask = ((n_masked_adj == 0) & ~same_sign_mask).squeeze()
            # cell_size_mask = cell_radius < 1e-1
            # prune_mask = contrib_mask * cell_size_mask

            # sdf_pos_mask = self.sdf > 5.0 
            # sdf_neg_mask = self.sdf < -5.0
            # neighbor_pos_mask = sdf_pos_mask.long()[point_adjacency.long()]
            # neighbor_pos_mask = torch.cat(
            #     [neighbor_pos_mask, torch.zeros_like(neighbor_pos_mask[:1])], dim=0
            # )
            # neighbor_neg_mask = sdf_neg_mask.long()[point_adjacency.long()]
            # neighbor_neg_mask = torch.cat(
            #     [neighbor_neg_mask, torch.zeros_like(neighbor_neg_mask[:1])], dim=0
            # )

            # nsum_neg = torch.cumsum(neighbor_neg_mask, dim=0)
            # nsum_pos = torch.cumsum(neighbor_pos_mask, dim=0)

            # offsets = point_adjacency_offsets.long()
            # n_masked_adj_neg = nsum_neg[offsets[1:]] - nsum_neg[offsets[:-1]]
            # n_masked_adj_pos = nsum_pos[offsets[1:]] - nsum_pos[offsets[:-1]]

            # contrib_mask_neg = ((n_masked_adj_neg == 0) & ~sdf_neg_mask).squeeze()
            # contrib_mask_pos = ((n_masked_adj_pos == 0) & ~sdf_pos_mask).squeeze()
            # contrib_mask = torch.logical_or(contrib_mask_neg, contrib_mask_pos)

            # cell_size_mask = cell_radius < 1e-1
            # prune_mask = contrib_mask * cell_size_mask

            ######################## Random sampling ########################
            num_curr_points = self.primal_points.shape[0]
            num_new_points = int((upsample_factor - 1) * num_curr_points)
            sample_mask = (sdf[self.indices].min(dim=-1).values <= self.config.sdf_threshold) & (sdf[self.indices].max(dim=-1).values >= -self.config.sdf_threshold)
            num_new_points = min(sample_mask.sum().item(), num_new_points)
            sampled_inds = torch.multinomial(
                sample_mask.float(),
                num_new_points,
                replacement=False,
            )
            #sampled_barycentric_coords = torch.rand((num_sample_points, 4), device=points.device, dtype=points.dtype)
            #sampled_barycentric_coords /= sampled_barycentric_coords.sum(dim=-1, keepdim=True)
            sampled_barycentric_coords = torch.full((num_new_points, 4), 1.0/4.0, device=self.primal_points.device)
            sampled_points = (sampled_barycentric_coords.unsqueeze(-1)*self.primal_points[self.indices[sampled_inds,:],:]).sum(dim=-2)

            # contrib_mask = torch.where(diff_sign_mask == 1, 0.99999, 0.00001) 
            # print(contrib_mask)
            # print(contrib_mask.max())
            # print(contrib_mask.min())
            # print(self.point_error_accum)
            # print(self.point_error_accum.max())
            # print(self.point_error_accum.min())
            # perturbation = 0.25 * (points[farthest_neighbor] - points)
            # delta = torch.randn_like(perturbation)
            # delta /= (delta.norm(dim=-1, keepdim=True) + 1e-6)
            # perturbation_norm = perturbation.norm(dim=-1, keepdim=True)
            # perturbation += (
            #     0.1 * perturbation_norm * delta
            # )
            # num_sample_points = num_new_points
            # sampled_inds = torch.multinomial(
            #     contrib_mask,
            #     num_sample_points,
            #     replacement=False,
            # )
            # sampled_points = (points + perturbation)[sampled_inds]

            new_params = {
                "primal_points": sampled_points
            }
            for fname in self.feature_names:
                feature = getattr(self, fname)
                sampled_feature = (sampled_barycentric_coords.unsqueeze(-1)*feature[self.indices[sampled_inds,:],:]).sum(dim=-2)
                new_params[fname] = sampled_feature


            # prune_mask = torch.cat(
            #     (
            #         prune_mask,
            #         torch.zeros(
            #             sampled_points.shape[0],
            #             device=prune_mask.device,
            #             dtype=bool,
            #         ),
            #     )
            # )

            self.add_points(new_params)
            #self.prune_points(prune_mask)
            self.point_grad_accum = torch.zeros_like(
                self.primal_points[..., 0], requires_grad=False
            )
            for fname in self.feature_grad_accum.keys():
                self.feature_grad_accum[fname] = torch.zeros_like(
                    getattr(self, fname), requires_grad=False
                )
            print(f"Number of pruned points: {prune_mask.sum()}")
            print(f"Number of new points: {num_new_points}")
            print(f"Number of total points: {self.primal_points.shape[0]}")

    
    #-------------------------------------
    # Export
    #-------------------------------------
    @torch.no_grad()
    def export(self, filename, scale = 1.0, transform = torch.eye(4), additional_fields = {}):
        scale = float(scale)
        points = scale * self.primal_points_uncontracted
        transform = transform.to(points.device).to(points.dtype)
        points = (torch.cat([points, torch.ones_like(points[:, :1])], dim=-1) @ transform.T)[:, :3]
        points = points.cpu().numpy()
        #points = (transform @ torch.cat([points, torch.ones_like(points[:,0:1])], dim=-1)[...,None])[:,:3,0]
        vertex_data = []
        for i in tqdm.trange(points.shape[0]):
            entry = [
                points[i, 0],
                points[i, 1],
                points[i, 2],
            ]
            for name, arr in additional_fields.items():
                entry.append(arr[i])
            vertex_data.append(tuple(entry))

        dtype = [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32)
        ]
        for name, arr in additional_fields.items():
            dtype.append((name, arr.dtype))

        vertex_data = np.array(vertex_data, dtype=dtype)
        vertex_element = PlyElement.describe(vertex_data, "vertex")

        edge_data = []
        edges = self.all_edges.cpu().numpy()
        for i in tqdm.trange(edges.shape[0]):
            entry = [
                edges[i, 0],
                edges[i, 1],
            ]
            edge_data.append(tuple(entry))

        dtype = [
            ("vertex1", np.uint32),
            ("vertex2", np.uint32),
        ]

        edge_data = np.array(edge_data, dtype=dtype)
        edge_element = PlyElement.describe(edge_data, "edge")

        PlyData([vertex_element, edge_element]).write(filename)



    #-------------------------------------
    # Additional methods
    #-------------------------------------
    def trace_rays(self, rays_o, rays_d, max_intersected_triangles=512, max_tets=512):
        tracer_output = self.tetrahedra_tracer.trace_rays(
            rays_o.contiguous(),
            rays_d.contiguous(),
            max_intersected_triangles
        )
        tracer_output["hit_distances"] = tracer_output["hit_distances"][:, :max_tets, :]
        tracer_output["barycentric_coordinates"] = tracer_output["barycentric_coordinates"][:, :max_tets, :, :]
        tracer_output["vertex_indices"] = tracer_output["vertex_indices"][:, :max_tets, :]
        tracer_output["num_visited_cells"] = torch.clamp(tracer_output["num_visited_cells"], max=max_tets)
        tracer_output["visited_cells"] = tracer_output["visited_cells"][:, :max_tets]
        t = torch.arange(tracer_output["hit_distances"].size(1), device=tracer_output["hit_distances"].device)
        mask = t.unsqueeze(0) >= tracer_output["num_visited_cells"].unsqueeze(1)
        tracer_output["hit_distances"][mask.unsqueeze(-1).expand_as(tracer_output["hit_distances"])] = float("inf")
        return tracer_output

    def interpolate(self, points, feature_names=None):
        if feature_names is None:
            feature_names = self.feature_names
        
        return_single = False
        if isinstance(feature_names, str):
            feature_names = [feature_names]
            return_single = True
            
        points = points.contiguous()

        # Guard: OptiX optixLaunch fails with OPTIX_ERROR_INVALID_VALUE when launched
        # with 0 points (happens on empty ray-chunks during full-image validation).
        if points.shape[0] == 0:
            results = {n: points.new_zeros((0, getattr(self, n).shape[-1])) for n in feature_names}
            return results[feature_names[0]] if return_single else results

        # Call the extension to find tetrahedra and barycentric coordinates
        tracer_output = self.tetrahedra_tracer.find_tetrahedra(points)
        
        # Handle dict or tuple return from the extension
        if isinstance(tracer_output, dict):
            tet_indices = tracer_output['tetrahedra']
            barycentric_coordinates = tracer_output['barycentric_coordinates']
            vertex_indices = tracer_output['vertex_indices'].long()
        else:
            tet_indices, barycentric_coordinates, vertex_indices = tracer_output
            vertex_indices = vertex_indices.long()
            
        # Compute the 4th barycentric coordinate (extension returns first 3)
        barycentric_coordinates_4 = torch.cat([
            barycentric_coordinates,
            1.0 - barycentric_coordinates.sum(dim=-1, keepdim=True)
        ], dim=-1) # [N, 4]
        
        # Mask for points that fell outside the tetrahedral grid
        mask = (tet_indices >= 0) & (tet_indices < self.indices.shape[0])
        
        results = {}
        for name in feature_names:
            feature = getattr(self, name) # [V, C]
            # Interpolate: sum over vertices of the weighted features
            # feature[vertex_indices]: [N, 4, C]
            # barycentric_coordinates_4: [N, 4] -> [N, 4, 1] for broadcasting
            interpolated = (feature[vertex_indices] * barycentric_coordinates_4.unsqueeze(-1)).sum(dim=-2)
            
            # Zero out points outside the grid
            interpolated[~mask] = 0.0
            results[name] = interpolated
            
        if return_single:
             return results[feature_names[0]]
        return results

    def gradient(self, sdf, tet_indices = None, eps = 1e-6, normalized = True, detach_positions = False):
        """
        Computes the normals of the tetrahedra based on the vertex indices.
        The normals are computed as the cross product of two edges of the tetrahedra.
        """
        if self.indices.shape[1] != 4:
            raise ValueError("Tetrahedra indices must have shape (N, 4)")

        if tet_indices is None:
            tet_indices = torch.arange(self.indices.shape[0], device=self.indices.device)

        if detach_positions:
            points = self.primal_points_uncontracted.detach()
        else:
            points = self.primal_points_uncontracted

        v0 = points[self.indices[tet_indices, 0]]
        v1 = points[self.indices[tet_indices, 1]]
        v2 = points[self.indices[tet_indices, 2]]
        v3 = points[self.indices[tet_indices, 3]]


        sdf0 = sdf[self.indices[tet_indices, 0]]
        sdf1 = sdf[self.indices[tet_indices, 1]]
        sdf2 = sdf[self.indices[tet_indices, 2]]
        sdf3 = sdf[self.indices[tet_indices, 3]]


        e1, e2, e3 = v1-v0, v2-v0, v3-v0

        g = ((sdf1-sdf0).unsqueeze(-1) * torch.cross(e2, e3, dim=-1) +
            (sdf2-sdf0).unsqueeze(-1) * torch.cross(e3, e1, dim=-1) +
            (sdf3-sdf0).unsqueeze(-1) * torch.cross(e1, e2, dim=-1))

        # normalize only
        if (normalized):
            g = g / g.norm(dim=-1, keepdim=True).clamp_min(eps)
        else:
            vol = torch.linalg.det(torch.stack([e1,e2,e3], dim=-1))
            vol = vol.clamp_min(eps)
            g = g / vol.unsqueeze(-1)
        return g

        J = torch.stack([v1-v0, v2-v0, v3-v0], dim=-1)  # shape (num_tets 3,3)
        detJ = torch.linalg.det(J)
        mask = detJ.abs() < eps
        if mask.any():
            eye = torch.eye(3, device=J.device, dtype=J.dtype)
            J[mask] = J[mask] + eps * eye

        Jinv = torch.inverse(J)
        #Jinv = torch.linalg.pinv(J)
        Jinv = Jinv.transpose(-1, -2)  # shape (num_tets, 3, 3)
        sdf_diffs = torch.stack([sdf1-sdf0, sdf2-sdf0, sdf3-sdf0], dim=-1)  # shape (3,)
        grad_f = (Jinv @ sdf_diffs.unsqueeze(-1)).squeeze(-1)
        norm = grad_f.norm(dim=-1, keepdim=True).clamp_min(eps)
        grad_f_normalized = grad_f / norm
        return grad_f_normalized

        norm = grad_f.norm(dim=-1, keepdim=True).clamp_min(eps)
        grad_f_normalized = grad_f / norm
        return grad_f_normalized
