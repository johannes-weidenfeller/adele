import torch
import torch.nn as nn
import gc

class GridManager(nn.Module):
    def __init__(self, config, tetrahedral_grid, 
                 get_viewport_mask_fn=None, 
                 get_contribution_mask_fn=None, 
                 get_densify_probs_fn=None):
        super().__init__()
        self.config = config
        self.grid = tetrahedral_grid
        self.get_viewport_mask_fn = get_viewport_mask_fn
        self.get_contribution_mask_fn = get_contribution_mask_fn
        self.get_densify_probs_fn = get_densify_probs_fn
        
        self.triangulation_update_period = 1
        self.iters_since_update = 1
        self.iters_since_densification = 0
        self.next_densification_after = 1
        self.densify_from, self.densify_until, self.min_steps_per_densification = self.config.densification.schedule[0], self.config.densification.schedule[1], self.config.densification.schedule[2]
        self.densify_factor = self.config.densification.factor
        self.triangulate_until = self.config.get("triangulate_until", 10000)
        self._last_update_step = -1

    def update_step(self, epoch, global_step):
        if not self.training:
            return False

        if self._last_update_step == global_step:
            return False
        
        changed = False
        if self.update_triangulation(global_step):
            changed = True
        if self.update_prune(global_step):
            changed = True
        if self.update_densify(global_step):
            changed = True
        
        self._last_update_step = global_step
        return changed

    def update_triangulation(self, global_step):
        if (self.iters_since_update >= self.triangulation_update_period and global_step <= self.triangulate_until) or global_step == self.triangulate_until:
            self.grid.update_triangulation(incremental=True)
            self.iters_since_update = 0

            if self.triangulation_update_period < 100:
                self.triangulation_update_period += 2
            
            if global_step == self.triangulate_until:
                self.grid.primal_points.requires_grad = False
            return True
        else:
            self.iters_since_update += 1
            return False

    def update_prune(self, global_step):
        v_sch = self.config.pruning.visibility.schedule
        prune_viewport_from, prune_viewport_until, prune_viewport_every = v_sch[0], v_sch[1], v_sch[2]

        m_sch = self.config.pruning.mesh_contribution.schedule
        prune_contribution_from, prune_contribution_until, prune_contribution_every = m_sch[0], m_sch[1], m_sch[2]

        global_prune_mask = None

        # Viewport pruning
        if (global_step >= prune_viewport_from and global_step < prune_viewport_until and 
            (global_step - prune_viewport_from) % prune_viewport_every == 0):
            if self.get_viewport_mask_fn is not None:
                in_view_mask = self.get_viewport_mask_fn()
                for _ in range(self.config.pruning.visibility.num_neighbors):
                    in_view_mask = self.grid.propagate_mask(in_view_mask)
                prune_mask = ~in_view_mask
                global_prune_mask = prune_mask if global_prune_mask is None else global_prune_mask | prune_mask

        # Contribution pruning
        if (global_step >= prune_contribution_from and global_step < prune_contribution_until and 
            (global_step - prune_contribution_from) % prune_contribution_every == 0):
            if self.get_contribution_mask_fn is not None:
                contrib_mask = self.get_contribution_mask_fn()
                for _ in range(self.config.pruning.mesh_contribution.num_neighbors):
                    contrib_mask = self.grid.propagate_mask(contrib_mask)
                prune_mask = ~contrib_mask
                global_prune_mask = prune_mask if global_prune_mask is None else global_prune_mask | prune_mask

        # Occupancy pruning (if applicable)
        #  if self.config.sampling == 'tetrahedra':
        #     o_sch = self.config.pruning.occupancy.schedule
        #     prune_occupancy_from, prune_occupancy_until, prune_occupancy_every = o_sch[0], o_sch[1], o_sch[2]
        #     if (global_step >= prune_occupancy_from and global_step < prune_occupancy_until and 
        #         (global_step - prune_occupancy_from) % prune_occupancy_every == 0):
        #         occ_mask = (self.grid.occupancies > self.config.pruning.occupancy.threshold).squeeze(-1)
        #         for _ in range(self.config.pruning.occupancy.num_neighbors):
        #             occ_mask = self.grid.propagate_mask(occ_mask)
        #         prune_mask = ~occ_mask
        #         global_prune_mask = prune_mask if global_prune_mask is None else global_prune_mask | prune_mask

        if global_prune_mask is not None and global_prune_mask.any():
            self.grid.prune_points(global_prune_mask)
            return True
        return False

    def update_densify(self, global_step):
        if global_step >= self.densify_from:
            self.iters_since_densification += 1

        if (self.iters_since_densification == self.next_densification_after and global_step < self.densify_until):
            num_new_points = int(self.densify_factor * self.grid.primal_points.shape[0]) - self.grid.primal_points.shape[0]
            
            changed = False
            if self.get_densify_probs_fn is not None:
                tet_probs, num_to_sample = self.get_densify_probs_fn(num_new_points)
                if num_to_sample > 0:
                    self.grid.sample_new_points(num_to_sample, probs=tet_probs, replacement=False)
                    changed = True
            
            self.iters_since_update = 0
            self.triangulation_update_period = 1
            gc.collect()

            self.iters_since_densification = 0
            # Linear growth calculation
            curr_pts = self.grid.primal_points.shape[0]
            denom = (self.config.num_final_points - curr_pts)
            if denom > 0:
                self.next_densification_after = int((self.densify_factor - 1) * curr_pts * (self.densify_until - global_step) / denom)
            else:
                self.next_densification_after = self.min_steps_per_densification
            self.next_densification_after = max(self.next_densification_after, self.min_steps_per_densification)
            return changed
        return False
