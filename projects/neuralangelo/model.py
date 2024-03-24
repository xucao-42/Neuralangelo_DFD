'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

from functools import partial
import torch
import torch.nn.functional as torch_F
from collections import defaultdict

from imaginaire.models.base import Model as BaseModel
from projects.nerf.utils import nerf_util, camera, render
from projects.neuralangelo.utils import misc
from projects.neuralangelo.utils.modules import NeuralSDF, NeuralRGB, BackgroundNeRF

from icecream import ic
import pyvista as pv

class Model(BaseModel):

    def __init__(self, cfg_model, cfg_data):
        super().__init__(cfg_model, cfg_data)
        self.cfg_render = cfg_model.render
        self.white_background = cfg_model.background.white
        self.with_background = cfg_model.background.enabled
        self.with_appear_embed = cfg_model.appear_embed.enabled
        self.anneal_end = cfg_model.object.s_var.anneal_end
        self.outside_val = 1000. * (-1 if cfg_model.object.sdf.mlp.inside_out else 1)
        self.image_size_train = cfg_data.train.image_size
        self.image_size_val = cfg_data.val.image_size
        # Define models.
        self.build_model(cfg_model, cfg_data)
        # Define functions.
        self.ray_generator = partial(nerf_util.ray_generator,
                                     camera_ndc=False,
                                     num_rays=cfg_model.render.rand_rays)
        self.sample_dists_from_pdf = partial(nerf_util.sample_dists_from_pdf,
                                             intvs_fine=cfg_model.render.num_samples.fine)
        self.to_full_val_image = partial(misc.to_full_image, image_size=cfg_data.val.image_size)
        self.render_mode = cfg_model.render.render_mode
        ic(self.render_mode)

    def build_model(self, cfg_model, cfg_data):
        # appearance encoding
        if cfg_model.appear_embed.enabled:
            assert cfg_data.num_images is not None
            self.appear_embed = torch.nn.Embedding(cfg_data.num_images, cfg_model.appear_embed.dim)
            if cfg_model.background.enabled:
                self.appear_embed_outside = torch.nn.Embedding(cfg_data.num_images, cfg_model.appear_embed.dim)
            else:
                self.appear_embed_outside = None
        else:
            self.appear_embed = self.appear_embed_outside = None
        self.neural_sdf = NeuralSDF(cfg_model.object.sdf)
        self.neural_rgb = NeuralRGB(cfg_model.object.rgb, feat_dim=cfg_model.object.sdf.mlp.hidden_dim,
                                    appear_embed=cfg_model.appear_embed)
        if cfg_model.background.enabled:
            self.background_nerf = BackgroundNeRF(cfg_model.background, appear_embed=cfg_model.appear_embed)
        else:
            self.background_nerf = None
        self.s_var = torch.nn.Parameter(torch.tensor(cfg_model.object.s_var.init_val, dtype=torch.float32))

    def forward(self, data):
        if self.render_mode=="pixel":
            # Randomly sample and render the pixels.
            output = self.render_pixels(data["pose"], data["intr"], image_size=self.image_size_train,
                                        stratified=self.cfg_render.stratified, sample_idx=data["idx"],
                                        ray_idx=data["ray_idx"])
        elif self.render_mode=="patch":
            # Randomly sample and render the patches of pixels.
            output = self.render_patches(data["pose"], data["intr"], image_size=self.image_size_train,
                                    stratified=self.cfg_render.stratified, sample_idx=data["idx"],
                                    patch_center_idx=data["patch_center_idx"], patch_all_idx=data["patch_all_idx"])  # [B,N,C]
        return output

    @torch.no_grad()
    def inference(self, data):
        self.eval()
        # Render the full images.
        output = self.render_image(data["pose"], data["intr"], image_size=self.image_size_val,
                                   stratified=False, sample_idx=data["idx"])  # [B,N,C]
        # Get full rendered RGB and depth images.
        rot = data["pose"][..., :3, :3]  # [B,3,3]
        normal_cam = -output["gradient"] @ rot.transpose(-1, -2)  # [B,HW,3]
        output.update(
            rgb_map=self.to_full_val_image(output["rgb"]),  # [B,3,H,W]
            opacity_map=self.to_full_val_image(output["opacity"]),  # [B,1,H,W]
            depth_map=self.to_full_val_image(output["depth"]),  # [B,1,H,W]
            normal_map=self.to_full_val_image(normal_cam),  # [B,3,H,W]
        )
        return output

    def render_image(self, pose, intr, image_size, stratified=False, sample_idx=None):
        """ Render the rays given the camera intrinsics and poses.
        Args:
            pose (tensor [batch,3,4]): Camera poses ([R,t]).
            intr (tensor [batch,3,3]): Camera intrinsics.
            stratified (bool): Whether to stratify the depth sampling.
            sample_idx (tensor [batch]): Data sample index.
        Returns:
            output: A dictionary containing the outputs.
        """
        output = defaultdict(list)
        for center, ray, _ in self.ray_generator(pose, intr, image_size, full_image=True):
            ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
            output_batch = self.render_rays(center, ray_unit, sample_idx=sample_idx, stratified=stratified)
            if not self.training:
                dist = render.composite(output_batch["dists"], output_batch["weights"])  # [B,R,1]
                depth = dist / ray.norm(dim=-1, keepdim=True)
                output_batch.update(depth=depth)
            for key, value in output_batch.items():
                if value is not None:
                    output[key].append(value.detach())
        # Concat each item (list) in output into one tensor. Concatenate along the ray dimension (1)
        for key, value in output.items():
            output[key] = torch.cat(value, dim=1)
        return output

    def render_pixels(self, pose, intr, image_size, stratified=False, sample_idx=None, ray_idx=None):
        center, ray = camera.get_center_and_ray(pose, intr, image_size)  # [B,HW,3]
        center = nerf_util.slice_by_ray_idx(center, ray_idx)  # [B,R,3]
        ray = nerf_util.slice_by_ray_idx(ray, ray_idx)  # [B,R,3]
        ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
        output = self.render_rays(center, ray_unit, sample_idx=sample_idx, stratified=stratified)
        return output

    def render_patches(self, pose, intr, image_size, stratified=False, sample_idx=None, patch_center_idx=None, patch_all_idx=None):
        center, ray = camera.get_center_and_ray(pose, intr, image_size)  # [B,HW,3]
        patch_ray_o = nerf_util.slice_by_ray_idx(center, patch_center_idx)  # [B,R,3]
        patch_ray_center = nerf_util.slice_by_ray_idx(ray, patch_center_idx)  # [B,R,3]
        patch_ray_center_unit = torch_F.normalize(patch_ray_center, dim=-1)  # [B,R,3]

        patch_ray_all = ray[0, patch_all_idx]  # [B, R, P_h, P_w, 3]
        patch_ray_all_unit = torch_F.normalize(patch_ray_all, dim=-1)  # [B, R, P_h, P_w, 3]

        # prepare the marching plane normal
        # pose is w2c
        # pad the pose to [B, 4, 4]
        pose_44 = torch.cat([pose, torch.zeros(pose.shape[0], 1, 4).to(pose)], dim=1)
        pose_44[:, 3, 3] = 1
        c2w = torch.inverse(pose_44)

        rays_ex = c2w[:, :3, 0].unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(patch_ray_all_unit.shape)  # [B, R, P_h, P_w, 3]
        rays_ey = c2w[:, :3, 1].unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(patch_ray_all_unit.shape)  # [B, R, P_h, P_w, 3]
        rays_ez = c2w[:, :3, 2].unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(patch_ray_all_unit.shape)  # [B, R, P_h, P_w, 3]

        rays_V = torch.cat([patch_ray_all_unit[..., None, :],
                            rays_ex[..., None, :],
                            rays_ey[..., None, :]], dim=-2)  # [B, 3, 1, 1, 3]
        rays_V_inverse = torch.inverse(rays_V)  # [B, 3, 1, 1, 3]

        output = self.render_rays_patch(patch_ray_o, patch_ray_center_unit, patch_ray_all_unit, rays_ez, rays_V_inverse, sample_idx=sample_idx, stratified=stratified)
        return output

    def render_rays_patch(self, patch_ray_o, patch_center_ray_unit, patch_all_ray_unit, rays_ez, rays_A_inverse, sample_idx=None, stratified=False):
        with torch.no_grad():
            near, far, outside = self.get_dist_bounds(patch_ray_o, patch_center_ray_unit)
        app, app_outside = self.get_appearance_embedding(sample_idx, patch_center_ray_unit.shape[1])
        output_object = self.render_rays_object_patch(patch_ray_o, patch_center_ray_unit, patch_all_ray_unit, rays_ez, rays_A_inverse, near, far, outside, app, stratified=stratified)
        if self.with_background:
            output_background = self.render_rays_background_patch(patch_ray_o, patch_all_ray_unit, far, app_outside, stratified=stratified)

            rgbs = torch.cat([output_object["rgbs"], output_background["rgbs"]], dim=-2)  # [B, P, P_h,P_w,No+Nb,3]
            dists = torch.cat([output_object["dists"], output_background["dists"]], dim=-2)  # [B, P, P_h,P_w,No+Nb,1]
            alphas = torch.cat([output_object["alphas"], output_background["alphas"]], dim=-1)  # [B, P, P_h,P_w,No+Nb]
        else:
            rgbs = output_object["rgbs"]  # [B,R,No,3]
            dists = output_object["dists"]  # [B,R,No,1]
            alphas = output_object["alphas"]  # [B,R,No]
        weights = render.alpha_compositing_weights_patch(alphas)  # [B,P, P_h,P_w,,No+Nb,1]
        # Compute weights and composite samples.
        rgb = render.composite_patch(rgbs, weights)  # [B,P, P_h,P_w,3]
        if self.white_background:
            opacity_all = render.composite(1., weights)  # [B,R,1]
            rgb = rgb + (1 - opacity_all)
        # Collect output.
        output = dict(
            rgb=rgb,  # [B,R,3]
            opacity=output_object["opacity"],  # [B,R,1]/None
            outside=outside,  # [B,R,1]
            dists=dists,  # [B,R,No+Nb,1]
            weights=weights,  # [B,R,No+Nb,1]
            gradient=output_object["gradient"],  # [B,R,3]/None
            gradients=output_object["gradients"],  # [B,R,No,3]
            hessians=output_object["hessians"],  # [B,R,No,3]/None
        )
        return output

    def render_rays(self, center, ray_unit, sample_idx=None, stratified=False):
        with torch.no_grad():
            near, far, outside = self.get_dist_bounds(center, ray_unit)
        app, app_outside = self.get_appearance_embedding(sample_idx, ray_unit.shape[1])
        output_object = self.render_rays_object(center, ray_unit, near, far, outside, app, stratified=stratified)
        if self.with_background:
            output_background = self.render_rays_background(center, ray_unit, far, app_outside, stratified=stratified)
            # Concatenate object and background samples.
            rgbs = torch.cat([output_object["rgbs"], output_background["rgbs"]], dim=2)  # [B,R,No+Nb,3]
            dists = torch.cat([output_object["dists"], output_background["dists"]], dim=2)  # [B,R,No+Nb,1]
            alphas = torch.cat([output_object["alphas"], output_background["alphas"]], dim=2)  # [B,R,No+Nb]
        else:
            rgbs = output_object["rgbs"]  # [B,R,No,3]
            dists = output_object["dists"]  # [B,R,No,1]
            alphas = output_object["alphas"]  # [B,R,No]
        weights = render.alpha_compositing_weights(alphas)  # [B,R,No+Nb,1]
        # Compute weights and composite samples.
        rgb = render.composite(rgbs, weights)  # [B,R,3]
        if self.white_background:
            opacity_all = render.composite(1., weights)  # [B,R,1]
            rgb = rgb + (1 - opacity_all)
        # Collect output.
        output = dict(
            rgb=rgb,  # [B,R,3]
            opacity=output_object["opacity"],  # [B,R,1]/None
            outside=outside,  # [B,R,1]
            dists=dists,  # [B,R,No+Nb,1]
            weights=weights,  # [B,R,No+Nb,1]
            gradient=output_object["gradient"],  # [B,R,3]/None
            gradients=output_object["gradients"],  # [B,R,No,3]
            hessians=output_object["hessians"],  # [B,R,No,3]/None
        )
        return output

    def render_rays_object(self, center, ray_unit, near, far, outside, app, stratified=False):
        with torch.no_grad():
            dists = self.sample_dists_all(center, ray_unit, near, far, stratified=stratified)  # [B,R,N,3]
        points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
        sdfs, feats = self.neural_sdf.forward(points)  # [B,R,N,1],[B,R,N,K]
        sdfs[outside[..., None].expand_as(sdfs)] = self.outside_val
        # Compute 1st- and 2nd-order gradients.
        rays_unit = ray_unit[..., None, :].expand_as(points).contiguous()  # [B,R,N,3]
        gradients, hessians = self.neural_sdf.compute_gradients(points, training=self.training, sdf=sdfs)
        normals = torch_F.normalize(gradients, dim=-1)  # [B,R,N,3]
        rgbs = self.neural_rgb.forward(points, normals, rays_unit, feats, app=app)  # [B,R,N,3]


        # SDF volume rendering.
        alphas = self.compute_neus_alphas(ray_unit, sdfs, gradients, dists, dist_far=far[..., None],
                                          progress=self.progress)  # [B,R,N]
        if not self.training:
            weights = render.alpha_compositing_weights(alphas)  # [B,R,N,1]
            opacity = render.composite(1., weights)  # [B,R,1]
            gradient = render.composite(gradients, weights)  # [B,R,3]
        else:
            opacity = None
            gradient = None
        # Collect output.
        output = dict(
            rgbs=rgbs,  # [B,R,N,3]
            sdfs=sdfs[..., 0],  # [B,R,N]
            dists=dists,  # [B,R,N,1]
            alphas=alphas,  # [B,R,N]
            opacity=opacity,  # [B,R,3]/None
            gradient=gradient,  # [B,R,3]/None
            gradients=gradients,  # [B,R,N,3]
            hessians=hessians,  # [B,R,N,3]/None
        )
        return output

    def render_rays_object_patch(self, rays_o, patch_center_ray_v, patch_all_ray_v, rays_ez, rays_V_inverse, near, far, outside, app, stratified=False):
        with torch.no_grad():
            dists_patch_center = self.sample_dists_all(rays_o, patch_center_ray_v, near, far, stratified=stratified)  # [B,P,N,1]

        dists_patch_all_numerator = (patch_center_ray_v[..., None, None, :] * rays_ez).sum(-1, keepdim=True)  # [B,P,P_h,P_w,N,1]
        dists_patch_all_denominator = (patch_all_ray_v * rays_ez).sum(-1, keepdim=True)  # [B,P,P_h,P_w,N,1]

        dists_patch_all = dists_patch_center[:, :, None, None, :, :] * dists_patch_all_numerator[..., None, :] / dists_patch_all_denominator[..., None, :]  # [B,P,P_h,P_w,N,1]
        points_patch_all = rays_o[..., None, None, None, :] + patch_all_ray_v[..., None, :] * dists_patch_all # [B,P,P_h,P_w,N,3]

        sdfs, feats = self.neural_sdf.forward(points_patch_all)  # [B,P,P_h,P_w, N, 1],[B,R,N,K]
        sdfs[outside[..., None, None, None, :].expand_as(sdfs)] = self.outside_val
        gradients, hessians = self.neural_sdf.compute_gradients(points_patch_all, training=self.training, sdf=sdfs,
                                                                rays_V_inverse=rays_V_inverse,
                                                                points_patch_all=points_patch_all, mode="dfd")  # [B,R,N,3],[B,R,N,3,3]
        normals = torch_F.normalize(gradients, dim=-1)  # [B,P, P_h, P_w, N,3]
        patch_all_ray_v_forward = patch_all_ray_v[..., None, :].expand_as(points_patch_all).contiguous()  # [B,P, P_h, P_w, N,3]
        rgbs = self.neural_rgb.forward(points_patch_all, normals, patch_all_ray_v_forward, feats, app=app)  # [B,R, P_h, P_w,3]
        # SDF volume rendering.
        num_patches = patch_all_ray_v.shape[1]
        far_expand = far[..., None, None, None].expand((1, num_patches , 3, 3, 1, 1))  # [B, P, 1] -> [B, P, P_h, P_w, 1, 1]
        alphas = self.compute_neus_alphas_patches(patch_all_ray_v_forward, sdfs, gradients, dists_patch_all, dist_far=far_expand,
                                          progress=self.progress)  # [B,R,N]
        if not self.training:
            weights = render.alpha_compositing_weights(alphas)  # [B,R,N,1]
            opacity = render.composite(1., weights)  # [B,R,1]
            gradient = render.composite(gradients, weights)  # [B,R,3]
        else:
            opacity = None
            gradient = None
        # Collect output.
        output = dict(
            rgbs=rgbs,  # [B,P,P_h,P_w,N,3]
            sdfs=sdfs[..., 0],  # [B,P,P_h,P_w,N]
            dists=dists_patch_all,  # [B,P,P_h,P_w,N,1]
            alphas=alphas,  # [B,P,P_h,P_w,N]
            opacity=opacity,  # [B,P,P_h,P_w,3]/None
            gradient=gradient,  # [B,P,P_h,P_w,3]/None
            gradients=gradients,  # [B,P,P_h,P_w,N,3]
            hessians=hessians,  # [B,P,P_h,P_w,N,3]/None
        )
        return output

    def render_rays_background(self, center, ray_unit, far, app_outside, stratified=False):
        with torch.no_grad():
            dists = self.sample_dists_background(ray_unit, far, stratified=stratified)
        points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
        rays_unit = ray_unit[..., None, :].expand_as(points)  # [B,R,N,3]
        rgbs, densities = self.background_nerf.forward(points, rays_unit, app_outside)  # [B,R,N,3]
        alphas = render.volume_rendering_alphas_dist(densities, dists)  # [B,R,N]
        # Collect output.
        output = dict(
            rgbs=rgbs,  # [B,R,3]
            dists=dists,  # [B,R,N,1]
            alphas=alphas,  # [B,R,N]
        )
        return output

    def render_rays_background_patch(self, rays_o, patch_all_ray_v, far, app_outside, stratified=False):
        num_patches = patch_all_ray_v.shape[1]
        far_expand = far[..., None, None, None].expand((1, num_patches, 3, 3, 1, 1))  # [B, P, 1] -> [B, P, P_h, P_w, 1, 1]
        with torch.no_grad():
            dists_patch_all = self.sample_dists_background_patch(patch_all_ray_v, far_expand, stratified=stratified)  # [B,P, P_h, P_w, N,1]

        points_patch_all = rays_o[..., None, None, None, :] + patch_all_ray_v[..., None, :] * dists_patch_all  # [B,P,P_h,P_w,N,3]

        patch_all_ray_v_forward = patch_all_ray_v[..., None, :].expand_as(points_patch_all)  # [B,R,N,3]
        rgbs, densities = self.background_nerf.forward(points_patch_all, patch_all_ray_v_forward, app_outside)  # [B,R,N,3]
        alphas = render.volume_rendering_alphas_dist(densities, dists_patch_all)  # [B,R,N]
        output = dict(
            rgbs=rgbs,  # [B,P, P_h, P_w,3]
            dists=dists_patch_all,  # [B,P, P_h, P_w,N,1]
            alphas=alphas,  # [B,P, P_h, P_w,N]
        )
        return output

    @torch.no_grad()
    def get_dist_bounds(self, center, ray_unit):
        dist_near, dist_far = nerf_util.intersect_with_sphere(center, ray_unit, radius=1.)
        dist_near.relu_()  # Distance (and thus depth) should be non-negative.
        outside = dist_near.isnan()
        dist_near[outside], dist_far[outside] = 1, 1.2  # Dummy distances. Density will be set to 0.
        return dist_near, dist_far, outside

    def get_appearance_embedding(self, sample_idx, num_rays):
        if self.with_appear_embed:
            # Object appearance embedding.
            num_samples_all = self.cfg_render.num_samples.coarse + \
                self.cfg_render.num_samples.fine * self.cfg_render.num_sample_hierarchy
            app = self.appear_embed(sample_idx)[:, None, None]  # [B,1,1,C]
            app = app.expand(-1, num_rays, num_samples_all, -1)  # [B,R,N,C]
            # Background appearance embedding.
            if self.with_background:
                app_outside = self.appear_embed_outside(sample_idx)[:, None, None]  # [B,1,1,C]
                app_outside = app_outside.expand(-1, num_rays, self.cfg_render.num_samples.background, -1)  # [B,R,N,C]
            else:
                app_outside = None
        else:
            app = app_outside = None
        return app, app_outside

    @torch.no_grad()
    def sample_dists_all(self, center, ray_unit, near, far, stratified=False):
        dists = nerf_util.sample_dists(ray_unit.shape[:2], dist_range=(near[..., None], far[..., None]),
                                       intvs=self.cfg_render.num_samples.coarse, stratified=stratified)
        delta_ray_center_coarse = (dists[..., 1:, :] - dists[..., :-1, :])  # [B,R,N,3]

        if self.cfg_render.num_sample_hierarchy > 0:
            points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
            sdfs = self.neural_sdf.sdf(points)  # [B,R,N]
        for h in range(self.cfg_render.num_sample_hierarchy):
            dists_fine = self.sample_dists_hierarchical(dists, sdfs, inv_s=(64 * 2 ** h))  # [B,R,Nf,1]
            dists_fine = dists_fine + torch.randn_like(dists_fine) * 0.05
            # check whether dists_fine contains the same value
            delta_dists_fine = (dists_fine[..., 1:, :] - dists_fine[..., :-1, :])  # [B,R,Nf,1]
            # assert (delta_dists_fine == 0).sum() == 0
            dists = torch.cat([dists, dists_fine], dim=2)  # [B,R,N+Nf,1]
            dists, sort_idx = dists.sort(dim=2)
            if h != self.cfg_render.num_sample_hierarchy - 1:
                points_fine = camera.get_3D_points_from_dist(center, ray_unit, dists_fine)  # [B,R,Nf,3]
                sdfs_fine = self.neural_sdf.sdf(points_fine)  # [B,R,Nf]
                sdfs = torch.cat([sdfs, sdfs_fine], dim=2)  # [B,R,N+Nf]
                sdfs = sdfs.gather(dim=2, index=sort_idx.expand_as(sdfs))  # [B,R,N+Nf,1]
        # ensure there is no repeated value in axis 2 of dists
        for _ in range(2):
            delta_ray_center = (dists[..., 1:, :] - dists[..., :-1, :])
            repeated_mask = delta_ray_center == 0
            dists[:, :, 1:, :][repeated_mask] = dists[:, :, 1:, :][repeated_mask] + 1e-4
            dists, sort_idx = dists.sort(dim=-2)
            delta_ray_center = (dists[..., 1:, :] - dists[..., :-1, :])
        return dists

    def sample_dists_hierarchical(self, dists, sdfs, inv_s, robust=True, eps=1e-5):
        sdfs = sdfs[..., 0]  # [B,R,N]

        prev_sdfs, next_sdfs = sdfs[..., :-1], sdfs[..., 1:]  # [B,R,N-1]
        prev_dists, next_dists = dists[..., :-1, 0], dists[..., 1:, 0]  # [B,R,N-1]
        mid_sdfs = (prev_sdfs + next_sdfs) * 0.5  # [B,R,N-1]
        # ic(mid_sdfs)
        cos_val = (next_sdfs - prev_sdfs) / (next_dists - prev_dists + 1e-5)  # [B,R,N-1]
        if robust:
            prev_cos_val = torch.cat([torch.zeros_like(cos_val)[..., :1], cos_val[..., :-1]], dim=-1)  # [B,R,N-1]
            cos_val = torch.stack([prev_cos_val, cos_val], dim=-1).min(dim=-1).values  # [B,R,N-1]
        dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N-1]
        est_prev_sdf = mid_sdfs - cos_val * dist_intvs * 0.5  # [B,R,N-1]
        est_next_sdf = mid_sdfs + cos_val * dist_intvs * 0.5  # [B,R,N-1]
        prev_cdf = (est_prev_sdf * inv_s).sigmoid()  # [B,R,N-1]
        next_cdf = (est_next_sdf * inv_s).sigmoid()  # [B,R,N-1]
        alphas = ((prev_cdf - next_cdf) / (prev_cdf + eps)).clip_(0.0, 1.0)  # [B,R,N-1]
        weights = render.alpha_compositing_weights(alphas)  # [B,R,N-1,1]
        dists_fine = self.sample_dists_from_pdf(dists, weights=weights[..., 0])  # [B,R,Nf,1]
        return dists_fine

    def sample_dists_background(self, ray_unit, far, stratified=False, eps=1e-5):
        inv_dists = nerf_util.sample_dists(ray_unit.shape[:2], dist_range=(1, 0),
                                           intvs=self.cfg_render.num_samples.background, stratified=stratified)
        dists = far[..., None] / (inv_dists + eps)  # [B,R,N,1]
        return dists

    def sample_dists_background_patch(self, patch_ray_all_v, far, stratified=False, eps=1e-5):
        inv_dists = nerf_util.sample_dists_patch(patch_ray_all_v.shape[:4], dist_range=(1, 0),
                                           intvs=self.cfg_render.num_samples.background, stratified=stratified)
        dists = far / (inv_dists + eps)  # [B, P, P_h, P_w, 1]
        return dists

    def compute_neus_alphas(self, ray_unit, sdfs, gradients, dists, dist_far=None, progress=1., eps=1e-5):
        sdfs = sdfs[..., 0]  # [B,R,N]
        # SDF volume rendering in NeuS.
        inv_s = self.s_var.exp()
        true_cos = (ray_unit[..., None, :] * gradients).sum(dim=-1, keepdim=False)  # [B,R,N]
        iter_cos = self._get_iter_cos(true_cos, progress=progress)  # [B,R,N]
        # Estimate signed distances at section points
        if dist_far is None:
            dist_far = torch.empty_like(dists[..., :1, :]).fill_(1e10)  # [B,R,1,1]
        dists = torch.cat([dists, dist_far], dim=2)  # [B,R,N+1,1]
        dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N]
        est_prev_sdf = sdfs - iter_cos * dist_intvs * 0.5  # [B,R,N]
        est_next_sdf = sdfs + iter_cos * dist_intvs * 0.5  # [B,R,N]
        prev_cdf = (est_prev_sdf * inv_s).sigmoid()  # [B,R,N]
        next_cdf = (est_next_sdf * inv_s).sigmoid()  # [B,R,N]
        alphas = ((prev_cdf - next_cdf) / (prev_cdf + eps)).clip_(0.0, 1.0)  # [B,R,N]
        # weights = render.alpha_compositing_weights(alphas)  # [B,R,N,1]
        return alphas

    def compute_neus_alphas_patches(self, ray_unit, sdfs, gradients, dists, dist_far=None, progress=1., eps=1e-5):
        sdfs = sdfs[..., 0]  # [B,P,P_h, P_w, N]
        # ray_unit: # [B,P,P_h, P_w, 1, 3]
        # gradients: # [B,P,P_h, P_w, N, 3]
        # dists: # [B,P,P_h, P_w, N, 1]
        # SDF volume rendering in NeuS.
        inv_s = self.s_var.exp()
        true_cos = (ray_unit * gradients).sum(dim=-1, keepdim=False)  # [B,R,N]
        iter_cos = self._get_iter_cos(true_cos, progress=progress)  # [B,R,N]
        # Estimate signed distances at section points
        if dist_far is None:
            dist_far = torch.empty_like(dists[..., :1, :]).fill_(1e10)  # [B,R,1,1]
        dists = torch.cat([dists, dist_far], dim=-2)  # [B,P,N+1,1]
        dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,P, P_h, P_w, N]

        est_prev_sdf = sdfs - iter_cos * dist_intvs * 0.5  # [B,R,N]
        est_next_sdf = sdfs + iter_cos * dist_intvs * 0.5  # [B,R,N]
        prev_cdf = (est_prev_sdf * inv_s).sigmoid()  # [B,R,N]
        next_cdf = (est_next_sdf * inv_s).sigmoid()  # [B,R,N]
        alphas = ((prev_cdf - next_cdf) / (prev_cdf + eps)).clip_(0.0, 1.0)  # [B,R,N]
        return alphas

    def _get_iter_cos(self, true_cos, progress=1.):
        anneal_ratio = min(progress / self.anneal_end, 1.)
        # The anneal strategy below keeps the cos value alive at the beginning of training iterations.
        return -((-true_cos * 0.5 + 0.5).relu() * (1.0 - anneal_ratio) +
                 (-true_cos).relu() * anneal_ratio)  # always non-positive
