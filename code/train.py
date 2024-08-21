import os
import time
import logging
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer

from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from gaussian_renderer import render
from argparse import ArgumentParser
from random import randint
import torchvision

from utils.loss_utils import l1_loss, ssim, l1_loss_edge_weighted

class Runner:
    def __init__(self, conf_path, case, gs_checkpoint, neus_checkpoint):
        # config
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['general.data_dir'] = self.conf['general.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.gs_iter = 0
        # gs
        self.data_dir = self.conf.get_string('general.data_dir')
        self.gaussians = GaussianModel(self.conf['gs.model'])
        self.scene = Scene(self.conf, self.gaussians, shuffle=False)
        self.gaussians.training_setup(self.conf['gs.opt'])
        if gs_checkpoint:
            self.load_gs_checkpoint(gs_checkpoint)
        white_background = self.conf.get_bool('gs.model.white_background')
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.viewpoint_stack = None
        # neus
        self.device = torch.device('cuda')
        self.dataset = Dataset(self.conf)
        self.image_perm = self.get_image_perm()
        self.neus_iter = 0

        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['neus.model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['neus.model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['neus.model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['neus.model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        learning_rate = self.conf.get_float('neus.train.learning_rate')
        self.optimizer = torch.optim.Adam(params_to_train, lr=learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['neus.model.neus_renderer'])
        
        if neus_checkpoint:
            self.load_neus_checkpoint(neus_checkpoint)


        #general
        self.pretrain_iter = self.conf.get_int('general.pretrain_iter')
        self.wo_inter_iter = self.conf.get_int('general.wo_inter_iter')
        self.w_inter_iter = self.conf.get_int('general.w_inter_iter')
    
    def load_gs_checkpoint(self, gs_checkpoint):
        (model_params, self.gs_iter) = torch.load(gs_checkpoint)
        self.gaussians.restore(model_params, self.conf['gs.opt'])

    def load_neus_checkpoint(self, neus_checkpoint):
        checkpoint = torch.load(neus_checkpoint, map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.neus_iter = checkpoint['neus_iter']

    def debug(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        # for iteration in range(100000):
        #     self.neus_train_one_iter(is_inter=True)
            # self.neus_iter += 1
        for iteration in range(15000):
            scale_mat_inv = torch.from_numpy(np.linalg.inv(self.dataset.scale_mats_np[0])).cuda()
            self.gaussians.sdp(self.sdf_network, scale_mat_inv, self.scene.cameras_extent)
            # self.gs_train_one_iter(is_inter=True)
            # self.gs_iter += 1

        # self.viewpoint_stack = self.scene.getTrainCameras().copy()
        # for i in range(len(self.viewpoint_stack)):
        #     print(self.viewpoint_stack[i].image_name)

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        print('Begin pretrain')
        res_pretrain_iter = self.pretrain_iter - self.gs_iter
        for iteration in tqdm(range(res_pretrain_iter)):
            self.gs_train_one_iter()
        print('End pretrain')

        print('Begin wo-inter train')
        res_wo_inter_train_iter = self.pretrain_iter + self.wo_inter_iter - self.gs_iter
        for iteration in range(res_wo_inter_train_iter):
            self.neus_train_one_iter()
            self.gs_train_one_iter()
        print('End wo-inter train')

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self, anneal_end):
        if anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.neus_iter / anneal_end])

    def update_learning_rate(self):
        warm_up_end = self.conf.get_int('neus.train.warm_up_end')
        learning_rate_alpha = self.conf.get_float('neus.train.learning_rate_alpha')
        end_iter = self.conf.get_int('neus.train.end_iter')
        learning_rate = self.conf.get_float('neus.train.learning_rate')

        if self.neus_iter < warm_up_end:
            learning_factor = self.neus_iter / warm_up_end
        else:
            alpha = learning_rate_alpha
            progress = (self.neus_iter - warm_up_end) / (end_iter - warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate * learning_factor
    
    def neus_train_one_iter(self, is_inter = False):
        self.neus_iter += 1
        self.update_learning_rate()

        idx = self.image_perm[self.neus_iter % len(self.image_perm)]
        batch_size = self.conf.get_int('neus.train.batch_size')
        data, pixels_x, pixels_y = self.dataset.gen_random_rays_at(idx, batch_size)

        rays_o, rays_d, true_rgb, true_normal, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 12], data[:, 12: 13]
    
        if not is_inter:
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
        else:
            viewpoint_stack = self.scene.getTrainCameras().copy()
            random_background = self.conf.get_bool('gs.opt.random_background')
            bg = torch.rand((3), device="cuda") if random_background else self.background
            render_pkg = render(viewpoint_stack[idx], self.gaussians, self.conf['gs.pipe'], bg, return_depth=True)
            depth = render_pkg['render_depth']
            depth = depth[(pixels_y, pixels_x)].reshape(batch_size, 1) / self.dataset.scale
            sdf = torch.abs(self.sdf_network.sdf(rays_o + depth * rays_d))

            gamma = self.conf.get_int('general.gamma')
            near = depth - gamma * sdf
            far = depth + gamma * sdf
        
        use_white_bkgd = self.conf.get_bool('neus.train.use_white_bkgd')
        background_rgb = None
        if use_white_bkgd:
            background_rgb = torch.ones([1, 3])

        mask_weight = self.conf.get_float('neus.train.mask_weight')
        if mask_weight > 0.0:
            mask = (mask > 0.5).float()
        else:
            mask = torch.ones_like(mask)

        mask_sum = mask.sum() + 1e-5
        anneal_end = self.conf.get_float('neus.train.anneal_end', default=0.0)
        render_out = self.renderer.render(rays_o, rays_d, near, far,
                                            background_rgb=background_rgb,
                                            cos_anneal_ratio=self.get_cos_anneal_ratio(anneal_end))
        
        color_fine = render_out['color_fine']
        s_val = render_out['s_val']
        cdf_fine = render_out['cdf_fine']
        gradient_error = render_out['gradient_error']
        weight_max = render_out['weight_max']
        weight_sum = render_out['weight_sum']
        normal = render_out['normal']

        normal = normal.permute(1, 0)
        rot = self.dataset.pose_all[idx, :3, :3].permute(1, 0)
        normal = torch.matmul(rot, normal).permute(1, 0).clip(-1, 1)

        color_error = (color_fine - true_rgb) * mask
        color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
        psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

        eikonal_loss = gradient_error

        mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

        normal = torch.nn.functional.normalize(normal, p=2, dim=-1)
        true_normal = torch.nn.functional.normalize(true_normal, p=2, dim=-1)
        normal_loss = torch.abs(normal - true_normal).mean()
        
        igr_weight = self.conf.get_float('neus.train.igr_weight')
        mask_weight = self.conf.get_float('neus.train.mask_weight')

        loss = color_fine_loss +\
                eikonal_loss * igr_weight +\
                mask_loss * mask_weight +\
                normal_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('NeusLoss/loss', loss, self.neus_iter)
        self.writer.add_scalar('NeusLoss/color_loss', color_fine_loss, self.neus_iter)
        self.writer.add_scalar('NeusLoss/eikonal_loss', eikonal_loss, self.neus_iter)
        self.writer.add_scalar('NeusLoss/normal_loss', normal_loss, self.neus_iter)
        self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.neus_iter)
        self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.neus_iter)
        self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.neus_iter)
        self.writer.add_scalar('Statistics/psnr', psnr, self.neus_iter)
        
        save_freq = self.conf.get_int('neus.train.save_freq')
        report_freq = self.conf.get_int('neus.train.report_freq')
        val_freq = self.conf.get_int('neus.train.val_freq')
        val_mesh_freq = self.conf.get_int('neus.train.val_mesh_freq')

        if self.neus_iter % report_freq == 0:
            print(f'[ITER {self.neus_iter}] NeuS loss = {loss}')
        if self.neus_iter % save_freq == 0:
            self.save_neus_checkpoint()
        if self.neus_iter % val_freq == 0:
            self.validate_image()
        if self.neus_iter % val_mesh_freq == 0:
            self.validate_mesh()
        if self.neus_iter % len(self.image_perm) == 0:
            self.image_perm = self.get_image_perm()

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'neus', 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'neus', 'meshes', '{:0>8d}.ply'.format(self.neus_iter)))
    def validate_gs_image(self, idx=-1):
        viewpoint_stack = self.scene.getTrainCameras().copy()
        if idx < 0:
            idx = np.random.randint(len(viewpoint_stack))

        random_background = self.conf.get_bool('gs.opt.random_background')
        bg = torch.rand((3), device="cuda") if random_background else self.background
        render_pkg = render(viewpoint_stack[idx], self.gaussians, self.conf['gs.pipe'], bg)
        img, normal = render_pkg['render'], render_pkg['render_normal']
        normal = normal / 2 + 0.5

        render_path = os.path.join(self.base_exp_dir, 'gs', "ours_{}".format(self.gs_iter), "renders")
        render_normal_path = os.path.join(self.base_exp_dir, 'gs', "ours_{}".format(self.gs_iter), "renders_normal")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(render_normal_path, exist_ok=True)
        
        torchvision.utils.save_image(img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(normal, os.path.join(render_normal_path, '{0:05d}'.format(idx) + ".png"))

        
    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.neus_iter, idx))

        if resolution_level < 0:
            resolution_level = self.conf.get_int('neus.train.validate_resolution_level')
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        batch_size = self.conf.get_int('neus.train.batch_size')
        rays_o = rays_o.reshape(-1, 3).split(batch_size)
        rays_d = rays_d.reshape(-1, 3).split(batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            use_white_bkgd = self.conf.get_bool('neus.train.use_white_bkgd')
            background_rgb = torch.ones([1, 3]) if use_white_bkgd else None

            anneal_end = self.conf.get_float('neus.train.anneal_end', default=0.0)
            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(anneal_end),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            out_normal_fine.append(render_out['normal'].detach().cpu().numpy())
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 255).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (((np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape([H, W, 3, -1])) * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)

        os.makedirs(os.path.join(self.base_exp_dir, 'neus', 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'neus', 'normals'), exist_ok=True)

        def imwrite_rgb(path, image):
            image_bgr_to_save = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            cv.imwrite(path, image_bgr_to_save)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                imwrite_rgb(os.path.join(self.base_exp_dir,
                                        'neus',
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.neus_iter, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                imwrite_rgb(os.path.join(self.base_exp_dir,
                                        'neus',
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.neus_iter, i, idx)),
                           normal_img[..., i])

    def save_neus_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'neus_iter': self.neus_iter,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'neus', 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'neus', 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.neus_iter)))
    
    def gs_densify_wo_inter(self, iteration, viewspace_point_tensor, visibility_filter, radii):
        densify_until_iter = self.conf.get_int('gs.opt.densify_until_iter')
        densify_from_iter = self.conf.get_int('gs.opt.densify_from_iter')
        densification_interval = self.conf.get_int('gs.opt.densification_interval')
        opacity_reset_interval = self.conf.get_int('gs.opt.opacity_reset_interval')
        densify_grad_threshold = self.conf.get_float('gs.opt.densify_grad_threshold')
        white_background = self.conf.get_bool('gs.model.white_background')
        with torch.no_grad():
            if iteration < densify_until_iter:
                # Keep track of max radii in image-space for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opacity_reset_interval else None
                    self.gaussians.densify_and_prune(densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
                
                if iteration % opacity_reset_interval == 0 or (white_background and iteration == densify_from_iter):
                    self.gaussians.reset_opacity()

    def gs_densify_w_inter(self, iteration):
        fsdp = self.conf.get_int('general.fsdp')
        fsgd = self.conf.get_int('general.fsgd')
        if iteration % fsdp == 0:
            self.gaussians.sdp(self.sdf_network)
        if iteration % fsgd == 0:
            pass

    def gs_train_one_iter(self, is_inter = False):
        self.gs_iter += 1
        iteration = self.gs_iter
        self.gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))

        # Render
        random_background = self.conf.get_bool('gs.opt.random_background')
        bg = torch.rand((3), device="cuda") if random_background else self.background

        render_pkg = render(viewpoint_cam, self.gaussians, self.conf['gs.pipe'], bg)
        image, normal, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg['render_normal'], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_normal = viewpoint_cam.normal.cuda()
        edge = viewpoint_cam.edge.cuda()

        lambda_dssim = self.conf.get_float('gs.opt.lambda_dssim')
        lambda_normal = self.conf.get_float('gs.opt.lambda_normal')
        Ll1 = l1_loss_edge_weighted(image, gt_image, edge)
        Ll1_normal = l1_loss(normal, gt_normal)
        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image)) + lambda_normal * Ll1_normal
        loss.backward()

        self.writer.add_scalar('GSLoss/loss', loss, self.gs_iter)
        self.writer.add_scalar('GSLoss/color_loss', Ll1, self.gs_iter)
        self.writer.add_scalar('GSLoss/normal_loss', Ll1_normal, self.gs_iter)

        log_freq = self.conf.get_int('gs.opt.log_freq')
        save_gs_freq = self.conf.get_int('gs.opt.save_gs_freq')
        # Log and save
        if iteration % log_freq == 0:
            print(f'[ITER {iteration}] GS loss {loss}')
        if iteration % save_gs_freq == 0:
            print(f"[ITER {iteration}] Saving Gaussians")
            self.scene.save(iteration)

        # Densification
        if not is_inter:
            self.gs_densify_wo_inter(iteration, viewspace_point_tensor, visibility_filter, radii)
        else:
            self.gs_densify_w_inter(iteration)

        # Optimizer step
        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none = True)

        save_checkpoint_freq = self.conf.get_int('gs.opt.save_checkpoint_freq')

        if iteration % save_checkpoint_freq == 0:
            print("[ITER {}] Saving Checkpoint".format(iteration))
            torch.save((self.gaussians.capture(), iteration), self.scene.model_path + "/chkpnt" + str(iteration) + ".pth")

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.WARNING, format=FORMAT)

    parser = ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument("--gs_checkpoint", type=str, default=None)
    parser.add_argument("--neus_checkpoint", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='train')

    # parser.add_argument('--debug_from', type=int, default=-1)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    # parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    # parser.add_argument('--mode', type=str, default='train')
    
    args = parser.parse_args()

    safe_state(args.quiet)
    torch.cuda.set_device(args.gpu)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    runner = Runner(args.conf, args.case, args.gs_checkpoint, args.neus_checkpoint)
    # runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'debug':
        runner.debug()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512)
    elif args.mode == 'validate_gs_image':
        runner.validate_gs_image(idx = 70)
    # elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
    #     _, img_idx_0, img_idx_1 = args.mode.split('_')
    #     img_idx_0 = int(img_idx_0)
    #     img_idx_1 = int(img_idx_1)
    #     runner.interpolate_view(img_idx_0, img_idx_1)
