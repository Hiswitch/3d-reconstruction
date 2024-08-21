import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import copy


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def get_world_normal(normal, extrin):
    '''
    Args:
        normal: N*3
        extrinsics: 4*4, world to camera
    Return:
        normal: N*3, in world space 
    '''
    extrinsics = copy.deepcopy(extrin)
    if torch.is_tensor(extrinsics):
        extrinsics = extrinsics.cpu().numpy()
        
    assert extrinsics.shape[0] ==4
    normal = normal.transpose()
    extrinsics[:3, 3] = np.zeros(3)  # only rotation, no translation

    normal_world = np.matmul(np.linalg.inv(extrinsics),
                            np.vstack((normal, np.ones((1, normal.shape[1])))))[:3]
    normal_world = normal_world.transpose((1, 0))

    return normal_world

def imread_rgb(path):
    bgr_image = cv.imread(path)
    rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
    return rgb_image

def imread_gray(path):
    gray_image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    return gray_image

def get_edge_coords(edge):
    edge_coords = (edge > 0.8).nonzero(as_tuple=False)
    return edge_coords

class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('general.data_dir')
        self.render_cameras_name = conf.get_string('neus.dataset.render_cameras_name')
        self.object_cameras_name = conf.get_string('neus.dataset.object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'images/*')))
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([imread_rgb(im_name) for im_name in self.images_lis]) / 255.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'masks/*.png')))
        self.masks_np = np.stack([imread_rgb(im_name) for im_name in self.masks_lis]) / 255.0

        self.normals_lis = sorted(glob(os.path.join(self.data_dir, 'normals/*.png')))
        self.normals_np = np.stack([(imread_rgb(im_name) / 255.0 - 0.5) * 2 for im_name in self.normals_lis])# -> range [-1, 1]
        self.edges_lis = sorted(glob(os.path.join(self.data_dir, 'edges/*.png')))
        self.edges_np = np.stack([imread_gray(im_name) for im_name in self.edges_lis]) / 255.0 # shape [n_images, H, W]
        

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []
        # self.scale_mats_np = self.estimated_scale_mat()
        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        self.scale = camera_dict['scale_mat_0'][0, 0]
        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        # normals_np = np.stack([(imread_rgb(im_name) / 255.0 - 0.5) * 2 for im_name in self.normals_lis])
        # normals_np_world = []
        # for i, _ in enumerate(normals_np):
        #     normal_img_curr = normals_np[i]
        
        #     # transform to world coordinates
        #     ex_i = torch.linalg.inv(self.pose_all[i])
        #     img_normal_w = get_world_normal(normal_img_curr.reshape(-1, 3), ex_i).reshape(self.H, self.W, 3)

        #     normals_np_world.append(img_normal_w)
        # normals_np_world = np.stack(normals_np_world)

        self.normals = torch.from_numpy(self.normals_np.astype(np.float32)).cpu() # [n_images, H, W, 3]
        self.edges = torch.from_numpy(self.edges_np.astype(np.float32)).cpu() # [n_images, H, W]
        self.edge_coords_lis = [get_edge_coords(edge) for edge in self.edges]

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')
    
    def estimated_scale_mat(self):
        assert len(self.world_mats_np) > 0
        rays_o = []
        rays_v = []
        for world_mat in self.world_mats_np:
            P = world_mat[:3, :4]
            intrinsics, c2w = load_K_Rt_from_P(None, P)
            rays_o.append(c2w[:3, 3])
            rays_v.append(c2w[:3, 0])
            rays_o.append(c2w[:3, 3])
            rays_v.append(c2w[:3, 1])

        rays_o = np.stack(rays_o, axis=0)   # N * 3
        rays_v = np.stack(rays_v, axis=0)   # N * 3
        dot_val = np.sum(rays_o * rays_v, axis=-1, keepdims=True)  # N * 1
        center, _, _, _ = np.linalg.lstsq(rays_v, dot_val)
        center = center.squeeze()
        radius = np.max(np.sqrt(np.sum((rays_o - center[None, :])**2, axis=-1)))
        print(radius, center)
        scale_mat = np.diag([radius, radius, radius, 1.0])
        scale_mat[:3, 3] = center
        scale_mat = scale_mat.astype(np.float32)
        scale_mats = [scale_mat for _ in self.world_mats_np]

        return scale_mats

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        # """
        edge_coords = self.edge_coords_lis[img_idx].cuda()
        edge_ratio = len(edge_coords) / (self.H * self.W)
        edge_rays_num = int(batch_size * edge_ratio)
        selected_indices = torch.randperm(edge_coords.shape[0])[:edge_rays_num]
        selected_coords = edge_coords[selected_indices]
        pixels_y_edge, pixels_x_edge = selected_coords[:, 0], selected_coords[:, 1]
        
        random_rays_num = batch_size - edge_rays_num
        pixels_x_random = torch.randint(low=0, high=self.W, size=[random_rays_num])
        pixels_y_random = torch.randint(low=0, high=self.H, size=[random_rays_num])

        pixels_x = torch.cat((pixels_x_edge, pixels_x_random))
        pixels_y = torch.cat((pixels_y_edge, pixels_y_random))

        # pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        # pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])

        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3

        normal = self.normals[img_idx][(pixels_y, pixels_x)]  # batch_size, 3

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, normal, mask[:, :1]], dim=-1).cuda(), pixels_x, pixels_y     # batch_size, 13

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

