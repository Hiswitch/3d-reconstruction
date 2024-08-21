from pathlib import Path
import pycolmap
from argparse import ArgumentParser
from pose_utils import load_colmap_data, save_poses
import numpy as np
import trimesh
import os
import glob
import cv2 as cv
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import shutil

def remove_outliers_iqr(points):
    Q1 = np.percentile(points, 25, axis=0)
    Q3 = np.percentile(points, 75, axis=0)
    IQR = Q3 - Q1
    mask = np.all((points >= (Q1 - 1.5 * IQR)) & (points <= (Q3 + 1.5 * IQR)), axis=1)
    return points[mask]

def sfm(base_dir:Path):
    imgs_dir = base_dir / 'images'
    outputs_dir = base_dir / 'sparse' /'0'
    outputs_dir.mkdir(parents=True, exist_ok=True)
    database_path = outputs_dir / 'database.db'
    # mvs_dir = base_dir / 'undistort'

    pycolmap.extract_features(database_path, imgs_dir, camera_model='PINHOLE')
    pycolmap.match_exhaustive(database_path)
    maps = pycolmap.incremental_mapping(database_path, imgs_dir, outputs_dir)
    maps[0].write(outputs_dir)

    # pycolmap.undistort_images(mvs_dir, outputs_dir, imgs_dir)
    # shutil.move(mvs_dir / 'images', base_dir)
    # sfm_dir = base_dir / 'sparse' / '0'
    # sfm_dir.mkdir(parents=True, exist_ok=True)
    # for item in (mvs_dir / 'sparse').glob('*.bin'):
    #     shutil.move(item, sfm_dir)


def get_cam_mask(base_dir:Path):
    poses, pts3d, perm, real_ids = load_colmap_data(base_dir)
    save_poses(base_dir, poses, pts3d, perm, real_ids)
    work_dir = base_dir
    poses_hwf = np.load(os.path.join(work_dir, 'poses.npy')) # n_images, 3, 5
    poses_raw = poses_hwf[:, :, :4]
    hwf = poses_hwf[:, :, 4]
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose[:3, :4] = poses_raw[0]
    pts = []
    pts.append((pose @ np.array([0, 0, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([1, 0, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([0, 1, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([0, 0, 1, 1])[:, None]).squeeze()[:3])
    pts = np.stack(pts, axis=0)
    pcd = trimesh.PointCloud(pts)
    pcd.export(os.path.join(work_dir, 'pose.ply'))

    cam_dict = dict()
    n_images = len(poses_raw)

    convert_mat = np.zeros([4, 4], dtype=np.float32)
    convert_mat[0, 1] = 1.0
    convert_mat[1, 0] = 1.0
    convert_mat[2, 2] =-1.0
    convert_mat[3, 3] = 1.0

    for i in range(n_images):
        pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        pose[:3, :4] = poses_raw[i]
        pose = pose @ convert_mat
        h, w, f = hwf[i, 0], hwf[i, 1], hwf[i, 2]
        intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
        intrinsic[0, 2] = (w - 1) * 0.5
        intrinsic[1, 2] = (h - 1) * 0.5
        w2c = np.linalg.inv(pose)
        world_mat = intrinsic @ w2c
        world_mat = world_mat.astype(np.float32)
        cam_dict['camera_mat_{}'.format(i)] = intrinsic
        cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
        cam_dict['world_mat_{}'.format(i)] = world_mat
        cam_dict['world_mat_inv_{}'.format(i)] = np.linalg.inv(world_mat)

    pcd = trimesh.load(os.path.join(work_dir, 'sparse_points_interest.ply'))
    vertices = pcd.vertices
    # vertices_f = remove_outliers_iqr(vertices)
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)
    center = (bbox_max + bbox_min) * 0.5
    radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
    print(radius, center)
    scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    scale_mat[:3, 3] = center

    for i in range(n_images):
        cam_dict['scale_mat_{}'.format(i)] = scale_mat
        cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)

    out_dir = work_dir
    os.makedirs(os.path.join(out_dir, 'masks'), exist_ok=True)

    with open(base_dir / 'names.txt', 'r') as file:
        images_path_real = [os.path.join(work_dir, 'images', line.strip()) for line in file.readlines()]

    images_path = glob.glob(os.path.join(work_dir, 'images/*'))
    images_path_delete = list(set(images_path) - set(images_path_real))
    delete_dir = os.path.join(work_dir, 'delete/')
    os.makedirs(delete_dir, exist_ok=True)
    for image_path in images_path_delete:
        shutil.move(image_path, delete_dir)

    for i, image_path in enumerate(images_path_real):
        img = cv.imread(image_path)
        cv.imwrite(os.path.join(out_dir, 'masks', f'{os.path.basename(image_path)[:-4]}.png'), np.ones_like(img) * 255)

    np.savez(os.path.join(out_dir, 'cameras_sphere.npz'), **cam_dict)

def get_normal(base_dir:Path):
    task = 'normal'
    img_path = base_dir / 'images'
    output_path = base_dir / 'normals'
    output_path.mkdir(parents=True, exist_ok=True)

    command = [
        'python', './ominidata/demo.py',
        '--task', task,
        '--img_path', img_path,
        '--output_path', output_path
    ]
    # 使用 subprocess 运行命令
    subprocess.run(command, capture_output=True, text=True)

def get_edge(base_dir:Path):
    savedir = str(base_dir / 'edges')
    datadir = str(base_dir / 'images')
    command = [
        'python', './pidinet/main.py',
        '--model', 'pidinet_converted',
        '--config', 'carv4',
        '--sa',
        '--dil',
        '-j', '4',
        '--gpu', '0',
        '--savedir', savedir,
        '--datadir', datadir,
        '--dataset', 'Custom',
        '--evaluate', './pidinet/trained_models/table5_pidinet.pth',
        '--evaluate-converted'
    ]

    # 使用 subprocess 运行命令
    subprocess.run(command, capture_output=True, text=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="room0")
    args = parser.parse_args()
    dataset = args.dataset

    base_dir = Path(f'../data/{dataset}')
    imgs_dir = base_dir / 'images'

    if not (base_dir / 'sparse' / '0').exists():
        sfm(base_dir)

    if not (base_dir / 'cameras_sphere.npz').exists():
        get_cam_mask(base_dir)

    if not (base_dir / 'normals').exists():
        get_normal(base_dir)

    if not (base_dir / 'edges').exists():
        get_edge(base_dir)

    

    


