from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval
)

import os
import h5py
import numpy as np
import cv2

from scipy.optimize import minimize
import torch

def get_image_ids(image_dir: Path):
    images = {}
    image_paths = sorted(image_dir.glob('*'))

    for i, image_path in enumerate(image_paths):
        images[os.path.basename(image_path)] = i + 1

    return images

def get_keypoints(path, name, return_uncertainty=False):
    with h5py.File(str(path), "r", libver="latest") as hfile:
        dset = hfile[name]["keypoints"]
        p = dset.__array__()
        uncertainty = dset.attrs.get("uncertainty")
    if return_uncertainty:
        return p, uncertainty
    return p

def get_pairs(path):
    with open(str(path), "r") as f:
        pairs = [p.split() for p in f.readlines()]
    return pairs

def names_to_pair(name0, name1, separator="/"):
    return separator.join((name0.replace("/", "-"), name1.replace("/", "-")))

def find_pair(hfile: h5py.File, name0: str, name1: str):
    pair = names_to_pair(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(
        f"Could not find pair {(name0, name1)}... "
        "Maybe you matched with a different list of pairs? "
    )

def get_matches(path, name0, name1, return_scores = False):
    with h5py.File(str(path), "r", libver="latest") as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        matches = hfile[pair]["matches0"].__array__()
        scores = hfile[pair]["matching_scores0"].__array__()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    if reverse:
        matches = np.flip(matches, -1)
    scores = scores[idx]
    if return_scores:
        return matches, scores
    return matches

def visualize_keypoints(image_dir, name, features, export_dir):
    points = get_keypoints(features, name)
    image_path = os.path.join(image_dir, name)

    image = cv2.imread(image_path)
    for point in points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
    
    export_path = os.path.join(export_dir, name)
    cv2.imwrite(export_path, image)

def visualize_matches(image_dir, name0, name1, features, matches, export_dir):
    image_path0 = os.path.join(image_dir, name0)
    image_path1 = os.path.join(image_dir, name1)
    points0 = get_keypoints(features, name0)
    points1 = get_keypoints(features, name1)
    matches, scores = get_matches(matches, name0, name1, True)

    img0 = cv2.imread(image_path0)
    img1 = cv2.imread(image_path1)

    height0, width0 = img0.shape[:2]
    height1, width1 = img1.shape[:2]
    combined_img = np.zeros((max(height0, height1), width0 + width1, 3), dtype=np.uint8)
    combined_img[:height0, :width0] = img0
    combined_img[:height1, width1:width1 + width1] = img1

    for (id0, id1), score in zip(matches, scores):
        if score < 0.95:
            continue
        x0, y0 = int(points0[id0][0]), int(points0[id0][1])
        x1, y1 = int(points1[id1][0]), int(points1[id1][1])

        x1 = x1 + width0
        cv2.circle(combined_img, (x0, y0), radius=5, color=(0, 0, 255), thickness=-1) 
        cv2.circle(combined_img, (x1, y1), radius=5, color=(0, 0, 255), thickness=-1)

        cv2.line(combined_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
    
    export_path = os.path.join(export_dir, name0.split('.')[0] + '_' + name1)
    cv2.imwrite(export_path, combined_img)

def rotation_matrix_2d(rad):
    return np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)]
    ])

def affine_epipolar_geometry_2_views(name0, name1, features, matches):
    print(f"image0: {name0}, image1: {name1}")
    points0 = get_keypoints(features, name0)
    points1 = get_keypoints(features, name1)
    matches, scores = get_matches(matches, name0, name1, True)

    r = []
    for (id0, id1), score in zip(matches, scores):
        # if score < 0.95:
        #     continue
        x0, y0 = points0[id0]
        x1, y1 = points1[id1]

        ri = np.array([[x1],
                       [y1],
                       [x0],
                       [y0]], dtype=np.float32)
        r.append(ri)

    r_mean = np.mean(r, axis=0)
    v = [r[i] - r_mean for i in range(len(r))]
    print(f"num of interest points: {len(v)}")

    W = np.zeros((4, 4))
    for i in range(len(r)):
        W += np.dot(v[i], v[i].T)
    
    rank_W = np.linalg.matrix_rank(W)
    print(f"rank of W: {rank_W}")

    eigvals, eigvecs = np.linalg.eig(W)
    min_index = np.argmin(eigvals)
    min_eigval = eigvals[min_index]
    min_eigvec = eigvecs[:, min_index]
    print(f"min eigval: {min_eigval}, min eigvec: {min_eigvec}")

    n = min_eigvec
    e = -np.dot(n.T, r_mean)
    print(f"e: {e}")

    covariance_matrix = np.zeros((4, 4))
    for i in range(len(eigvals)):
        if i != min_index:
            covariance_matrix += np.dot(eigvecs[:, i], eigvecs[:, i].T) / eigvals[i]
    print(f"covariance matrix: \n{covariance_matrix}")

    s = np.sqrt(min_eigvec[3]**2 + min_eigvec[2]**2) / np.sqrt(min_eigvec[1]**2 + min_eigvec[0]**2)
    axis_projection = np.arctan(min_eigvec[1] / min_eigvec[0])
    cyclotorsion = axis_projection - np.arctan(min_eigvec[3] / min_eigvec[2])
    rotation_axis = np.array([[np.cos(axis_projection)],
                              [np.sin(axis_projection)]])
    rp = 0
    rn = 0
    for i in range(len(v)):
        delta_x1 = np.array([[v[i][0]],
                             [v[i][1]]])
        delta_x0 = np.array([[v[i][2]],
                             [v[i][3]]])
        
        rn += (np.dot(delta_x1.T, rotation_axis) - s * np.dot(np.dot(delta_x0.T, rotation_matrix_2d(-cyclotorsion)), rotation_axis)) ** 2
        rp += (np.dot(delta_x1.T, rotation_axis) + s * np.dot(np.dot(delta_x0.T, rotation_matrix_2d(-cyclotorsion)), rotation_axis)) ** 2
    if rp < rn:
        cyclotorsion += np.pi
    print(f"scale: {s}, axis projection: {axis_projection}, cyclotorsion: {cyclotorsion}")

    return s, axis_projection, cyclotorsion

def get_common_matches(path, name0, name1, name2):
    matches01 = get_matches(path, name0, name1)
    matches02 = get_matches(path, name0, name2)

    matches_dict01 = {}
    matches_dict02 = {}
    for match in matches01:
        matches_dict01[match[0]] = match[1]
    for match in matches02:
        matches_dict02[match[0]] = match[1]
    
    matches012 = []
    for match in matches01:
        if match[0] in matches_dict02:
            matches012.append([match[0], match[1], matches_dict02[match[0]]])
    
    return np.array(matches012)

def affine_epipolar_geometry_3_views(name0, name1, name2, features, matches):
    s01, axis_projection01, cyclotorsion01 = affine_epipolar_geometry_2_views(name0, name1, features, matches)
    s02, axis_projection02, cyclotorsion02 = affine_epipolar_geometry_2_views(name0, name2, features, matches)

    def D(axis_projection, cyclotorsion):
        return np.dot(np.array([[np.cos(axis_projection)], [np.sin(axis_projection)]]),
                      np.array([[np.cos(axis_projection - cyclotorsion), np.sin(axis_projection - cyclotorsion)]]))
    def E(cyclotorsion, D):
        return np.array([[np.cos(cyclotorsion), -np.sin(cyclotorsion)],
                         [np.sin(cyclotorsion), np.cos(cyclotorsion)]]) - D
    def P(axis_projection):
        return np.array([[np.sin(axis_projection)],
                         [-np.cos(axis_projection)]])
    
    D01 = D(axis_projection01, cyclotorsion01)
    E01 = E(cyclotorsion01, D01)
    P01 = P(axis_projection01)
    D02 = D(axis_projection02, cyclotorsion02)
    E02 = E(cyclotorsion02, D02)
    P02 = P(axis_projection02)

    matches012 = get_common_matches(matches, name0, name1, name2)
    keypoints0 = get_keypoints(features, name0)
    keypoints1 = get_keypoints(features, name1)
    keypoints2 = get_keypoints(features, name2)

    r = []
    for i in range(len(matches012)):
        id0, id1, id2 = matches012[i]
        r.append(np.hstack((keypoints0[id0].T, keypoints1[id1].T, keypoints2[id2].T)))
    
    r_mean = np.mean(r, axis=0)
    v = [r[i] - r_mean for i in range(len(matches012))]

    def ETK3(rho_list):
        def temp(s, D, E, P, rho):
            t1 = np.array(s * (D + np.cos(rho) * E))
            t2 = np.array(s * np.sin(rho) * P)
            return np.hstack((t1, t2))
        
        L = np.vstack((np.array([[1, 0, 0], [0, 1, 0]]), temp(s01, D01, E01, P01, rho_list[0]), temp(s02, D02, E02, P02, rho_list[1])))
        res = 0
        for i in range(len(v)):
            res += np.linalg.norm(v[i] - L @ np.linalg.pinv(L.T @ L) @ L.T @ v[i]) ** 2

        return res
    
    # print(ETK3([0, 0]))
    
    rho_list = minimize(ETK3, [0, 0], method = 'Nelder-Mead')
    print(rho_list)
    
import torch
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_points(points):
    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 保存图像
    plt.savefig('3d_points_matplotlib.png')

def affine_epipolar_geometry_3_views_tensor(name0, name1, name2, features, matches):
    s01, axis_projection01, cyclotorsion01 = affine_epipolar_geometry_2_views(name0, name1, features, matches)
    s02, axis_projection02, cyclotorsion02 = affine_epipolar_geometry_2_views(name0, name2, features, matches)
    
    # Convert values to tensors
    axis_projection01 = torch.tensor(axis_projection01, dtype=torch.float64)
    cyclotorsion01 = torch.tensor(cyclotorsion01, dtype=torch.float64)
    axis_projection02 = torch.tensor(axis_projection02, dtype=torch.float64)
    cyclotorsion02 = torch.tensor(cyclotorsion02, dtype=torch.float64)
    
    keypoints0 = torch.tensor(get_keypoints(features, name0), dtype=torch.float64)
    keypoints1 = torch.tensor(get_keypoints(features, name1), dtype=torch.float64)
    keypoints2 = torch.tensor(get_keypoints(features, name2), dtype=torch.float64)
    
    def D(axis_projection, cyclotorsion):
        axis_projection = axis_projection.unsqueeze(0)
        cos_axis_projection = torch.cos(axis_projection)
        sin_axis_projection = torch.sin(axis_projection)
        cos_diff = torch.cos(axis_projection - cyclotorsion)
        sin_diff = torch.sin(axis_projection - cyclotorsion)
        return torch.matmul(torch.tensor([[cos_axis_projection], [sin_axis_projection]]),
                            torch.tensor([[cos_diff, sin_diff]]))
    
    def E(cyclotorsion, D):
        cos_cyclotorsion = torch.cos(cyclotorsion)
        sin_cyclotorsion = torch.sin(cyclotorsion)
        R = torch.tensor([[cos_cyclotorsion, -sin_cyclotorsion],
                          [sin_cyclotorsion, cos_cyclotorsion]])
        return R - D
    
    def P(axis_projection):
        sin_axis_projection = torch.sin(axis_projection)
        cos_axis_projection = torch.cos(axis_projection)
        return torch.tensor([[sin_axis_projection], [-cos_axis_projection]])
    
    D01 = D(axis_projection01, cyclotorsion01)
    E01 = E(cyclotorsion01, D01)
    P01 = P(axis_projection01)
    D02 = D(axis_projection02, cyclotorsion02)
    E02 = E(cyclotorsion02, D02)
    P02 = P(axis_projection02)

    matches012 = get_common_matches(matches, name0, name1, name2)
    print(f"num of interest point: {len(matches012)}")

    r = []
    for i in range(len(matches012)):
        id0, id1, id2 = matches012[i]
        r.append(torch.cat((keypoints0[id0], keypoints1[id1], keypoints2[id2]), dim=0))
    
    r = torch.stack(r)
    r_mean = torch.mean(r, dim=0)
    v = [r[i] - r_mean for i in range(len(matches012))]

    def Lf(rho_list):
        rho_list = torch.tensor(rho_list, dtype=torch.float64)
        def temp(s, D, E, P, rho):
            t1 = s * (D + torch.cos(rho) * E)
            t2 = s * torch.sin(rho) * P
            return torch.cat((t1, t2), dim=1)
        
        L = torch.cat((
            torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float64),
            temp(s01, D01, E01, P01, rho_list[0]),
            temp(s02, D02, E02, P02, rho_list[1])
        ), dim=0)
        
        return L
    
    def ETK3(rho_list):
        L = Lf(rho_list)
        
        L_pseudo_inv = torch.linalg.pinv(L.T @ L) @ L.T
        res = 0
        for i in range(len(v)):
            residual = v[i] - L @ L_pseudo_inv @ v[i]
            res += torch.norm(residual) ** 2
        
        return res.item()

    rho_list = minimize(ETK3, [0, 0], method='Nelder-Mead').x
    print(rho_list)

    L = Lf(rho_list)
    X = []
    for i in range(len(v)):
        L_pseudo_inv = torch.linalg.pinv(L.T @ L) @ L.T
        delta_Xi = L_pseudo_inv @ v[i]
        X.append(delta_Xi.unsqueeze(0).numpy())
    X = np.vstack(X)
    visualize_3d_points(X)


if __name__ == "__main__": 
    images = Path("data/orth/")

    outputs = Path("outputs/orth/")
    sfm_pairs = outputs / "pairs-netvlad.txt"
    sfm_dir = outputs / "sfm_superpoint+superglue"

    # retrieval_conf = extract_features.confs["netvlad"]
    # feature_conf = extract_features.confs["superpoint_aachen"]
    # matcher_conf = match_features.confs["superglue"]

    # retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    # pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

    # feature_path = extract_features.main(feature_conf, images, outputs)
    # match_path = match_features.main(
    #     matcher_conf, sfm_pairs, feature_conf["output"], outputs
    # )

    feature_path = Path("/mnt/data3/hushuaiwei/3d-reconstruction/outputs/orth/feats-superpoint-n4096-r1024.h5")
    match_path = Path("/mnt/data3/hushuaiwei/3d-reconstruction/outputs/orth/feats-superpoint-n4096-r1024_matches-superglue_pairs-netvlad.h5")
    pairs = get_pairs(sfm_pairs)

    # affine_epipolar_geometry_2_views(pairs[0][0], pairs[0][1], feature_path, match_path)
    # affine_epipolar_geometry_3_views(pairs[0][0], pairs[0][1], pairs[1][1], feature_path, match_path)
    affine_epipolar_geometry_3_views_tensor(pairs[0][0], pairs[0][1], pairs[2][1], feature_path, match_path)

    # visualize_matches(images, pairs[0][0], pairs[0][1], feature_path, match_path, outputs)

    # model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)
