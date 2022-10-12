import os

import cv2
import numpy as np
from easydict import EasyDict

COLORMAP = [(255, 158, 0),  # Orange
            (255, 99, 71),  # Tomato
            (255, 140, 0),  # Darkorange
            (255, 69, 0),  # Orangered
            (233, 150, 70),  # Darksalmon
            (220, 20, 60),  # Crimson
            (255, 61, 99),  # Red
            (0, 0, 230),  # Blue
            (47, 79, 79),  # Darkslategrey
            (112, 128, 144),  # Slategrey
            ]


def rotz(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  -s,  0],
                     [s,   c,  0],
                     [0,   0,  1]])


def isAnyBack(corners, forward_axis):
    return any(corners[:, forward_axis] < 0)


def generate8Corners(x, y, z, dx, dy, dz, rz, gt_mode):
    size = np.array([dx, dy, dz])
    center = np.array([x, y, z])
    if gt_mode:
        offset = np.array([
            [1/2,    1/2,  1/2],
            [-1/2,    1/2,  1/2],
            [-1/2,   -1/2,  1/2],
            [1/2,   -1/2,  1/2],
            [1/2,    1/2,  -1/2],
            [-1/2,    1/2,  -1/2],
            [-1/2,   -1/2,  -1/2],
            [1/2,   -1/2,  -1/2]
        ])
    else:
        offset = np.array([
            [1/2,    1/2,  0],
            [-1/2,    1/2,  0],
            [-1/2,   -1/2,  0],
            [1/2,   -1/2,  0],
            [1/2,    1/2,  1],
            [-1/2,    1/2,  1],
            [-1/2,   -1/2,  1],
            [1/2,   -1/2,  1]
        ])
    offset *= np.tile(size, (8, 1))
    offset = np.dot(rotz(rz), offset.T).T
    corners = offset + np.tile(center, (8, 1))
    return corners


def convert(boxes_3d, scores_3d, labels_3d):
    objs = []
    for label, score, loc, dim, rz in zip(labels_3d, scores_3d, boxes_3d[:, :3], boxes_3d[:, 3:6], boxes_3d[:, 6]):
        x, y, z = loc
        dx, dy, dz = dim
        a = rz
        obj = EasyDict()
        obj.x, obj.y, obj.z = loc
        obj.dx, obj.dy, obj.dz = dim
        obj.a = a
        obj.score = score
        obj.label = label
        objs.append(obj)
    return objs


def draw_proj_box(img, obj, l2c, intrinsic, colormap, gt_mode):
    ''' draw projected 3d bounding boxe on the rgb image of camera
    img: rgb image
    obj: 3d bounding box in the Camera Coordinate for KITTI
    P  : matrix of transformation from the Camera Coordinate to the image plane (P2 * rect)
    '''
    corners = generate8Corners(
        obj.x, obj.y, obj.z, obj.dx, obj.dy, obj.dz, obj.a, gt_mode)
    homo_corners = np.hstack([corners, np.array([1]*8).reshape(-1, 1)])
    corners_on_cam = np.dot(l2c, homo_corners.T).T

    if isAnyBack(corners_on_cam, forward_axis=2):
        return img
    proj_pts = project_3d_pts(corners_on_cam, intrinsic)
    img = draw_projected_box3d(
        img, proj_pts, color=colormap[obj.label], thickness=2)
    return img


def project_3d_pts(corners, P):
    homo_pts = corners
    proj_pts = np.dot(P, homo_pts.T).T
    pc_img = np.zeros((8, 3))
    pc_img[:, 0] = proj_pts[:, 0] / proj_pts[:, 2]
    pc_img[:, 1] = proj_pts[:, 1] / proj_pts[:, 2]
    pc_img[:, 2] = proj_pts[:, 2]
    return pc_img


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=1):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0, 4):

        i, j = k, (k+1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                 qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k+4, (k+1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                 qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k, k+4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                 qs[j, 1]), color, thickness, cv2.LINE_AA)
    return image


def draw_3d_boxes_on_image(boxes_3d, scores_3d, labels_3d, cam_info, gt_mode=True, score_thresh=0.5):
    img_path = cam_info['data_path']
    c2l = np.eye(4)
    c2l[:3, :3] = cam_info['sensor2lidar_rotation']
    c2l[:3, 3] = cam_info['sensor2lidar_translation']
    l2c = np.linalg.inv(c2l)
    intrinsic = np.hstack(
        [cam_info['cam_intrinsic'], np.array([[0], [0], [0]])])

    rgb_3d_map = cv2.imread(img_path)
    dt_objs = convert(boxes_3d, scores_3d, labels_3d)
    for obj in dt_objs:
        if obj.score < score_thresh:
            continue
        rgb_3d_map = draw_proj_box(
            rgb_3d_map, obj, l2c, intrinsic, COLORMAP, gt_mode)
    return rgb_3d_map
