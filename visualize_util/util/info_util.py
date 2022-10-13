import os
import pickle

import numpy as np


def load_info(dataset_dir: str, train_or_val: str):
    info_path = os.path.join(dataset_dir, f"nuscenes_infos_{train_or_val}.pkl")
    assert os.path.exists(info_path), f"info {info_path} not found"
    info = pickle.load(open(info_path, "rb"))
    return info


def process_info(d: dict, dataset_dir: str):
    cam_info = d["cams"]["CAM_FRONT"]
    # rewrite
    img_name = os.path.basename(cam_info["data_path"])
    cam_info["data_path"] = os.path.join(
        dataset_dir, f"samples/CAM_FRONT/{img_name}")
    # show only pedestrian, car
    mask = np.where((d["gt_names"] == "pedestrian")
                    | (d["gt_names"] == "car"))
    boxes_3d = d["gt_boxes"][mask]
    scores_3d = np.ones(len(boxes_3d), dtype=np.float32)
    labels_3d = d["gt_names"][mask]
    labels_3d = np.where(labels_3d == "pedestrian", 0, 1)

    # load points
    lidar_name = os.path.basename(d["lidar_path"])
    lidar_path = os.path.join(
        dataset_dir, f"samples/LIDAR_TOP/{lidar_name}")
    points = np.fromfile(lidar_path, dtype=np.float32)
    points = points.reshape(-1, 5)
    points = points[:, :3]

    return cam_info, boxes_3d, labels_3d, scores_3d, points
