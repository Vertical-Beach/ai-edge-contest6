import argparse
import json
import os
import pickle
from random import choice
from typing import Optional

import cv2
import numpy as np
import open3d

from util import visualize_2d, visualize_3d
from util.info_util import load_info, process_info


def transform_bbox(bbox):
    # z coordinate format is different from ground truth format.
    cx, cy, jimenz, sx, sy, sz, rot = bbox
    cz = jimenz + sz / 2
    bbox = np.array([cx, cy, cz, sx, sy, sz, rot])
    return bbox


def load_pred_json(json_path: str, score_thresh: float):
    preddict = {}
    jobj = json.loads(open(json_path).read())
    for obj in jobj:
        lidar_file = obj["lidar_file"]
        boxes = []
        scores = []
        labels = []
        for pred in obj["pred"]:
            if pred["score"] < score_thresh:
                continue
            boxes.append(transform_bbox(pred["box"][:7]))
            scores.append(pred["score"])
            labels.append(pred["label"])
        preddict[lidar_file] = (
            np.array(boxes),
            np.array(scores),
            np.array(labels),
        )
    return preddict


def save_camera_param_callback(vis):
    ctl = vis.get_view_control()
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    open3d.io.write_pinhole_camera_parameters("params.json", param)


def visualize_open3d_gui(
    train_or_val: str,
    dataset_dir: str,
    lidar_file_name: str,
    gt_mode: bool,
    result_json_path: Optional[str] = None,
    score_thresh: float = 0
):

    info = load_info(dataset_dir, train_or_val)

    # create open3d and setup camera
    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1024, height=1024)
    ctr = vis.get_view_control()
    # press s key to save camera parameter as json file
    vis.register_key_callback(83, save_camera_param_callback)
    open3d_camparam = open3d.io.read_pinhole_camera_parameters(
        "./data/open3d_cam_param.json")

    cam_info = None
    boxes_3d = None
    labels_3d = None
    scores_3d = None
    points = None

    # load gt and meta data
    for d in info["infos"]:
        if lidar_file_name == os.path.basename(d["lidar_path"]):
            cam_info, boxes_3d, labels_3d, scores_3d, points = process_info(
                d, dataset_dir)
            break
    assert cam_info is not None, f"sample {lidar_file_name} is not found in dump data"

    if not gt_mode:
        # load pred data
        preddict = load_pred_json(result_json_path, score_thresh)
        assert lidar_file_name in preddict, f"sample {lidar_file_name} is not found in result json file"
        boxes_3d, scores_3d, labels_3d = preddict[lidar_file_name]

    visualize_3d.draw_scenes(
        vis, points, gt_boxes=boxes_3d, ref_labels=labels_3d, ref_scores=scores_3d)

    ctr.convert_from_pinhole_camera_parameters(
        open3d_camparam)  # なぜかここで設定する必要がある
    vis.run()

    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_or_val", type=str,
                        choices=["train", "val"], required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["gt", "pred"])
    parser.add_argument("--lidar_file", type=str, required=True)
    parser.add_argument("--result_json", type=str,
                        help="only used when --mode=pred")
    parser.add_argument("--score_thresh", type=float, default=0,
                        help="only used when --mode=pred")

    args = parser.parse_args()

    gt_mode = (args.mode == "gt")
    lidar_file = os.path.basename(args.lidar_file)
    visualize_open3d_gui(
        args.train_or_val, args.dataset_dir, lidar_file, gt_mode, args.result_json, args.score_thresh)
