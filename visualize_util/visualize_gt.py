import argparse
import os
import pickle

import cv2
import numpy as np
import open3d

from util import visualize_2d, visualize_3d
from util.info_util import load_info, process_info


def visualize_2d_gt(train_or_val: str, dataset_dir: str, out_basedir: str):
    info = load_info(dataset_dir, train_or_val)
    out_dir = os.path.join(out_basedir, "2d")
    os.makedirs(out_dir, exist_ok=True)

    for d in info["infos"]:
        cam_info, boxes_3d, labels_3d, scores_3d, points = process_info(
            d, dataset_dir)
        img = visualize_2d.draw_3d_boxes_on_image(
            boxes_3d, scores_3d, labels_3d, cam_info, score_thresh=0)

        save_img_name = os.path.basename(d["lidar_path"])[:-4] + ".png"
        save_img_path = os.path.join(out_dir, save_img_name)
        print(save_img_path)
        cv2.imwrite(save_img_path, img)


def visualize_3d_gt(train_or_val: str, dataset_dir: str, out_basedir: str):
    info = load_info(dataset_dir, train_or_val)
    out_dir = os.path.join(out_basedir, "3d")
    os.makedirs(out_dir, exist_ok=True)

    # create open3d and setup camera
    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1024, height=1024)
    ctr = vis.get_view_control()
    open3d_camparam = open3d.io.read_pinhole_camera_parameters(
        "./data/open3d_cam_param.json")

    for d in info["infos"]:
        cam_info, boxes_3d, labels_3d, scores_3d, points = process_info(
            d, dataset_dir)
        visualize_3d.draw_scenes(
            vis, points, gt_boxes=boxes_3d, ref_labels=labels_3d, ref_scores=scores_3d)

        ctr.convert_from_pinhole_camera_parameters(
            open3d_camparam)  # なぜかここで設定する必要がある
        vis.poll_events()
        vis.update_renderer()

        save_img_name = os.path.basename(d["lidar_path"])[:-4] + ".png"
        save_img_path = os.path.join(out_dir, save_img_name)

        vis.capture_screen_image(save_img_path, True)
        vis.clear_geometries()

    vis.destroy_window()


def concat_2d_and_3d(train_or_val: str, dataset_dir: str, out_basedir: str):
    def hconcat_resize_max(im_list, interpolation=cv2.INTER_CUBIC):
        h_max = max(im.shape[0] for im in im_list)
        im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_max / im.shape[0]), h_max), interpolation=interpolation)
                          for im in im_list]
        return cv2.hconcat(im_list_resize)

    info = load_info(dataset_dir, train_or_val)
    out_dir = os.path.join(out_basedir, "both")
    os.makedirs(out_dir, exist_ok=True)

    for d in info["infos"]:
        save_img_name = os.path.basename(d["lidar_path"])[:-4] + ".png"
        img_2d = cv2.imread(os.path.join(out_basedir, "2d/" + save_img_name))
        img_3d = cv2.imread(os.path.join(out_basedir, "3d/" + save_img_name))

        concat_img = hconcat_resize_max([img_2d, img_3d])
        save_img_path = os.path.join(out_dir, save_img_name)
        print(save_img_path)
        cv2.imwrite(save_img_path, concat_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--train_or_val", type=str,
                        choices=["train", "val"], required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--mode", type=str,
                        choices=["2d", "3d", "both"], required=True)

    args = parser.parse_args()

    if args.mode == "2d":
        visualize_2d_gt(args.train_or_val, args.dataset_dir, args.out_dir)
    elif args.mode == "3d":
        visualize_3d_gt(args.train_or_val, args.dataset_dir, args.out_dir)
    elif args.mode == "both":
        visualize_2d_gt(args.train_or_val, args.dataset_dir, args.out_dir)
        visualize_3d_gt(args.train_or_val, args.dataset_dir, args.out_dir)
        concat_2d_and_3d(args.train_or_val, args.dataset_dir, args.out_dir)
