
import os

os.environ["W_QUANT"] = "0"

import torch
import numpy as np
from mmdet3d.core.bbox.structures import box_3d_mode, lidar_box3d
from mmdet3d.models import build_detector
from mmcv import Config
from mmcv.runner import load_checkpoint
from util import lidar_to_global

class ScoringService(object):
    @classmethod
    def get_model(cls, dummy):
        config_file = "../model/hv_pointpillars_secfpn_sbn-all_4x4_2x_nus-3d.py"
        weights_path = "../model/pointpillars-nus.pth"
        cfg = Config.fromfile(config_file)
        # open("cfg.txt", "w").write(cfg.pretty_text)
        cls.model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(cls.model, weights_path, map_location='cpu')
        cls.model.cuda()
        cls.model.eval()

        return True

    @classmethod
    def load_points(cls, pts_filename):
        pts = np.fromfile(pts_filename, dtype=np.float32)
        pts = pts.reshape((-1, 5))
        pts[:, 3] *= 255 # need???????
        pts = np.ascontiguousarray(pts)
        pts = torch.from_numpy(pts)
        x = torch.Tensor.cuda(pts)
        return x

    @classmethod
    def get_dummy_image_meta(cls, pts_filename):
        return {
            "flip": False,
            "pcd_horizontal_flip": False,
            "pcd_vertical_flip": False,
            "box_mode_3d": box_3d_mode.Box3DMode,
            "box_type_3d": lidar_box3d.LiDARInstance3DBoxes,
            "pcd_trans": np.array([0, 0, 0]),
            "sample_idx": "dummy",
            "pcd_scale_factor": 1.0,
            "pts_filename": pts_filename
        }

    @classmethod
    def run(cls, pts_filename):
        x = cls.load_points(pts_filename)
        img_meta = cls.get_dummy_image_meta(pts_filename)
        pred = cls.model.simple_test([x], [img_meta])[0]
        pred = pred["pts_bbox"]
        boxes_3d = pred["boxes_3d"].tensor.numpy()
        scores_3d = pred["scores_3d"].numpy()
        labels_3d = pred["labels_3d"].numpy()

        return boxes_3d, scores_3d, labels_3d

    @classmethod
    def postprocess(cls, boxes_3d, scores_3d, labels_3d, lidar_pos, thresh=0.20):
        """
        boxes_3d (N, 9)
        scores_3d (N)
        labels_3d (N)
        lidar_pos (2)
        """
        # box contains only (cx, cy)
        boxes_3d = boxes_3d[:, :2]
        # category filter
        CAR = 0
        PEDESTRIAN = 7
        target_class_flg = ((labels_3d == CAR) | (labels_3d == PEDESTRIAN))
        boxes_3d = boxes_3d[target_class_flg]
        scores_3d = scores_3d[target_class_flg]
        labels_3d = labels_3d[target_class_flg]
        # thresh filter
        thresh_ok_flg = scores_3d > thresh
        boxes_3d = boxes_3d[thresh_ok_flg]
        scores_3d = scores_3d[thresh_ok_flg]
        labels_3d = labels_3d[thresh_ok_flg]
        # 自車からの距離
        dist_from_lidar = np.linalg.norm(boxes_3d - lidar_pos, axis=1)
        ped_flg = ((labels_3d == PEDESTRIAN) & (dist_from_lidar <= 40.0))
        car_flg = ((labels_3d == CAR) & (dist_from_lidar <= 50.0))

        ped_res = [[pos[0], pos[1], score] for pos, score in zip(boxes_3d[ped_flg].tolist(), scores_3d[ped_flg].tolist())]
        car_res = [[pos[0], pos[1], score] for pos, score in zip(boxes_3d[car_flg].tolist(), scores_3d[car_flg].tolist())]
        # 各カテゴリーの数を50以下に制限
        ped_res = ped_res[:50]
        car_res = car_res[:50]

        res = {}
        if len(ped_res) > 0:
            res["pedestrian"] = ped_res
        if len(car_res) > 0:
            res["vehicle"] = car_res

        return res

    @classmethod
    def predict(cls, input):
        # inputデータを取得
        test_key = input["test_key"]
        # cam_path = input["cam_path"]
        lidar_path = input["lidar_path"]
        # cam_ego_pose = input["cam_ego_pose"]
        # cam_calibration = input["cam_calibration"]
        lidar_ego_pose = input["lidar_ego_pose"]
        lidar_calibration = input["lidar_calibration"]

        lidar_pos = np.array(lidar_ego_pose['translation'][:2])

        boxes_3d, scores_3d, labels_3d = cls.run(lidar_path)
        boxes_3d = lidar_to_global(boxes_3d, lidar_ego_pose, lidar_calibration)
        res = cls.postprocess(boxes_3d, scores_3d, labels_3d, lidar_pos, thresh=0.2)

        return {test_key: res}
