import json

import numpy as np

from predictor import ScoringService


def save_results_json_for_visualization(pts_filename, pred, output_path):
    boxes_3d, scores_3d, labels_3d = pred
    res = {}
    res["lidar_file"] = pts_filename
    res["pred"] = [
        {
            "label": label.tolist(),
            "box": bbox.tolist(),
            "score": score.tolist()
        }
        for label, bbox, score in zip(labels_3d, boxes_3d, scores_3d)
    ]
    open(output_path, "w").write(json.dumps([res]))


def test_signate_postprocess(pred, output_path, thresh=0.2):
    boxes_3d, scores_3d, labels_3d = pred
    lidar_pos = np.array([0, 0])
    resdict = ScoringService.postprocess(
        boxes_3d, scores_3d, labels_3d, lidar_pos, thresh=0.2)
    print(resdict)
    open(output_path, "w").write(json.dumps(resdict))


pts_filename = "../Vitis-AI/signate_dataset/samples/LIDAR_TOP/0TyydnMdYWU1YD7nw5uCNGs8_1.bin"
ScoringService.get_model("dummy")
pred = ScoringService.run(pts_filename)
save_results_json_for_visualization(pts_filename, pred, "result.json")

test_signate_postprocess(pred, "result_postprocess.json", thresh=0.2)
