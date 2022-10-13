# visualize_util
<img src="https://github.com/Vertical-Beach/ai-edge-contest6/blob/visualize/dataset_util/data/img1.png?raw=true" width="50%">

## センサ情報の生成
mmdetection3dの`create_data.py`を使用して`nuscenes_infos_train.pkl, nuscenes_infos_val.pkl`を生成する。  
このデータにはLidarやカメラの情報が含まれている。ダウンロードする際はこの手順をスキップすることができる。  
Vitis-AIの`pt_pointpillars_nuscenes_40000_64_108G_2.5`の中のREADME.mdの手順に従って行う
```bash
conda activate vitis-ai-pytorch
sudo apt-get update && sudo apt-get install -y cuda-toolkit-11-3
export CUDA_HOME=/usr/local/cuda    
sudo update-alternatives --config gcc #gcc7を選択
cd code/mmcv
MMCV_WITH_OPS=1 pip install --user -e .  
cd ..
cd mmdetection
pip install --user -r requirements/build.txt
pip install --user -v -e . 
cd ..
cd mmdetection3d
pip install --user -r requirements.txt
pip install --user -v -e . 
cd ../../
pip uninstall pycocotools
pip uninstall mmpycocotools
pip install --user mmpycocotools

export PYTHONPATH=${PWD}/code/mmdetection3d:${PYTHONPATH}
```
`tools/data_converter/nuscenes_converter.py`の`camera_types`を`CAM_FRONT`のみに変更する（SIGNATEデータセットではカメラは1つしかないので）
```sh
python code/mmdetection3d/tools/create_data.py nuscenes --root-path /workspace/signat_dataset --out-dir /workspace/signat_dataset --extra-tag nuscenes
```

`AssertionError: Database version not found: /workspace/signat_dataset/v1.0-test`が出ても`v1.0-trainval`に関してはできてるので気にしない

## nuscenes_infos_train.pkl, nuscenes_infos_val.pklの中身
```
{
    "infos":[
        {
            "token": str (sample token),
            "gt_boxes": np.ndarray (boxnum, 7),
            "gt_names": List[str] len=boxnum,
            "lidar2ego_translation": List[float] len=3,
            "lidar2ego_rotation": List[float] len=4,
            "ego2global_translation": List[float] len=3,
            "ego2global_rotation": List[float] len=4,
            "lidar_path": str (path to lidar .bin data),
            "num_lidar_pts": np.ndarray (boxnum),
            "num_radar_pts": np.ndarray (boxnum),
            "cams":{
                "CAM_FRONT":{
                    "data_path": str (path to camera image .jpg),
                    "sample_data_token": str (camera sample token),
                    "sensor2ego_translation": List[float] len=3,
                    "sensor2ego_rotation": List[float] len=4,
                    "ego2global_translation": List[float] len=3,
                    "ego2global_rotation": List[float] len=4,
                    "sensor2lidar_translation": List[float] len=3,
                    "sensor2lidar_rotation": List[List[float]] len=3x3,
                }
            }
        },
        {

        },...
    ]
}
```
pathはdockerコンテナ内でのフルパスが指定されているので適宜可視化スクリプト側で書き換える必要がある

## 可視化環境のセットアップ
dockerコンテナ不使用、venvのみ使用  
Python 3.9.12を使用
```sh
python -m venv .venv
source .venv/bin/activate
pip install opencv-python
pip install easydict
pip install open3d
```
## アノテーションデータの可視化
trainまたはvalのデータセットを可視化する。2dまたは3d、あるいは両方を実行して画像を連結する
```
usage: visualize_gt.py [-h] [--out_dir OUT_DIR] [--train_or_val {train,val}] [--dataset_dir DATASET_DIR] [--mode {2d,3d,both}]

optional arguments:
  -h, --help            show this help message and exit
  --out_dir OUT_DIR
  --train_or_val {train,val}
  --dataset_dir DATASET_DIR
  --mode {2d,3d,both}
```

```sh
python visualize_gt.py --out_dir res --train_or_val val --dataset_dir /media/lp6m/HDD6TB/aiedge6/materials/train/3d_labels/ --mode both
```


## 推論結果の可視化
推論結果のjsonを読み込んで可視化する。推論した点群データがtrain/valのどちらに含まれるかは現状指定する必要がある。
```
usage: visualize_pred.py [-h] --out_dir OUT_DIR --train_or_val {train,val} --dataset_dir DATASET_DIR --result_json RESULT_JSON --mode {2d,3d,both} [--score_thresh SCORE_THRESH]

optional arguments:
  -h, --help            show this help message and exit
  --out_dir OUT_DIR
  --train_or_val {train,val}
  --dataset_dir DATASET_DIR
  --result_json RESULT_JSON
  --mode {2d,3d,both}
  --score_thresh SCORE_THRESH
```

```
python visualize_pred.py --out_dir pred --train_or_val val --dataset_dir /media/lp6m/HDD6TB/aiedge6/materials/train/3d_labels/ --result_json ./data/result.json --mode both
```

## Open3DでのGUI可視化
<img src="https://github.com/Vertical-Beach/ai-edge-contest6/blob/visualize/dataset_util/data/img2.png?raw=true" width="50%">
open3dでマウスでグリグリカメラを回して可視化したい場合に使用する。  
`--lidar_file`に指定した点群ファイル1つを表示する。
### アノテーションデータの可視化
```
python visualize_gui.py --train_or_val val --dataset_dir /media/lp6m/HDD6TB/aiedge6/materials/train/3d_labels/ --mode gt --lidar_file em5VCQcE1fwFkTHI4wZ0Tm5y_0.bin
```

### 推論結果の可視化
指定した点群ファイルに対応する推論結果が`--result_json`で指定される推論結果に含まれている必要がある。
```
python visualize_gui.py --train_or_val val --dataset_dir /media/lp6m/HDD6TB/aiedge6/materials/train/3d_labels/ --mode pred --lidar_file em5VCQcE1fwFkTHI4wZ0Tm5y_0.bin --result_json ./data/result.json
```

## 注意点
bboxの座標、サイズ、向きの情報は以下の7つの値で表される。  
`(center_x, center_y, center_z, size_x, size_y, size_z, y-axis-rotation)`  
DPUの推論結果（正確にはVitis-AI-Libraryのpotprocess）はなぜか、
`(center_x, center_y, min_z, size_x, size_y, size_z, y-axis-rotation)`のようにz座標だけ**min_z**になっているので可視化前に変換する必要がある。実装は`visualize_pred.py::transform_bbox()`にある。