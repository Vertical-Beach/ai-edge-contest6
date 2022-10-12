
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

### カメラ画像への可視化(2D)

### 鳥瞰図への可視化(3D)

## Open3Dでの可視化


