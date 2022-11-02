# signate_docker
SIGNATE評価投稿用ディレクトリ  
[`signatelab/runtime-gpu`](https://github.com/signatelab/runtime-gpu)で動かすための環境


# ローカルテスト用

## 環境構築
```sh
cd ..
bash signate_docker/run_docker.sh
```
以下Docker環境内で作業
```sh
cd /opt/ml/signate_docker
conda create -n devenv python=3.9 anaconda
conda init bash #シェルいったん閉じる
conda activate devenv
cd /opt/ml/signate_docker
pip install -r requirements.txt
```
## テスト
ローカルでのテストでは`lidar2global`を実行しないので出力はlidar座標系になる。  
```sh
cd src
python local_test.py
```


# requirements.txtについて
`mmcv`と`mmdet3d`はwhlでインストールしている。
## mmcv
`mmcv._ext`が必要なため予めビルドしたwhlを使用する  
whlの作成方法
```
cd code
git clone https://github.com/open-mmlab/mmcv.git -b v1.1.5
cd mmcv
MMCV_WITH_OPS=1 python setup.py bdist_wheel
```

## mmdet3d
Vitis-AIの`mmdet3d`は`v0.6.1`から派生してつくられたカスタム版なのでwhlを使用する  
さらにVitis-AI版の`mmdet3d`の`requirements.txt`だとSIGNATEのdocker環境でうまく動かないので`requirements.txt`を勝手に変えている（色々問題がありそうだがとりあえず動くので無視している）

whlの作成方法
```
cd code/mmdetection3d
python setup.py bdist_wheel
```


