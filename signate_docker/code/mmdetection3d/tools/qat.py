# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and 
# limitations under the License.


from __future__ import division

import argparse
import copy
import logging
import mmcv
import os
import sys
import time
import torch
from mmcv import Config, DictAction
from mmcv.runner import init_dist
from mmcv.runner import load_checkpoint
from os import path as osp

from mmdet3d import __version__
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed, train_detector

from pytorch_nndct import QatProcessor


def get_eval_model(cfg_path, ckpt_path, device):

    cfg = Config.fromfile(cfg_path)

    cfg.model.pretrained = None
    # cfg.data.test.test_mode = True
    cfg.data.val.test_mode = True

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = None
    if ckpt_path:
        checkpoint = load_checkpoint(model, ckpt_path, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
    model =  model.to(device)

    # scale input channels & weights to improve quantization accuracy
    if hasattr(cfg, 'quant_opt'):
        input_scale = cfg.quant_opt.input_scale
        scaled_weight_key = cfg.quant_opt.scaled_weight_key
        input_scale = torch.tensor(input_scale).reshape(1, len(input_scale), 1, 1).to(device)
        weight_scale = 1. / input_scale
        state_dict = model.state_dict()
        state_dict[scaled_weight_key] *= weight_scale
        model.load_state_dict(state_dict)
        print('Scale {} in in pretrained weights for QAT'.format(scaled_weight_key))
    else:
        input_scale = None

    pts_voxel_layer = cfg.model.pts_voxel_layer
    pts_voxel_encoder = cfg.model.pts_voxel_encoder
    max_num_points = pts_voxel_layer.max_num_points
    max_voxels = pts_voxel_layer.max_voxels[1]
    in_channels = pts_voxel_encoder.in_channels
    features = torch.randn([1, in_channels, max_voxels, max_num_points]).to(device)
    voxels = torch.randn([max_voxels, max_num_points, in_channels]).to(device)
    coors = torch.randn([max_voxels, 4]).to(device)
    dummy_inputs = (features, voxels, coors)

    return model, dummy_inputs


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('quant_config', help='quant config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--calib-dir', default='', help='the dir contains quant_info from calibration')
    parser.add_argument('--convert_qat_model', action='store_true', help='whether to convert model after QAT')
    parser.add_argument('--convert_dir', help='the dir to save converted results')
    parser.add_argument('--qat_trained_model', help='the model after QAT to be converted')
    parser.add_argument(
        '--load-from', default='', help='the checkpoint file to load from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # add a logging filter
    logging_filter = logging.Filter('mmdet')
    logging_filter.filter = lambda record: record.find('mmdet') != -1

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    if args.load_from:
        checkpoint = load_checkpoint(model, args.load_from, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']

    eval_model, dummpy_inputs = get_eval_model(args.quant_config, None, device=torch.device('cuda'))

    # logger.info(f'Model:\n{model}')
    # logger.info(f'EvalModel:\n{eval_model}')

    qat_processor = QatProcessor(eval_model,
	                             inputs=dummpy_inputs,
	                             bitwidth=8,
	                             mix_bit=False)
    qat_processor._model = model
    model = qat_processor.trainable_model(calib_dir=args.calib_dir)

    logger.info(f'QAT Model:\n{model}')

    if args.convert_qat_model:
        # Load trained weights.
        qat_checkpoint = torch.load(args.qat_trained_model)
        state_dict = qat_checkpoint['state_dict']
        model.load_state_dict(state_dict)
        # Get deployable model.
        convert_dir = args.convert_dir
        if not os.path.exists(convert_dir):
            os.makedirs(convert_dir)
        convert_path = os.path.join(convert_dir, 'qat_converted.pth')
        qat_processor._model = eval_model
        deployable_net = qat_processor.convert_to_deployable(model, output_dir=convert_dir)
        qat_checkpoint['state_dict'] = deployable_net.state_dict()
        torch.save(qat_checkpoint, convert_path)
        print('The converted QAT result is saved to {}'.format(convert_dir))
        sys.exit()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
