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


import argparse
import mmcv
import os
import sys
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet3d.apis.quant_test import single_gpu_quant_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.core import wrap_fp16_model
from tools.fuse_conv_bn import fuse_module

import random
from pytorch_nndct.apis import torch_quantizer

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--quant_mode', default='float', choices=['float', 'calib', 'test'], help='quantization mode. float: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
    parser.add_argument('--calib_len', type=int, default=500, help='set sample num for calibration')
    parser.add_argument('--calib_sample_method', type=str, default='random')
    parser.add_argument('--quant_dir', default='q_results')
    parser.add_argument('--bitwidth', type=int, default=8)
    parser.add_argument('--dump_xmodel', action='store_true', default=False)
    parser.add_argument('--fast_finetune', action='store_true', default=False)
    parser.add_argument('--fft_len', type=int, default=300)
    parser.add_argument('--fft_sample_method', type=str, default='random')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
           or args.quant_mode=='calib' or args.dump_xmodel, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results) with the argument "--out", "--eval", "--format_only" '
         'or "--show"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    print(cfg)
    raise RuntimeError()
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # cfg.data.test.test_mode = True
    cfg.data.val.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    # samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    # dataset = build_dataset(cfg.data.test)
    # samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
    samples_per_gpu = 1
    dataset = build_dataset(cfg.data.val, dict(test_mode=True))

    if args.quant_mode == 'calib' and 0 < args.calib_len < len(dataset):
        if args.calib_sample_method == 'random':
            dataset = torch.utils.data.Subset(dataset, random.sample(range(0, len(dataset)), args.calib_len))
        else:
            dataset = torch.utils.data.Subset(dataset, list(range(args.calib_len)))

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_module(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # scale input channels & weights to improve quantization accuracy
    if hasattr(cfg, 'quant_opt'):
        quant_opt = cfg.quant_opt
        input_scale = torch.tensor(quant_opt.input_scale).reshape(1, len(quant_opt.input_scale), 1, 1)
        weight_scale = 1. / input_scale
        state_dict = model.state_dict()
        state_dict[quant_opt.scaled_weight_key] *= weight_scale
        model.load_state_dict(state_dict)
        print('Scaling {} for quantization...'.format(quant_opt.scaled_weight_key))
    else:
        input_scale = None

    import copy
    float_model = copy.deepcopy(model)
    quant_mode = args.quant_mode
    output_dir = args.quant_dir
    bitwidth = args.bitwidth

    device = torch.device('cuda')
    if quant_mode == 'test' and args.dump_xmodel:
        device = torch.device('cpu')

    if quant_mode != 'float':
        pts_voxel_layer = cfg.model.pts_voxel_layer
        pts_voxel_encoder = cfg.model.pts_voxel_encoder
        max_num_points = pts_voxel_layer.max_num_points
        max_voxels = pts_voxel_layer.max_voxels[1]
        in_channels = pts_voxel_encoder.in_channels
        features = torch.randn([1, in_channels, max_voxels, max_num_points])
        voxels = torch.randn([max_voxels, max_num_points, in_channels])
        coors = torch.randn([max_voxels, 4])
        quantizer = torch_quantizer(quant_mode=quant_mode,
                                    module=model,
                                    input_args=(features, voxels, coors),
                                    output_dir=output_dir,
                                    bitwidth=bitwidth,
                                    device=device)
        model = quantizer.quant_model

    if args.fast_finetune:
        if quant_mode == 'calib':
            subdataset = build_dataset(cfg.data.val, dict(test_mode=True))
            if args.fft_len <= 0 or args.fft_len > len(subdataset):
                args.fft_len = len(subdataset)
            if args.fft_sample_method == 'random':
                subdataset = torch.utils.data.Subset(subdataset, random.sample(range(0, len(subdataset)), args.fft_len))
            else:
                subdataset = torch.utils.data.Subset(subdataset, list(range(args.fft_len)))
            subdata_loader = build_dataloader(
                subdataset,
                samples_per_gpu=samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)
            quantizer.fast_finetune(single_gpu_quant_test,
                                    (cfg, model, float_model, subdata_loader, device, False, None, input_scale, False))
        elif quant_mode == 'test':
            quantizer.load_ft_param()

    outputs = single_gpu_quant_test(cfg, model, float_model, data_loader, device, args.show, args.show_dir, input_scale, args.dump_xmodel)

    if quant_mode == 'calib':
        quantizer.export_quant_config()
    elif quant_mode == 'test' and args.dump_xmodel:
        quantizer.export_xmodel(output_dir=output_dir, deploy_check=True)
        sys.exit()

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.options is None else args.options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval and args.quant_mode != 'calib':
            dataset.evaluate(outputs, args.eval, **kwargs)


if __name__ == '__main__':
    main()
