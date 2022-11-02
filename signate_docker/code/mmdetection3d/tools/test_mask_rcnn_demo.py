import argparse
import mmcv
import os
import torch
import numpy as np
from PIL import Image
from os import path as osp
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.core import wrap_fp16_model
from tools.fuse_conv_bn import fuse_module


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
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


def single_gpu_demo(model, data_loader, show=False, out_dir=None):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """

    Name2Color = {
        'barrier': (112, 128, 144),  # Slategrey
        'bicycle': (220, 20, 60),  # Crimson
        'bus': (255, 69, 0),  # Orangered
        'car': (255, 158, 0),  # Orange
        'construction_vehicle': (233, 150, 70),  # Darksalmon
        'motorcycle': (255, 61, 99),  # Red
        'pedestrian': (0, 0, 230),  # Blue
        'traffic_cone': (47, 79, 79),  # Darkslategrey
        'trailer': (255, 140, 0),  # Darkorange
        'truck': (255, 99, 71),  # Tomato
    }

    if isinstance(model, MMDataParallel):
        ClassTable = model.module.CLASSES
    else:
        ClassTable = model.CLASSES
    
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    if out_dir:
        mmcv.mkdir_or_exist(out_dir)

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        semseg_mask = np.ones((900, 1600)).astype('uint8') * 255
        filepath = data['img_metas'][0].data[0][0]['filename']
        # render_img = np.array(Image.open(filepath).convert('RGB'))

        cls_masks = result[0][1]
        for i, masks in enumerate(cls_masks):
            for mask in masks:
                semseg_mask[mask] = i
                # render_img[mask]  = render_img[mask]//2 + np.array(Name2Color[ClassTable[i]])//2

            # mask_file = osp.join(out_dir, f"mask_{i}.png")
            # semseg_mask = Image.fromarray(semseg_mask)
            # semseg_mask.save(mask_file)

        filename_stem = osp.split(filepath)[-1].split('.')[0]
        mask_file = osp.join(out_dir, f"{filename_stem}_mask.png")
        semseg_mask = Image.fromarray(semseg_mask)
        semseg_mask.save(mask_file)
        
        # render_file = osp.join(out_dir, f"{filename_stem}_render.png")
        # render_img = Image.fromarray(render_img)
        # render_img.save(render_file)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results) with the argument "--out", "--eval", "--format_only" '
         'or "--show"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

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
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    dataset = build_dataset(cfg.data.test)
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

    # if not distributed:
    model = MMDataParallel(model, device_ids=[0])
    # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    outputs = single_gpu_demo(model, data_loader, args.show, args.show_dir)
    '''
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)
    '''
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.options is None else args.options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        # if args.eval:
        #     dataset.evaluate(outputs, args.eval, **kwargs)


if __name__ == '__main__':
    main()
