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


import mmcv
import torch

def single_gpu_quant_test(cfg, model, float_model, data_loader, device, show=False, 
                          out_dir=None, input_scale=None, dump_xmodel=False):
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
    float_model.to(device)
    model.to(device)
    if input_scale is not None:
        input_scale.to(device)

    float_model.eval()
    model.eval()

    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            features, voxels, coors = float_model.preprocess_api(**data)
            features.to(device)
            voxels.to(device)
            coors.to(device)
            
            if input_scale is not None:
                features *= input_scale

            pts_voxel_layer = cfg.model.pts_voxel_layer
            pts_voxel_encoder = cfg.model.pts_voxel_encoder
            max_num_points = pts_voxel_layer.max_num_points
            max_voxels = pts_voxel_layer.max_voxels[1]
            in_channels = pts_voxel_encoder.in_channels

            ext_features = torch.zeros(1, in_channels, max_voxels, max_num_points, device=device)
            ext_voxels = torch.zeros(max_voxels, max_num_points, in_channels, device=device)
            ext_coors = -1 * torch.ones(max_voxels, 4, device=device) # exit_coors[:,0] is used as batch_id in scatter
            voxel_num = features.shape[2]
            ext_features[:, :, :voxel_num, :] = features
            ext_voxels[:voxel_num, :, :] = voxels
            ext_coors[:voxel_num, :] = coors
            outs = model(ext_features, ext_voxels, ext_coors)

            if dump_xmodel:
                return results

            img_metas = data['img_metas'][0].data[0]
            result = float_model.postprocess_api(outs, img_metas, rescale=True)

        if show:
            float_model.show_results(data, result, out_dir)

        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
