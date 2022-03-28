import argparse
import os
import shutil
import warnings
import numpy as np
from PIL import Image

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

configs = [
    "../work_dirs/remote/swb384_22k_1x_16bs_all/remote_swb.py",
    "../work_dirs/remote/dl3pr101_1x_16bs_all/remote_dl3pr101.py"
]
ckpts = [
    "../work_dirs/remote/swb384_22k_1x_16bs_all/latest.pth",
    "../work_dirs/remote/dl3pr101_1x_16bs_all/latest.pth"
]

cfg = mmcv.Config.fromfile(configs[0])
torch.backends.cudnn.benchmark = True
cfg.model.pretrained = None
cfg.data.test.test_mode = True
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=64,
    workers_per_gpu=4, #cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False)

models = []
for config, ckpt in zip(configs, ckpts):
    cfg = mmcv.Config.fromfile(config)
    torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, ckpt, map_location='cpu')
    model.CLASSES = dataset.CLASSES
    model.PALETTE = dataset.PALETTE
    torch.cuda.empty_cache()
    eval_kwargs = {"imgfile_prefix": "../work_imgs"}
    tmpdir = eval_kwargs['imgfile_prefix']
    mmcv.mkdir_or_exist(tmpdir)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    models.append(model)

results = []
dataset = data_loader.dataset
prog_bar = mmcv.ProgressBar(len(dataset))
loader_indices = data_loader.batch_sampler

for batch_indices, data in zip(loader_indices, data_loader):
    result = []
    for model in models:
        with torch.no_grad():
            result.append(model(return_loss=False, **data))
    result = [np.stack(_, 0).sum(0).argmax(0) for _ in zip(*result)]
    for res, batch_index in zip(result, batch_indices):
        img_info = dataset.img_infos[batch_index]
        file_name = os.path.join(tmpdir, img_info['ann']['seg_map'])
        Image.fromarray(res.astype(np.uint8)).save(file_name)
        prog_bar.update()