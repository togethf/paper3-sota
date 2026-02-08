# Copyright (c) OpenMMLab. All rights reserved.
"""
Training script for pest-pvt using MMDetection 3.x
Usage: python train_mmdet3.py configs/ricepests_pvt.py
"""
import argparse
import os
import os.path as osp

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume', action='store_true', help='resume from the latest checkpoint')
    parser.add_argument('--amp', action='store_true', help='enable automatic mixed precision training')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    # Work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # Enable automatic mixed precision training
    if args.amp:
        cfg.optim_wrapper = dict(
            type='AmpOptimWrapper',
            loss_scale='dynamic',
            _delete_=True,
            optimizer=cfg.optim_wrapper.get('optimizer', dict(type='AdamW', lr=0.0001)))

    # Resume training
    if args.resume:
        cfg.resume = True

    # Build the runner
    runner = Runner.from_cfg(cfg)

    # Start training
    runner.train()


if __name__ == '__main__':
    main()
