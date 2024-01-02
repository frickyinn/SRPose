import os
from datetime import datetime
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from lightglue import SuperPoint
from datasets import Augmentor, dataset_dict
from model import LightPose
from utils import seed_torch, rot_degree_error
from configs.default import get_cfg_defaults


def setup(rank, master_port, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, args, config, model, trainset, validset):
    world_size = args.world_size
    master_port = config.TRAINER.MASTER_PORT
    epochs = config.TRAINER.EPOCHS
    batch_size = config.TRAINER.BATCH_SIZE
    num_workers = config.TRAINER.NUM_WORKERS
    pin_memory = config.TRAINER.PIN_MEMORY
    lr = config.TRAINER.LEARNING_RATE
    save_path_ = config.TRAINER.SAVE_PATH

    num_keypoints = config.MODEL.NUM_KEYPOINTS

    if world_size > 1:
        print(f"Running DDP on rank {rank}.")
        setup(rank, master_port, world_size)

        trainsampler = DistributedSampler(trainset)
        validsampler = DistributedSampler(validset, shuffle=False)
        trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, sampler=trainsampler)
        validloader = DataLoader(validset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, sampler=validsampler)

        model = model.to(rank)
        model = DDP(model, device_ids=[rank])

    else:
        trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
        validloader = DataLoader(validset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        
        model = model.to(rank)

    augment = RGBDAugmentor()
    extractor = SuperPoint(max_num_keypoints=num_keypoints, detection_threshold=0.0).eval().to(rank)  # load the extractor
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(trainloader), epochs=epochs)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6*len(trainloader), gamma=0.9)
    # rotation_criterion = rot_angle_error
    criterion = torch.nn.HuberLoss()

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join(save_path_, f'{args.task}_{args.dataset}_{run_id}')
    os.makedirs(save_path_, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    if world_size == 1 or rank == 0:
        train_writer = SummaryWriter(f'./log/{args.task}_{args.dataset}_{run_id}_train')
        valid_writer = SummaryWriter(f'./log/{args.task}_{args.dataset}_{run_id}_valid')
    
    start_epoch = 0
    if args.resume is not None and os.path.isfile(args.resume):
        if world_size > 1:
            dist.barrier()
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(args.resume, map_location=map_location)
            model.module.load_state_dict(checkpoint["model"])
        else:
            map_location = rank
            checkpoint = torch.load(args.resume, map_location=map_location)
            model.load_state_dict(checkpoint["model"])
        
        start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(rank)

    min_derr = 180
    for e in range(start_epoch, epochs):
        with tqdm(desc=f'Train Epoch {e+1}', total=len(trainloader), disable=(world_size > 1 and rank != 0)) as t:
            train_loss_r = 0
            train_loss_t = 0
            train_batch = 0
            train_degrees = []
            train_meters = []

            model.train()
            for i, batch in enumerate(trainloader):
                optimizer.zero_grad()

                images = batch['images']
                rotation = batch['rotation'].to(rank)
                translation = batch['translation'].to(rank)
                intrinsics = batch['intrinsics'].to(rank)

                # image_size = images.shape[-2:][::-1]
                image0 = images[:, 0, ...]
                image1 = images[:, 1, ...]
                image0 = augment(image0)
                image1 = augment(image1)

                with torch.no_grad():
                    feats0 = extractor({'image': image0.to(rank)})
                    feats1 = extractor({'image': image1.to(rank)})
                
                if 'scales' in batch:
                    scales = batch['scales'].to(rank)
                    feats0['keypoints'] *= scales[:, 0].unsqueeze(1)
                    feats1['keypoints'] *= scales[:, 1].unsqueeze(1)

                if args.task == 'scene':
                    pred_r, pred_t = model({'image0': {**feats0, 'intrinsics': intrinsics[:, 0]}, 'image1': {**feats1, 'intrinsics': intrinsics[:, 1]}})
                elif args.task == 'object':
                    bboxes = batch['bboxes'].to(rank)
                    pred_r, pred_t = model({'image0': {**feats0, 'intrinsics': intrinsics[:, 0], 'bbox': bboxes[:, 0]}, 'image1': {**feats1, 'intrinsics': intrinsics[:, 1]}})
                
                err_r = rot_degree_error(pred_r, rotation)
                loss_r = criterion(err_r, torch.zeros_like(err_r))

                loss_t = criterion(pred_t, translation)
                loss_t += criterion(pred_t / pred_t.norm(2, dim=1, keepdim=True), translation / translation.norm(2, dim=1, keepdim=True))

                loss = loss_r + loss_t

                degrees = err_r.detach()
                meters = (pred_t.detach() - translation).norm(2, dim=1)
                
                loss.backward()
                optimizer.step()
                scheduler.step()

                batch = images.size(0)
                train_loss_r += loss_r.detach() * batch
                train_loss_t += loss_t.detach() * batch
                train_degrees.append(degrees)
                train_meters.append(meters)
                train_batch += batch

                td = torch.cat(train_degrees) * 180 / torch.pi
                tm = torch.cat(train_meters)
                t.set_postfix({
                    'Train R Loss': f'{train_loss_r / train_batch:.4f}',
                    'Train T Loss': f'{train_loss_t / train_batch:.4f}',
                    'Train D Med. Avg.': f'{td.median():.2f}, {td.mean():.2f}',
                    'Train M Med. Avg.': f'{tm.median():.2f}, {tm.mean():.2f}',
                })
                t.update(1)
                # break
            if world_size == 1 or rank == 0:
                train_writer.add_scalar('Rotation Loss', train_loss_r / train_batch, e+1)
                train_writer.add_scalar('Translation Loss', train_loss_t / train_batch, e+1)
                train_writer.add_scalar('Degree Error Med.', td.median(), e+1)
                train_writer.add_scalar('Degree Error Avg.', td.mean(), e+1)
                train_writer.add_scalar('Meter Error Med.', tm.median(), e+1)
                train_writer.add_scalar('Meter Error Avg.', tm.mean(), e+1)
                train_writer.add_scalar('Learning Rate', scheduler.get_last_lr()[-1], e+1)

        with tqdm(desc=f'Valid Epoch {e+1}', total=len(validloader), disable=(world_size > 1 and rank != 0)) as t:
            valid_loss_r = 0
            valid_loss_t = 0
            valid_batch = 0
            valid_degrees = []
            valid_meters = []

            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(validloader):
                    images = batch['images']
                    rotation = batch['rotation'].to(rank)
                    translation = batch['translation'].to(rank)
                    intrinsics = batch['intrinsics'].to(rank)

                    image0 = images[:, 0, ...]
                    image1 = images[:, 1, ...]

                    feats0 = extractor({'image': image0.to(rank)})
                    feats1 = extractor({'image': image1.to(rank)})

                    if 'scales' in batch:
                        scales = batch['scales'].to(rank)
                        feats0['keypoints'] *= scales[:, 0].unsqueeze(1)
                        feats1['keypoints'] *= scales[:, 1].unsqueeze(1)

                    # image_size = images.shape[-2:][::-1]
                    if args.task == 'scene':
                        pred_r, pred_t = model({'image0': {**feats0, 'intrinsics': intrinsics[:, 0]}, 'image1': {**feats1, 'intrinsics': intrinsics[:, 1]}})
                    elif args.task == 'object':
                        bboxes = batch['bboxes'].to(rank)
                        pred_r, pred_t = model({'image0': {**feats0, 'intrinsics': intrinsics[:, 0], 'bbox': bboxes[:, 0]}, 'image1': {**feats1, 'intrinsics': intrinsics[:, 1]}})

                    err_r = rot_degree_error(pred_r, rotation)
                    loss_r = criterion(err_r, torch.zeros_like(err_r))
                    
                    loss_t = criterion(pred_t, translation)
                    loss_t += criterion(pred_t / pred_t.norm(2, dim=1, keepdim=True), translation / translation.norm(2, dim=1, keepdim=True))

                    loss = loss_r + loss_t

                    degrees = err_r.detach()
                    meters = (pred_t.detach() - translation).norm(2, dim=1)

                    batch = images.size(0)
                    valid_loss_r += loss_r.detach() * batch
                    valid_loss_t += loss_t.detach() * batch
                    valid_degrees.append(degrees)
                    valid_meters.append(meters)
                    valid_batch += batch

                    vd = torch.cat(valid_degrees) * 180 / torch.pi
                    vm = torch.cat(valid_meters)
                    t.set_postfix({
                        'Valid R Loss': f'{valid_loss_r / valid_batch:.4f}',
                        'Valid T Loss': f'{valid_loss_t / valid_batch:.4f}',
                        'Valid D Med. Avg.': f'{vd.median():.2f}, {vd.mean():.2f}',
                        'Valid M Med. Avg.': f'{vm.median():.2f}, {vm.mean():.2f}',
                    })
                    t.update(1)
                    # break
                if world_size == 1 or rank == 0:
                    valid_writer.add_scalar('Rotation Loss', valid_loss_r / valid_batch, e+1)
                    valid_writer.add_scalar('Translation Loss', valid_loss_t / valid_batch, e+1)
                    valid_writer.add_scalar('Degree Error Med.', vd.median(), e+1)
                    valid_writer.add_scalar('Degree Error Avg.', vd.mean(), e+1)
                    valid_writer.add_scalar('Meter Error Med.', vm.median(), e+1)
                    valid_writer.add_scalar('Meter Error Avg.', vm.mean(), e+1)

        if world_size == 1 or rank == 0:
            module = model.module if world_size > 1 else model
            checkpoint = {
                "model": module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": e,
                "last_lr": scheduler.get_last_lr()[-1],
                "degree error": vd.mean()
            }
            
            torch.save(checkpoint, os.path.join(save_path, f"{args.task}_{args.dataset}_model_latest.pth"))
            if vd.mean() < min_derr:
                min_derr = vd.mean()
                checkpoint = {
                    "model": module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": e,
                    "last_lr": scheduler.get_last_lr()[-1],
                    "degree error": min_derr
                }
                torch.save(checkpoint, os.path.join(save_path, f"{args.task}_{args.dataset}_model_best.pth"))

    if world_size > 1:
        cleanup()


def main(args):
    config = get_cfg_defaults()
    config.merge_from_file(args.config)

    seed_torch(config.RANDOM_SEED)

    model = LightPose(features='superpoint', task=args.task)
        
    build_fn = dataset_dict[args.task][args.dataset]
    trainset = build_fn('train', config)
    validset = build_fn('val', config)

    world_size = args.world_size
    if world_size > 1:
        mp.spawn(
            train,
            args=(args, config, model, trainset, validset),
            nprocs=world_size,
            join=True,
        )
    else:
        train(args.device, args, config, model, trainset, validset)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, help='scene | object')
    parser.add_argument('--dataset', type=str, help='matterport | megadepth | scannet | bop')
    parser.add_argument('--config', type=str, help='.yaml configure file path')
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
