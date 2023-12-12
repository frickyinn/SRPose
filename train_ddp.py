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

from datasets import Matterport3D
from model import LightPose
from augmentation import RGBDAugmentor
from lightglue import SuperPoint
from utils import seed_torch, rot_angle_error


def setup(rank, master_port, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, args, model, trainset, validset):
    print(f"Running DDP on rank {rank}.")
    setup(rank, args.master_port, args.world_size)

    epochs = args.epochs

    trainsampler = DistributedSampler(trainset)
    validsampler = DistributedSampler(validset, shuffle=False)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=trainsampler)
    validloader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=validsampler)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    augment = RGBDAugmentor()
    extractor = SuperPoint(max_num_keypoints=1024, detection_threshold=0.0).eval().to(rank)  # load the extractor
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(trainloader), epochs=epochs)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6*len(trainloader), gamma=0.9)
    # rotation_criterion = rot_angle_error
    criterion = torch.nn.HuberLoss()

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join(args.save_path, f'{args.task_name}_{run_id}')
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    train_writer = SummaryWriter(f'./log/{args.task_name}_{run_id}_train')
    valid_writer = SummaryWriter(f'./log/{args.task_name}_{run_id}_valid')
    
    start_epoch = 0
    # if args.resume is not None:
    #     if os.path.isfile(args.resume):
    #         checkpoint = torch.load(args.resume)
    #         start_epoch = checkpoint["epoch"] + 1
    #         model.load_state_dict(checkpoint["model"])
    #         optimizer.load_state_dict(checkpoint["optimizer"])
    #         for state in optimizer.state.values():
    #             for k, v in state.items():
    #                 if isinstance(v, torch.Tensor):
    #                     state[k] = v.to(args.rank)

    min_derr = 180
    for e in range(start_epoch, epochs):
        with tqdm(desc=f'Train Epoch {e+1}', total=len(trainloader)) as t:
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

                image_size = images.shape[-2:][::-1]
                image0 = images[:, 0, ...]
                image1 = images[:, 1, ...]
                image0 = augment(image0)
                image1 = augment(image1)

                with torch.no_grad():
                    feats0 = extractor({'image': image0.to(rank)})
                    feats1 = extractor({'image': image1.to(rank)})
                
                pred_r, pred_t = model({'image0': {**feats0, 'image_size': image_size}, 'image1': {**feats1, 'image_size': image_size}})
                
                err_r = rot_angle_error(pred_r, rotation)
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
            if rank == 0:
                train_writer.add_scalar('Rotation Loss', train_loss_r / train_batch, e+1)
                train_writer.add_scalar('Translation Loss', train_loss_t / train_batch, e+1)
                train_writer.add_scalar('Degree Error Med.', td.median(), e+1)
                train_writer.add_scalar('Degree Error Avg.', td.mean(), e+1)
                train_writer.add_scalar('Meter Error Med.', tm.median(), e+1)
                train_writer.add_scalar('Meter Error Avg.', tm.mean(), e+1)
                train_writer.add_scalar('Learning Rate', scheduler.get_last_lr()[-1], e+1)

        with tqdm(desc=f'Valid Epoch {e+1}', total=len(validloader)) as t:
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

                    image0 = images[:, 0, ...]
                    image1 = images[:, 1, ...]

                    feats0 = extractor({'image': image0.to(rank)})
                    feats1 = extractor({'image': image1.to(rank)})

                    image_size = images.shape[-2:][::-1]
                    pred_r, pred_t = model({'image0': {**feats0, 'image_size': image_size}, 'image1': {**feats1, 'image_size': image_size}})

                    err_r = rot_angle_error(pred_r, rotation)
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
                if rank == 0:
                    valid_writer.add_scalar('Rotation Loss', valid_loss_r / valid_batch, e+1)
                    valid_writer.add_scalar('Translation Loss', valid_loss_t / valid_batch, e+1)
                    valid_writer.add_scalar('Degree Error Med.', vd.median(), e+1)
                    valid_writer.add_scalar('Degree Error Avg.', vd.mean(), e+1)
                    valid_writer.add_scalar('Meter Error Med.', vm.median(), e+1)
                    valid_writer.add_scalar('Meter Error Avg.', vm.mean(), e+1)

        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": e,
                "last_lr": scheduler.get_last_lr()[-1],
            }
            
            torch.save(checkpoint, os.path.join(save_path, f"{args.task_name}_model_latest.pth"))
            if vd.mean() < min_derr:
                min_derr = vd.mean()
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": e,
                    "last_lr": scheduler.get_last_lr()[-1],
                    "degree error": min_derr
                }
                torch.save(checkpoint, os.path.join(save_path, f"{args.task_name}_model_best.pth"))

    cleanup()


def main(args):
    seed_torch(3407)

    trainset = Matterport3D(args.data_root, 'train')
    validset = Matterport3D(args.data_root, 'val')

    model = LightPose(features='superpoint')

    mp.spawn(
        train,
        args=(args, model, trainset, validset),
        nprocs=args.world_size,
        join=True,
    )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', type=str, default='r6d')
    parser.add_argument('--data_root', type=str, default='/mnt/ssd/yinrui/mp3d')

    parser.add_argument('--save_path', type=str, default='./checkpoints')
    # parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=2e-4)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    
    parser.add_argument('--master_port', type=int, default=12355)
    parser.add_argument('--world_size', type=int, default=2)
    # parser.add_argument('--use_amp', action='store_true')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
