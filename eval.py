import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.matterport3d import Matterport3D
from model import LightPose
from lightglue import SuperPoint
from utils import rot_angle_error


def train(args, model, testset):
    device = args.device

    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    extractor = SuperPoint(max_num_keypoints=1024, detection_threshold=0.0).eval().to(device)  # load the extractor
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6*len(trainloader), gamma=0.9)
    # rotation_criterion = rot_angle_error
    criterion = torch.nn.HuberLoss()
    
    # if args.resume is not None:
    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])

    model = model.to(device)

    with tqdm(desc=f'Test Epoch {epoch}', total=len(testloader)) as t:
        test_loss_r = 0
        test_loss_t = 0
        test_batch = 0
        test_degrees = []
        test_meters = []

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(testloader):
                images = batch['images']
                rotation = batch['rotation'].to(device)
                # rotation = rotation_matrix_from_quaternion(rotation)
                translation = batch['translation'].to(device)

                image0 = images[:, 0, ...]
                image1 = images[:, 1, ...]

                feats0 = extractor({'image': image0.to(device)})
                feats1 = extractor({'image': image1.to(device)})

                image_size = images.shape[-2:][::-1]
                pred_r, pred_t = model({'image0': {**feats0, 'image_size': image_size}, 'image1': {**feats1, 'image_size': image_size}})

                err_r = rot_angle_error(pred_r, rotation)
                loss_r = criterion(err_r, torch.zeros_like(err_r))
                
                loss_t = criterion(pred_t, translation)
                loss_t += criterion(pred_t / pred_t.norm(2, dim=1, keepdim=True), translation / translation.norm(2, dim=1, keepdim=True))

                # loss = loss_r + loss_t

                degrees = err_r.detach()
                meters = (pred_t.detach() - translation).norm(2, dim=1)

                batch = images.size(0)
                test_loss_r += loss_r.detach() * batch
                test_loss_t += loss_t.detach() * batch
                test_degrees.append(degrees)
                test_meters.append(meters)
                test_batch += batch

                td = torch.cat(test_degrees) * 180 / torch.pi
                tm = torch.cat(test_meters)
                t.set_postfix({
                    'Test R Loss': f'{test_loss_r / test_batch:.4f}',
                    'Test T Loss': f'{test_loss_t / test_batch:.4f}',
                    'Test D Med. Avg. 30d': f'{td.median():.2f}, {td.mean():.2f}, {((td < 30).sum() / len(td) * 100):.1f}',
                    'Test M Med. Avg. 1m': f'{tm.median():.2f}, {tm.mean():.2f}, {((tm < 1).sum() / len(tm) * 100):.1f}',
                })
                t.update(1)
                # break


def main(args):
    testset = Matterport3D(args.data_root, 'test')

    model = LightPose(features='superpoint', pct_pruning=args.pct_pruning)
    train(args, model, testset)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', type=str, default='r6d')
    parser.add_argument('--data_root', type=str, default='/mnt/ssd/yinrui/mp3d')

    parser.add_argument('--resume', type=str, default='../LightPose2/checkpoints/match_aug_20231209_0349/match_aug_model_best.pth')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--pct_pruning', type=float, default=0.)
    
    parser.add_argument('--device', type=str, default='cuda:1')
    # parser.add_argument('--use_amp', action='store_true')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
