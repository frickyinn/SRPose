import argparse
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor

from datasets import dataset_dict
from pl_trainer import PL_LightPose
from utils import seed_torch
from configs.default import get_cfg_defaults


def main(args):
    config = get_cfg_defaults()
    config.merge_from_file(args.config)

    seed_torch(config.RANDOM_SEED)
    batch_size = config.TRAINER.BATCH_SIZE
    num_workers = config.TRAINER.NUM_WORKERS
    pin_memory = config.TRAINER.PIN_MEMORY
    lr = config.TRAINER.LEARNING_RATE
    epochs = config.TRAINER.EPOCHS
    num_keypoints = config.MODEL.NUM_KEYPOINTS
        
    build_fn = dataset_dict[args.task][args.dataset]
    trainset = build_fn('train', config)
    validset = build_fn('val', config)

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    validloader = DataLoader(validset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    pl_lightpose = PL_LightPose(
        task=args.task,
        lr=lr,
        num_keypoints=num_keypoints,
        steps_per_epoch=len(trainloader),
        epochs=epochs,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = L.Trainer(devices=[0, 1], accelerator='gpu', strategy='ddp_find_unused_parameters_true', max_epochs=epochs, callbacks=[lr_monitor])
    trainer.fit(pl_lightpose, trainloader, validloader)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, help='scene | object', required=True)
    parser.add_argument('--dataset', type=str, help='matterport | megadepth | scannet | bop', required=True)
    parser.add_argument('--config', type=str, help='.yaml configure file path', required=True)
    parser.add_argument('--resume', type=str, default=None)

    # parser.add_argument('--world_size', type=int, default=2)
    # parser.add_argument('--device', type=str, default='cuda:0')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
