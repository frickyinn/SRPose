import argparse
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from datasets import dataset_dict, RandomConcatSampler
from model import PL_RelPose
from utils import seed_torch
from configs.default import get_cfg_defaults


def main(args):
    config = get_cfg_defaults()
    config.merge_from_file(args.config)

    task = config.DATASET.TASK
    dataset = config.DATASET.DATA_SOURCE

    batch_size = config.TRAINER.BATCH_SIZE
    num_workers = config.TRAINER.NUM_WORKERS
    pin_memory = config.TRAINER.PIN_MEMORY
    n_samples_per_subset = config.TRAINER.N_SAMPLES_PER_SUBSET
    lr = config.TRAINER.LEARNING_RATE
    epochs = config.TRAINER.EPOCHS
    pct_start = config.TRAINER.PCT_START

    num_keypoints = config.MODEL.NUM_KEYPOINTS
    n_layers = config.MODEL.N_LAYERS
    num_heads = config.MODEL.NUM_HEADS

    seed = config.RANDOM_SEED
    seed_torch(seed)

    build_fn = dataset_dict[task][dataset]
    trainset = build_fn('train', config)
    validset = build_fn('val', config)

    if dataset == 'scannet' or dataset == 'megadepth' or dataset == 'linemod' or dataset == 'ho3d':
        sampler = RandomConcatSampler(
            trainset,
            n_samples_per_subset=n_samples_per_subset,
            subset_replacement=True,
            shuffle=True, 
            seed=seed
        )
        trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, sampler=sampler)
    else:
        trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    validloader = DataLoader(validset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    if args.weights is None:
        pl_relpose = PL_RelPose(
            task=task,
            lr=lr,
            epochs=epochs,
            pct_start=pct_start,
            n_layers=n_layers,
            num_heads=num_heads,
            num_keypoints=num_keypoints,
        )
    else:
        pl_relpose = PL_RelPose.load_from_checkpoint(
            checkpoint_path=args.weights,
            task=task,
            lr=lr,
            epochs=epochs,
            pct_start=pct_start,
            n_layers=n_layers,
            num_heads=num_heads,
            num_keypoints=num_keypoints,
        )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    latest_checkpoint_callback = ModelCheckpoint()
    best_checkpoint_callback = ModelCheckpoint(monitor='valid_auc@20', mode='max')
    trainer = L.Trainer(
        devices=[0, 1], accelerator='gpu', strategy='ddp_find_unused_parameters_true', 
        max_epochs=epochs, 
        callbacks=[lr_monitor, latest_checkpoint_callback, best_checkpoint_callback],
        precision="bf16-mixed",
        # fast_dev_run=1,
    )
    
    trainer.fit(pl_relpose, trainloader, validloader, ckpt_path=args.resume)


def get_parser():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--task', type=str, help='scene | object', choices={'scene', 'object'}, required=True)
    # parser.add_argument('--dataset', type=str, help='matterport | megadepth | scannet | bop | ho3d', required=True)
    parser.add_argument('config', type=str, help='.yaml configure file path')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)

    # parser.add_argument('--world_size', type=int, default=2)
    # parser.add_argument('--device', type=str, default='cuda:0')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
