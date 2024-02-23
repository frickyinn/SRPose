import argparse
from torch.utils.data import DataLoader
import lightning as L
from lightglue import SuperPoint

from datasets import dataset_dict
from model import PL_RelPose
from configs.default import get_cfg_defaults


def main(args):
    config = get_cfg_defaults()
    config.merge_from_file(args.config)

    task = config.DATASET.TASK
    dataset = config.DATASET.DATA_SOURCE

    batch_size = config.TRAINER.BATCH_SIZE
    # batch_size = 32
    num_workers = config.TRAINER.NUM_WORKERS
    pin_memory = config.TRAINER.PIN_MEMORY
    
    test_num_keypoints = config.MODEL.TEST_NUM_KEYPOINTS

    # seed = config.RANDOM_SEED
    # seed_torch(seed)
    
    build_fn = dataset_dict[task][dataset]
    testset = build_fn('test', config)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    pl_relpose = PL_RelPose.load_from_checkpoint(args.ckpt_path)
    pl_relpose.extractor = SuperPoint(max_num_keypoints=test_num_keypoints, detection_threshold=0.0).eval()

    trainer = L.Trainer(
        devices=[0], 
        # accelerator='gpu', strategy='ddp_find_unused_parameters_true', 
        # precision="bf16-mixed",
    )
    
    trainer.test(pl_relpose, dataloaders=testloader)


def get_parser():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--task', type=str, help='scene | object', required=True)
    # parser.add_argument('--dataset', type=str, help='matterport | megadepth | scannet | bop', required=True)
    parser.add_argument('config', type=str, help='.yaml configure file path')
    parser.add_argument('--ckpt_path', type=str, required=True)

    # parser.add_argument('--world_size', type=int, default=2)
    # parser.add_argument('--device', type=str, default='cuda:0')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
