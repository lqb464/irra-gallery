from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRRA Test")
    parser.add_argument("--config_file", default='logs/CUHK-PEDES/iira/configs.yaml')
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--hard_score_weight", type=float, default=None,
                        help="override the saved hard-candidate reranking weight")
    cli_args = parser.parse_args()
    args = load_train_configs(cli_args.config_file)

    args.training = False
    args.eval_split = cli_args.split
    if cli_args.hard_score_weight is not None:
        args.hard_score_weight = cli_args.hard_score_weight
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    if device.type == "cpu":
        model.float()
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    do_inference(model, test_img_loader, test_txt_loader)
