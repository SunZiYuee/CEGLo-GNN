import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + "/../.."))
import torch
from basicts import launch_training

torch.set_num_threads(2) # aviod high cpu avg usage#


def parse_args():
    parser = ArgumentParser(description="CEGLo-GNN")
    # parser.add_argument("-c", "--cfg", default="ceglognn/Encoder_DYG_doz.py", help="training config")
    parser.add_argument("-c", "--cfg", default="ceglognn/CEGLo_DYG_doz.py", help="training config")

    parser.add_argument("--gpus", default="0,1", help="visible gpus")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    launch_training(args.cfg, args.gpus)
