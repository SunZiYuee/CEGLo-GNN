import os
import sys
# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from .ceglognn_arch import CEGLo
from .ceglognn_runner import CEGLoRunner
from .ceglognn_loss import ceglognn_loss
from .ceglognn_data import ForecastingDataset

CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "CEGLo(DYG_doz) demo"
CFG.RUNNER = CEGLoRunner
CFG.DATASET_CLS = ForecastingDataset
CFG.DATASET_NAME = "DYG_doz"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.DATASET_ARGS = {
    "seq_len": 2016
    }
CFG.GPU_NUM = 2

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "CEGLo"
CFG.MODEL.ARCH = CEGLo

CFG.MODEL.PARAM = {
    "pretrained_encoder_path": "encoder_ckpt/Encoder_DYG_doz.pt",
    "encoder_args": {
                    "patch_size":12,
                    "in_channel":1,
                    "embed_dim":96,
                    "num_heads":4,
                    "mlp_ratio":4,
                    "dropout":0.1,
                    "num_token":168,
                    "mask_ratio":0.8,
                    "encoder_depth":4,
                    "decoder_depth":1,
                    "mode":"forecasting"
                    },
    "gw_args":      {
                    "num_nodes" : 14,
                    "support_len" : 3,
                    "dropout"   : 0.3,
                    "gcn_bool"  : True,
                    "addaptadj" : True,
                    "aptinit"   : None,
                    "in_dim"    : 2,
                    "out_dim"   : 12,
                    "residual_channels" : 32,
                    "dilation_channels" : 32,
                    "skip_channels"     : 256,
                    "end_channels"      : 512,
                    "kernel_size"       : 2,
                    "blocks"            : 4,
                    "layers"            : 2
                    },
    "gragen_args":  {
                "k": 3

                    }
}
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True
CFG.MODEL.LOCAL_K = 14
CFG.MODEL.GLOBAL_K = 6
# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = ceglognn_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.005,
    "weight_decay":1.0e-5,
    "eps":1.0e-8,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "milestones":[1, 18, 36, 54, 72],
    "gamma":0.5
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 3.0
}
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = True
# curriculum learning
CFG.TRAIN.CL = EasyDict()
CFG.TRAIN.CL.WARM_EPOCHS = 0
CFG.TRAIN.CL.CL_EPOCHS = 6
CFG.TRAIN.CL.PREDICTION_LENGTH = 12

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 32
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = True

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# evluation
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 32
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = True
