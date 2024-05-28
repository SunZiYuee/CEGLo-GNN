# <div align="center"> Causality Enhanced Global-Local Graph Neural Network for Bioprocess Factor Forecasting </div>
## Requirements
This work is based on [BasicTS](https://github.com/zezhishao/BasicTS) with `torch==1.10.0+cu111` and `easy-torch==1.2.10`. Other dependencies can be seen in `requirements.txt`.
## Train CEGLo-GNN
1. Run `Decomposition/run_DYG_doz.py` to perform global-local decomposition. The result is saved in `Decomposition/FXL_DYG_doz`.
2. Run `ceglognn/Encoder_DYG_doz.py` to perform embedding. Move the best checkpoints to `encoder_ckpt`
4. Run `ceglognn/CEGLo_DYG_doz.py` to perform graph generation and prediction.
