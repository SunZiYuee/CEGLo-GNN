import torch
import numpy as np
from easytorch.utils.dist import master_only
from basicts.runners import BaseTimeSeriesForecastingRunner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # Get forward and target features from the model configuration
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)
        self.mini = torch.from_numpy(np.load("result_DYG_doz/mini_DYG.npy")).requires_grad_(False).to(device)
        self.mean = torch.from_numpy(np.load("result_DYG_doz/mean_DYG.npy")).requires_grad_(False).to(device)
        self.std = torch.from_numpy(np.load("result_DYG_doz/std_DYG.npy")).requires_grad_(False).to(device)
    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        # Select specified forward features from the data
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        # Select specified target features from the data
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        future_data, history_data = data
        history_data = self.to_running_device(history_data)      
        future_data = self.to_running_device(future_data) 

        # Select input features for history data
        history_data = self.select_input_features(history_data)

        # Feed data into the model for reconstruction and label masked tokens
        reconstruction_masked_tokens, label_masked_tokens = self.model(
            history_data=history_data, 
            future_data=None, 
            batch_seen=iter_num, 
            epoch=epoch
        )
        
        return reconstruction_masked_tokens, label_masked_tokens

    @torch.no_grad()
    @master_only
    def test(self):

        for _, data in enumerate(self.test_data_loader):

            forward_return = self.forward(data=data, epoch=None, iter_num=None, train=False)
            
            # Rescale the predictions and real values
            prediction_rescaled = forward_return[0] - self.mini
            prediction_rescaled = prediction_rescaled * self.std[None, None, :] + self.mean[None, None, :]
            real_value_rescaled = forward_return[1] - self.mini
            real_value_rescaled = real_value_rescaled * self.std[None, None, :] + self.mean[None, None, :]
            
            # Compute and update metrics
            for metric_name, metric_func in self.metrics.items():
                metric_item = metric_func(prediction_rescaled, real_value_rescaled, null_val=self.null_val)
                self.update_epoch_meter("test_" + metric_name, metric_item.item())