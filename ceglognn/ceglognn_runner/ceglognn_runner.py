import torch
import numpy as np
from basicts.runners import BaseTimeSeriesForecastingRunner
from basicts.metrics import masked_mae, masked_rmse, masked_mape

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CEGLoRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # Initialize metrics from configuration or use default metrics
        self.metrics = cfg.get("METRICS", {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape})
        # Get forward and target features from the model configuration
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)
        self.scores = torch.from_numpy(np.load('result_DYG_doz/F_DYG.npy')).requires_grad_(False)
        self.global_k = cfg["MODEL"].get("GLOBAL_K", None)
        self.local_k = cfg["MODEL"].get("LOCAL_K", None)
    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        # Select specified forward features from the data
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        # Select specified target features from the data
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data, epoch, iter_num, train, **kwargs) -> tuple:
        # Load and prepare the precomputed score tensor

        scores = self.scores.to(device)
        
        future_data_raw, history_data_raw, long_history = data
        
        history_data_raw = self.to_running_device(history_data_raw)   
        long_history = self.to_running_device(long_history) 
        future_data_raw = self.to_running_device(future_data_raw) 

        # Apply feature tensor transformation to future and history data
        future_data = torch.einsum('ik, abkc -> abic', scores, future_data_raw[:, :, self.local_k:(self.local_k+self.global_k), :]) + future_data_raw[:, :, 0:self.local_k, :]
        history = torch.einsum('ik, abkc -> abic', scores, history_data_raw[:, :, self.local_k:(self.local_k+self.global_k), :]) + history_data_raw[:, :, 0:self.local_k, :] 

        # Select input features for history and long history data
        history = self.select_input_features(history)
        long_history = self.select_input_features(long_history)

        # Feed data into the model for prediction
        prediction, pred_adj, prior_adj, lambda_graph = self.model(
            history=history, 
            long_history=long_history, 
            future=None, 
            batch_seen=iter_num, 
            epoch=epoch
        )

        # Ensure the prediction shape matches the expected output shape
        batch_size, length, num_nodes, _ = future_data.shape
        assert list(prediction.shape)[:3] == [batch_size, length, num_nodes], "error shape of the output"

        # Select target features from the prediction and real future data
        prediction = self.select_target_features(prediction)
        real_value = self.select_target_features(future_data)
        
        return prediction, real_value, pred_adj, prior_adj, lambda_graph