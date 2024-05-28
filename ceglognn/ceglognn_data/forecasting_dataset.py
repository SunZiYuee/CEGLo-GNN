import os
import torch
from torch.utils.data import Dataset
from basicts.utils import load_pkl

class ForecastingDataset(Dataset):

    def __init__(self, data_file_path: str, index_file_path: str, mode: str, seq_len: int) -> None:
        super().__init__()
        
        # Ensure mode is one of the allowed values
        assert mode in ["train", "valid", "test"], "error mode"
        # Load and process the data
        self._check_if_file_exists(data_file_path, index_file_path)
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        self.data = torch.from_numpy(processed_data).float()
        
        # Load the index for the specified mode
        self.index = load_pkl(index_file_path)[mode]    
        # Set the sequence length
        self.seq_len = seq_len
        
        # Initialize a mask tensor for long history data
        self.mask = torch.zeros(self.seq_len, self.data.shape[1], self.data.shape[2])

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str):
        # Raise an error if the data and index file does not exist
        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("BasicTS can not find data file {0}".format(data_file_path))
        
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("BasicTS can not find index file {0}".format(index_file_path))

    def __getitem__(self, index: int) -> tuple:
        # Retrieve the index range for the current sample
        idx = list(self.index[index])

        # Extract history and future data based on the index range
        history_data = self.data[idx[0]:idx[1]]     
        future_data = self.data[idx[1]:idx[2]]      
        
        # Extract long history data or use the mask if the range is out of bounds
        if idx[1] - self.seq_len < 0:
            long_history_data = self.mask
        else:
            long_history_data = self.data[idx[1] - self.seq_len:idx[1]]     

        return future_data, history_data, long_history_data

    def __len__(self):
        # Return the total number of samples
        return len(self.index)