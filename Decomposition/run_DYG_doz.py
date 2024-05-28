import sys
import argparse
import numpy as np


sys.path.append('./')
from GLDTR.GLDTR import *
from GLDTR.TCN import *

class Logger(object):
    def __init__(self, filename="run.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger("Decomposition/run_DYG.txt")

# Load the dataset
MTS = np.load("Decomposition/datasets/data_demo.npy")
print(MTS.shape)

# Define hyperparameters and other configurations
ver_batch_size = 7  # Vertical batch size
hor_batch_size = 256  # Horizontal batch size
TCN_channels = [16, 16, 32, 32, 16, 1]  # Number of channels for TCN
kernel_size = 7  # Kernel size for TCN
dropout = 0.1  # Dropout rate during training
rank = 6 
lr = 0.0005  # Learning rate
val_len = 1000  # Validation length
end_index = MTS.shape[1] - 1
init_epochs = 100  # Max number of iterations for initializing factors
alt_iters = 20  # Number of alternating iterations
normalize = True

def main(args):
    """
    Main function to initialize and train the GLDTR model.
    """
    # Initialize the GLD-TR model
    DG = GLDTR(
        MTS,
        ver_batch_size=ver_batch_size,
        hor_batch_size=hor_batch_size,
        TCN_channels=TCN_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        rank=rank,
        lr=lr,
        val_len=val_len,
        end_index=end_index,
        normalize=normalize,
    )

    # Train GLD-TR and save the factors
    X, F = DG.train_GLDTR(init_epochs=init_epochs, alt_iters=alt_iters)
    np.save('Decomposition/FXL_DYG_doz/X_doz', X)
    np.save('Decomposition/FXL_DYG_doz/F_doz', F)
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Call the main function
    main(args)