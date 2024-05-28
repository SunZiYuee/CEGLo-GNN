import torch
import numpy as np

class data_loader(object):
    """
    Data Loader Class for GLDTR
    """

    def __init__(self, MTS, ver_batch_size=1, hor_batch_size=100, end_index=20000, val_len=30, shuffle=False):
        """
        Arguments:
        MTS: time-series matrix n*T
        ver_batch_size: vertical batch size
        hor_batch_size: horizontal batch size
        end_index: training and validation set is only from 0:end_index
        val_len: validation length. The last 'val_len' time-points for every time-series is the validation set
        shuffle: data is shuffled if True (this is deprecated and set to False)
        """
        n, T = MTS.shape
        self.vindex = 0
        self.hindex = 0
        self.epoch = 0
        self.ver_batch_size = ver_batch_size
        self.hor_batch_size = hor_batch_size
        self.MTS = MTS
        self.val_len = val_len
        self.end_index = end_index
        self.val_index = np.random.randint(0, n - self.ver_batch_size - 2)
        self.shuffle = shuffle
        self.I = np.array(range(n))

    def next_batch(self, option=1):
        """
        Fetches the next batch of data for training.

        Arguments:
        option = 1: data is returned as a PyTorch tensor of shape [nd, cd, td]
                    where nd is ver_batch_size, hb is hsize, and cd is the number of channels (depends on covariates)
        option = 0: deprecated

        Returns:
        inp: input batch
        out: one shifted output batch
        vindex: starting vertical index of input batch
        hindex: starting horizontal index of input batch
        """
        n, T = self.MTS.shape

        # Check if the horizontal index exceeds the end index
        if self.hindex + self.hor_batch_size + 1 >= self.end_index:
            pr_hindex = self.hindex
            self.hindex = 0

            # Check if the vertical index exceeds the number of samples
            if self.vindex + self.ver_batch_size >= n:
                pr_vindex = self.vindex
                self.vindex = 0
                self.epoch += 1

                # Shuffle the data if required (deprecated)
                if self.shuffle:
                    I = np.random.choice(n, n, replace=False)
                    self.I = I
                    self.MTS = self.MTS[self.I, :]
            else:
                pr_vindex = self.vindex
                self.vindex += self.ver_batch_size
        else:
            pr_hindex = self.hindex
            self.hindex += self.hor_batch_size
            pr_vindex = self.vindex

        # Extract the input and output batches
        data = self.MTS[int(pr_vindex):int(pr_vindex + self.ver_batch_size),
                         int(pr_hindex):int(min(self.end_index, pr_hindex + self.hor_batch_size))]
        out_data = self.MTS[int(pr_vindex):int(pr_vindex + self.ver_batch_size),
                             int(pr_hindex + 1):int(min(self.end_index, pr_hindex + self.hor_batch_size) + 1)]

        nd, Td = data.shape

        # Convert the data to PyTorch tensors and reshape
        if option == 1:
            inp = torch.from_numpy(data).view(1, nd, Td).transpose(0, 1).float()
            out = torch.from_numpy(out_data).view(1, nd, Td).transpose(0, 1).float()
        else:
            inp = torch.from_numpy(data).float()
            out = torch.from_numpy(out_data).float()

        # Replace NaNs with zeros
        inp[torch.isnan(inp)] = 0
        out[torch.isnan(out)] = 0

        return inp, out, self.vindex, self.hindex

    def supply_test(self, option=1):
        """
        Supplies validation set in the same format as above.

        Arguments:
        option = 1: data is returned as a PyTorch tensor of shape [nd, cd, td]
                    where nd is ver_batch_size, hb is hsize, and cd is the number of channels (depends on covariates)
        option = 0: deprecated

        Returns:
        inp: input validation batch
        out: one shifted output validation batch
        vindex: starting vertical index of input batch
        hindex: starting horizontal index of input batch
        """
        index = self.val_index

        # Extract the input and output validation batches
        in_data = self.MTS[int(index):int(index + self.ver_batch_size),
                            int(self.end_index):int(self.end_index + self.val_len)]
        out_data = self.MTS[int(index):int(index + self.ver_batch_size),
                             int(self.end_index + 1):int(self.end_index + self.val_len + 1)]

        nd, Td = in_data.shape

        # Convert the data to PyTorch tensors and reshape
        if option == 1:
            inp = torch.from_numpy(in_data).view(1, nd, Td).transpose(0, 1).float()
            out = torch.from_numpy(out_data).view(1, nd, Td).transpose(0, 1).float()
        else:
            inp = torch.from_numpy(in_data).float()
            out = torch.from_numpy(out_data).float()

        return inp, out, self.vindex, self.hindex