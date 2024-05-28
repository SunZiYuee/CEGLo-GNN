import torch
import numpy as np
from torch.nn.utils import weight_norm
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from GLDTR.data_loader import data_loader
import pickle

use_cuda = True

class Chomp1d(nn.Module):
    """
    A layer to remove the extra padding added to the input.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    A temporal block consisting of two convolutional layers, each followed by a Chomp1d, ReLU, and Dropout layer.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1, init=True):
        super(TemporalBlock, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init = init
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights of the convolutional layers.
        """
        if self.init:
            nn.init.normal_(self.conv1.weight, std=1e-3)
            nn.init.normal_(self.conv2.weight, std=1e-3)
            self.conv1.weight[:, 0, :] += 1.0 / self.kernel_size
            self.conv2.weight += 1.0 / self.kernel_size
            nn.init.normal_(self.conv1.bias, std=1e-6)
            nn.init.normal_(self.conv2.bias, std=1e-6)
        else:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalBlock_last(nn.Module):
    """
    A temporal block for the last layer, similar to TemporalBlock but without ReLU activation at the end.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, init=True):
        super(TemporalBlock_last, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.dropout1, self.conv2, self.chomp2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init = init
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights of the convolutional layers.
        """
        if self.init:
            nn.init.normal_(self.conv1.weight, std=1e-3)
            nn.init.normal_(self.conv2.weight, std=1e-3)
            self.conv1.weight[:, 0, :] += 1.0 / self.kernel_size
            self.conv2.weight += 1.0 / self.kernel_size
            nn.init.normal_(self.conv1.bias, std=1e-6)
            nn.init.normal_(self.conv2.bias, std=1e-6)
        else:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res  # No ReLU at the end

class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network consisting of multiple TemporalBlock layers.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.1, init=True):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.num_channels = num_channels
        self.num_inputs = num_inputs
        self.kernel_size = kernel_size
        self.dropout = dropout
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            if i == num_levels - 1:
                layers += [TemporalBlock_last(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size, dropout=dropout, init=init)]
            else:
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size, dropout=dropout, init=init)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(object):
    """
    Temporal Convolutional Network (TCN) for time-series forecasting.
    """
    def __init__(self, MTS, num_inputs=1, num_channels=[32, 32, 32, 32, 32, 1], kernel_size=7, dropout=0.2, ver_batch_size=300, hor_batch_size=128, num_epochs=100, lr=0.0005, val_len=10, test=True, end_index=120, normalize=False):
        """
        Arguments:
        MTS: input time-series n*T
        num_inputs: always set to 1
        num_channels: list containing channel progression of temporal convolution network
        kernel_size: kernel size of temporal convolution filters
        dropout: dropout rate for each layer
        ver_batch_size: vertical batch size
        hor_batch_size: horizontal batch size
        num_epochs: max. number of epochs
        lr: learning rate
        val_len: validation length
        test: always set to True
        end_index: no data is touched for training or validation beyond end_index
        normalize: normalize dataset before training or not
        """
        self.ver_batch_size = ver_batch_size
        self.hor_batch_size = hor_batch_size
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.num_epochs = num_epochs
        self.lr = lr
        self.val_len = val_len
        self.MTS = MTS
        self.test = test
        self.end_index = end_index
        self.normalize = normalize
        self.kernel_size = kernel_size

        # Normalize the data if required
        if normalize:
            Y = MTS
            m = np.mean(Y[:, 0 : self.end_index], axis=1)
            s = np.std(Y[:, 0 : self.end_index], axis=1)
            s += 1.0
            Y = (Y - m[:, None]) / s[:, None]
            mini = np.abs(np.min(Y))
            self.MTS = Y + mini
            self.m = m
            self.s = s
            self.mini = mini

        # Initialize the Temporal Convolutional Network
        self.seq = TemporalConvNet(num_inputs=self.num_inputs, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout, init=True)
        self.seq = self.seq.float()
        self.Dataloader = data_loader(MTS=self.MTS, ver_batch_size=ver_batch_size, hor_batch_size=hor_batch_size, end_index=end_index, val_len=val_len)
        self.val_len = val_len
        if use_cuda:
            self.seq = self.seq.cuda()

    def __loss__(self, out, target, dic=None):
        """
        Calculate the loss between the model output and the target.
        
        Arguments:
        out: model output
        target: ground truth target
        dic: additional dictionary (not used here)
        
        Returns:
        loss: normalized L1 loss
        """
        criterion = nn.L1Loss()
        return criterion(out, target) / torch.abs(target.data).mean()

    def __prediction__(self, data):
        """
        Make a prediction using the model.
        
        Arguments:
        data: input data
        
        Returns:
        out: model output
        dic: additional dictionary (not used here)
        """
        dic = None
        out = self.seq(data)
        return out, dic

    def train_model(self, early_stop=False, tenacity=3):
        """
        Train the Temporal Convolutional Network model.
        
        Arguments:
        early_stop: flag to enable early stopping
        tenacity: number of epochs to wait before early stopping
        
        Returns:
        None
        """
        print("Training TCN")
        if use_cuda:
            self.seq = self.seq.cuda()
        
        optimizer = optim.Adam(params=self.seq.parameters(), lr=self.lr)
        iter_count = 0
        loss_all = []
        loss_test_all = []
        vae = float("inf")
        scount = 0
        
        while self.Dataloader.epoch < self.num_epochs:
            last_epoch = self.Dataloader.epoch
            inp, out_target, _, _ = self.Dataloader.next_batch()
            
            if self.test:
                inp_test, out_target_test, _, _ = self.Dataloader.supply_test()
            
            current_epoch = self.Dataloader.epoch
            
            if use_cuda:
                inp = inp.cuda()
                out_target = out_target.cuda()
            
            inp = Variable(inp)
            out_target = Variable(out_target)
            optimizer.zero_grad()
            
            out, dic = self.__prediction__(inp)
            loss = self.__loss__(out, out_target, dic)
            iter_count += 1
            
            for p in self.seq.parameters():
                p.requires_grad = True
            
            loss.backward()
            
            for p in self.seq.parameters():
                p.grad.data.clamp_(max=1e5, min=-1e5)
            
            optimizer.step()
            loss_all.append(loss.cpu().item())
            
            if self.test:
                if use_cuda:
                    inp_test = inp_test.cuda()
                    out_target_test = out_target_test.cuda()
                
                inp_test = Variable(inp_test)
                out_target_test = Variable(out_target_test)
                out_test, dic = self.__prediction__(inp_test)
                losst = self.__loss__(out_test, out_target_test, dic)
                loss_test_all.append(losst.cpu().item())
            
            if current_epoch > last_epoch:
                ve = loss_test_all[-1]
                print("Entering Epoch# ", current_epoch)
                print("Train Loss:", np.mean(loss_all))
                print("Validation Loss:", ve)
                
                if ve <= vae:
                    vae = ve
                    scount = 0
                    self.saved_seq = pickle.loads(pickle.dumps(self.seq))
                else:
                    scount += 1
                    if scount > tenacity and early_stop:
                        self.seq = self.saved_seq
                        if use_cuda:
                            self.seq = self.seq.cuda()
                        break