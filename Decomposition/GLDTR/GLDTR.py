from __future__ import print_function
import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from GLDTR.TCN import *
import pickle

use_cuda = True
"""
    Refference Code: https://github.com/rajatsen91/deepglo/blob/master/DeepGLO/DeepGLO.py
"""
class GLDTR(object):
    def __init__(
        self,
        MTS,
        ver_batch_size=1,
        hor_batch_size=256,
        TCN_channels=[32, 32, 32, 32, 1],
        kernel_size=7,
        dropout=0.1,
        rank=64,
        lr=0.0005,
        val_len=1000,
        end_index=4999,
        normalize=True,
    ):
        # Initialize class variables
        self.dropout = dropout
        self.TCN = TemporalConvNet(
            num_inputs=1,
            num_channels=TCN_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            init=True,
        )

        # Normalize the input data if required
        if normalize:
            self.s = np.std(MTS[:, 0:end_index], axis=1) + 1.0
            self.m = np.mean(MTS[:, 0:end_index], axis=1)
            self.MTS = (MTS - self.m[:, None]) / self.s[:, None]
            self.mini = np.abs(np.min(self.MTS))
            self.MTS = self.MTS + self.mini 

        else:
            self.MTS = MTS

        self.normalize = normalize
        n, T = self.MTS.shape
        t0 = end_index + 1

        if t0 > T:
            self.MTS = np.hstack([self.MTS, self.MTS[:, -1].reshape(-1, 1)])

        # Initialize factor matrices

        self.global_matrix = torch.normal(torch.zeros(rank, t0).float(), 0.1).float()
        self.scores = torch.normal(torch.zeros(n, rank).float(), 0.1).float()
                
        self.ver_batch_size = ver_batch_size
        self.hor_batch_size = hor_batch_size
        self.TCN_channels = TCN_channels
        self.rank = rank
        self.kernel_size = kernel_size
        self.lr = lr
        self.val_len = val_len
        self.end_index = end_index
        self.Dataloader = data_loader(
            MTS=self.MTS,
            ver_batch_size=ver_batch_size,
            hor_batch_size=hor_batch_size,
            end_index=end_index,
            val_len=val_len,
            shuffle=False,
        )

    def tensor2d_to_temporal(self, T):
        """
        Convert a 2D tensor to a temporal tensor
        """
        T = T.view(1, T.size(0), T.size(1))
        T = T.transpose(0, 1)
        return T

    def temporal_to_tensor2d(self, T):
        """
        Convert a temporal tensor to a 2D tensor
        """
        T = T.view(T.size(0), T.size(2))
        return T

    def step_global_matrix_loss(self, inp, out, last_vindex, last_hindex, reg=0.2):
        """
        Compute and update the loss for factor X
        """
        global_matrix_out = self.global_matrix[:, last_hindex + 1 : last_hindex + 1 + out.size(2)]
        scores_out = self.scores[self.Dataloader.I[last_vindex : last_vindex + out.size(0)], :]
        
        if use_cuda:
            global_matrix_out = global_matrix_out.cuda()
            scores_out = scores_out.cuda()

        global_matrix_out = Variable(global_matrix_out, requires_grad=True)
        out = self.temporal_to_tensor2d(out)
        optim_X = optim.Adam(params=[global_matrix_out], lr=self.lr)
        Hout = torch.matmul(scores_out, global_matrix_out)
        optim_X.zero_grad()
        loss = torch.mean(torch.pow(Hout - out.detach(), 2))
        l2 = torch.mean(torch.pow(global_matrix_out, 2))
        r = loss.detach() / l2.detach()
        loss = loss + r * reg * l2
        loss.backward()
        optim_X.step()
        self.global_matrix[:, last_hindex + 1 : last_hindex + 1 + inp.size(2)] = global_matrix_out.cpu().detach()
        return loss

    def step_scores_loss(self, inp, out, last_vindex, last_hindex, reg=0.2):
        """
        Compute and update the loss for factor F
        """
        global_matrix_out = self.global_matrix[:, last_hindex + 1 : last_hindex + 1 + out.size(2)]
        scores_out = self.scores[self.Dataloader.I[last_vindex : last_vindex + out.size(0)], :]
        
        if use_cuda:
            global_matrix_out = global_matrix_out.cuda()
            scores_out = scores_out.cuda()

        scores_out = Variable(scores_out, requires_grad=True)
        optim_F = optim.Adam(params=[scores_out], lr=self.lr)
        out = self.temporal_to_tensor2d(out)
        Hout = torch.matmul(scores_out, global_matrix_out)
        optim_F.zero_grad()
        loss = torch.mean(torch.pow(Hout - out.detach(), 2))
        l2 = torch.mean(torch.pow(scores_out, 2))
        r = loss.detach() / l2.detach()
        loss = loss + r * reg * l2
        loss.backward()
        optim_F.step()
        self.scores[self.Dataloader.I[last_vindex : last_vindex + inp.size(0)], :] = scores_out.cpu().detach()
        return loss

    def step_temporal_loss_global_matrix(self, inp, last_vindex, last_hindex, lam=0.2):
        """
        Compute and update the temporal loss for factor X
        """
        Xin = self.global_matrix[:, last_hindex : last_hindex + inp.size(2)]
        global_matrix_out = self.global_matrix[:, last_hindex + 1 : last_hindex + 1 + inp.size(2)]
        
        for p in self.TCN.parameters():
            p.requires_grad = False
        
        if use_cuda:
            Xin = Xin.cuda()
            global_matrix_out = global_matrix_out.cuda()

        Xin = Variable(Xin, requires_grad=True)
        global_matrix_out = Variable(global_matrix_out, requires_grad=True)
        optim_out = optim.Adam(params=[global_matrix_out], lr=self.lr)
        Xin = self.tensor2d_to_temporal(Xin)
        global_matrix_out = self.tensor2d_to_temporal(global_matrix_out)
        hatX = self.TCN(Xin)
        optim_out.zero_grad()
        loss = lam * torch.mean(torch.pow(global_matrix_out - hatX.detach(), 2))
        loss.backward()
        optim_out.step()
        temp = self.temporal_to_tensor2d(global_matrix_out.detach())
        self.global_matrix[:, last_hindex + 1 : last_hindex + 1 + inp.size(2)] = temp
        return loss

    def predict_future_batch(self, model, inp, future=10, cpu=True):
        """
        Predict future values for a batch of input sequences
        """
        if cpu:
            model = model.cpu()
            inp = inp.cpu()
        else:
            inp = inp.cuda()

        out = model(inp)
        output = out[:, :, out.size(2) - 1].view(out.size(0), out.size(1), 1)
        out = torch.cat((inp, output), dim=2)
        torch.cuda.empty_cache()
        
        for i in range(future - 1):
            inp = out
            out = model(inp)
            output = out[:, :, out.size(2) - 1].view(out.size(0), out.size(1), 1)
            out = torch.cat((inp, output), dim=2)
            torch.cuda.empty_cache()

        out = self.temporal_to_tensor2d(out)
        out = np.array(out.cpu().detach())
        return out

    def predict_future(self, model, inp, future=10, cpu=True, bsize=90):
        """
        Predict future values for the entire input sequence
        """
        n = inp.size(0)
        inp = inp.cpu()
        ids = np.arange(0, n, bsize)
        ids = list(ids) + [n]
        out = self.predict_future_batch(model, inp[ids[0] : ids[1], :, :], future, cpu)
        torch.cuda.empty_cache()

        for i in range(1, len(ids) - 1):
            temp = self.predict_future_batch(
                model, inp[ids[i] : ids[i + 1], :, :], future, cpu
            )
            torch.cuda.empty_cache()
            out = np.vstack([out, temp])

        out = torch.from_numpy(out).float()
        return self.tensor2d_to_temporal(out)

    def predict_global(
        self, ind, last_step=100, future=10, cpu=False, normalize=False, bsize=90
    ):
        """
        Predict global future values for given indices
        """
        if ind is None:
            ind = np.arange(self.MTS.shape[0])
        
        if cpu:
            self.TCN = self.TCN.cpu()
        
        self.TCN = self.TCN.eval()
        rg = 1 + 2 * (self.kernel_size - 1) * 2 ** (len(self.TCN_channels) - 1)
        X = self.global_matrix[:, last_step - rg : last_step]
        X = self.tensor2d_to_temporal(X)
        
        outX = self.predict_future(
            model=self.TCN, inp=X, future=future, cpu=cpu, bsize=bsize
        )
        outX = self.temporal_to_tensor2d(outX)
        
        F = self.scores
        Y = torch.matmul(F, outX)
        Y = np.array(Y[ind, :].cpu().detach())
        
        self.TCN = self.TCN.cuda()
        del F
        torch.cuda.empty_cache()
        
        for p in self.TCN.parameters():
            p.requires_grad = True
        
        if normalize:
            Y = Y - self.mini
            Y = Y * self.s[ind, None] + self.m[ind, None]
        
        return Y

    def train_TCN(self, MTS, num_epochs=20, early_stop=False, tenacity=3):
        """
        Train the TCN model
        """
        seq = self.TCN
        num_channels = self.TCN_channels
        kernel_size = self.kernel_size
        ver_batch_size = min(self.ver_batch_size, MTS.shape[0] / 2)

        for p in seq.parameters():
            p.requires_grad = True

        TC = TCN(
            MTS=MTS,
            num_inputs=1,
            num_channels=num_channels,
            kernel_size=kernel_size,
            ver_batch_size=ver_batch_size,
            hor_batch_size=self.hor_batch_size,
            normalize=False,
            end_index=self.end_index - self.val_len,
            val_len=self.val_len,
            lr=self.lr,
            num_epochs=num_epochs,
        )

        TC.train_model(early_stop=early_stop, tenacity=tenacity)
        self.TCN = TC.seq

    def train_factors(
        self,
        reg_X=0.2,
        reg_F=0.2,
        mod=5,
        early_stop=False,
        tenacity=3,
        ind=None,
        seed=False,
    ):
        """
        Train the factor matrices X and F
        """
        self.Dataloader.epoch = 0
        self.Dataloader.vindex = 0
        self.Dataloader.hindex = 0
        
        if use_cuda:
            self.TCN = self.TCN.cuda()
        
        for p in self.TCN.parameters():
            p.requires_grad = True

        l_F = [0.0]
        l_X = [0.0]
        l_X_temporal = [0.0]
        iter_count = 0
        vae = float("inf")
        scount = 0
        global_matrix_best = self.global_matrix.clone()
        scores_best = self.scores.clone()
        
        while self.Dataloader.epoch < self.num_epochs:
            last_epoch = self.Dataloader.epoch
            last_vindex = self.Dataloader.vindex
            last_hindex = self.Dataloader.hindex
            
            inp, out, _, _ = self.Dataloader.next_batch(option=1)
            
            if use_cuda:
                inp = inp.float().cuda()
                out = out.float().cuda()
            
            if iter_count % mod >= 0:
                l1 = self.step_scores_loss(inp, out, last_vindex, last_hindex, reg=reg_F)
                l_F = l_F + [l1.cpu().item()]
            
            if iter_count % mod >= 0:
                l1 = self.step_global_matrix_loss(inp, out, last_vindex, last_hindex, reg=reg_X)
                l_X = l_X + [l1.cpu().item()]
            
            if not seed and iter_count % mod == 1:
                l2 = self.step_temporal_loss_global_matrix(inp, last_vindex, last_hindex)
                l_X_temporal = l_X_temporal + [l2.cpu().item()]
            
            iter_count += 1

            if self.Dataloader.epoch > last_epoch:
                print("Entering Epoch# ", self.Dataloader.epoch)
                print("Factorization Loss F: ", np.mean(l_F))
                print("Factorization Loss X: ", np.mean(l_X))
                print("Temporal Loss X: ", np.mean(l_X_temporal))
                
                if ind is None:
                    ind = np.arange(self.MTS.shape[0])
                
                inp = self.predict_global(
                    ind,
                    last_step=self.end_index - self.val_len,
                    future=self.val_len,
                    cpu=False,
                )
                
                R = self.MTS[ind, self.end_index - self.val_len : self.end_index]
                S = inp[:, -self.val_len : :]
                ve = np.abs(R - S).mean() / np.abs(R).mean()
                print("Validation Loss (Global): ", ve)
                
                if ve <= vae:
                    vae = ve
                    scount = 0
                    global_matrix_best = self.global_matrix.clone()
                    scores_best = self.scores.clone()
                    TCNbest = pickle.loads(pickle.dumps(self.TCN))
                else:
                    scount += 1
                    if scount > tenacity and early_stop:
                        print("Early Stopped")
                        self.global_matrix = global_matrix_best
                        self.scores = scores_best
                        self.TCN = TCNbest
                        if use_cuda:
                            self.TCN = self.TCN.cuda()
                        break
        
        return self.global_matrix, self.scores

    def train_GLDTR(
        self, init_epochs=100, alt_iters=10, tenacity=7, mod=5
    ):
        """
        Train all models using alternating training strategy
        """
        print("Initializing Factors.....")
        self.num_epochs = init_epochs
        self.train_factors()

        if alt_iters % 2 == 1:
            alt_iters += 1

        print("Starting Alternate Training.....")

        for i in range(1, alt_iters):
            if i % 2 == 0:
                print(
                    "--------------------------------------------Training global martix and scores. Iter#: "
                    + str(i)
                    + "-------------------------------------------------------"
                )
                self.num_epochs = 100
                X, F = self.train_factors(
                    seed=False, early_stop=True, tenacity=tenacity, mod=mod
                )
            else:
                print(
                    "--------------------------------------------Training TCN. Iter#: "
                    + str(i)
                    + "-------------------------------------------------------"
                )
                self.num_epochs = 100
                T = np.array(self.global_matrix.cpu().detach())
                self.train_TCN(
                    MTS=T,
                    num_epochs=self.num_epochs,
                    early_stop=True,
                    tenacity=tenacity,
                )

        return X, F