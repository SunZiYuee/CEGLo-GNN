import numpy as np
from torch import nn
from basicts.losses import masked_mae

def ceglognn_loss(prediction, real_value, normalized_probability, knn_graph, lambda_graph, null_val=np.nan):
    # graph structure learning loss
    B, N, N = normalized_probability.shape
    BCE_loss = nn.BCELoss()
    graph_regularization = BCE_loss(normalized_probability.view(B, N*N), knn_graph.view(B, N*N))
    # prediction loss
    loss_pred = masked_mae(preds=prediction, labels=real_value, null_val=null_val)
    # final loss
    loss = loss_pred + graph_regularization * lambda_graph
    return loss