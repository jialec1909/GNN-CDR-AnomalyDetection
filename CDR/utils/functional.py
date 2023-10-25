import torch
import torch.nn as nn

def recons_cross_loss(x, x_):
    diff = torch.pow(x - x_, 2)
    error = torch.sqrt(torch.sum(diff, 1))
    # softmax_error = nn.functional.softmax(error, dim = -1)
    #
    # target = y.int()
    # # loss = nn.CrossEntropyLoss(softmax_error, target)
    # loss = nn.CrossEntropyLoss(error, target)

    return error

