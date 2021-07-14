import torch.nn as nn
import torch
import numpy as np
#['x', 'image_w', 'bbox', 'num_bbox', 'image_h']
import numpy as np
import codecs,re


class ImageSelfAttention(nn.Module):
    """ Self-attention module for CNN's feature map.
    Inspired by: Zhang et al., 2018 The self-attention mechanism in SAGAN.
    """

    def __init__(self, planes):
        super(ImageSelfAttention, self).__init__()
        inner = planes // 8
        self.conv_f = nn.Conv1d(planes, inner, kernel_size=1, bias=False)
        self.conv_g = nn.Conv1d(planes, inner, kernel_size=1, bias=False)
        self.conv_h = nn.Conv1d(planes, planes, kernel_size=1, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        f = self.conv_f(x)
        g = self.conv_g(x)
        h = self.conv_h(x)
        sim_beta = torch.matmul(f.transpose(1, 2), g)
        beta = nn.functional.softmax(sim_beta, dim=1)
        o = torch.matmul(h, beta)
        return o


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):

        diagonal = scores.diag().view(-1, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()

c = torch.tensor([[ 1, 2, 3],[-1, 1, 4]], dtype= torch.float)
print(torch.norm(c, dim=1))