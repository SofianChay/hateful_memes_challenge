from torch.autograd import Variable
import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np 

class FocalLoss(nn.Module):
    def __init__(self, gamma=.5, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class RocStar(nn.Module):
    
    def __init__(self):
        super(RocStar, self).__init__()
        self.gamma = torch.tensor(0.2, dtype=torch.float)

    def forward(self, _y_true, y_pred, _epoch_true, epoch_pred):
        """
        Nearly direct loss function for AUC.
        See article,
        C. Reiss, "Roc-star : An objective function for ROC-AUC that actually works."
        https://github.com/iridiumblue/articles/blob/master/roc_star.md
            _y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
            y_pred: `Tensor` . Predictions.
            gamma  : `Float` Gamma, as derived from last epoch.
            _epoch_true: `Tensor`.  Targets (labels) from last epoch.
            epoch_pred : `Tensor`.  Predicions from last epoch.
        """
        self.gamma = self.gamma.to(y_pred.device)
        #convert labels to boolean
        y_true = (_y_true>=0.50)
        epoch_true = (_epoch_true>=0.50)

        # if batch is either all true or false return small random stub value.
        if torch.sum(y_true)==0 or torch.sum(y_true) == y_true.shape[0]: return torch.sum(y_pred)*1e-8

        pos = y_pred[y_true]
        neg = y_pred[~y_true]

        epoch_pos = epoch_pred[epoch_true]
        epoch_neg = epoch_pred[~epoch_true]

        # Take random subsamples of the training set, both positive and negative.
        max_pos = 1000 # Max number of positive training samples
        max_neg = 1000 # Max number of positive training samples
        cap_pos = epoch_pos.shape[0]
        cap_neg = epoch_neg.shape[0]
        epoch_pos = epoch_pos[torch.rand_like(epoch_pos) < max_pos/cap_pos]
        epoch_neg = epoch_neg[torch.rand_like(epoch_neg) < max_neg/cap_pos]

        ln_pos = pos.shape[0]
        ln_neg = neg.shape[0]

        # sum positive batch elements agaionst (subsampled) negative elements
        if ln_pos>0 :
            pos_expand = pos.view(-1,1).expand(-1,epoch_neg.shape[0]).reshape(-1)
            neg_expand = epoch_neg.repeat(ln_pos)

            diff2 = neg_expand - pos_expand + self.gamma
            l2 = diff2[diff2>0]
            m2 = l2 * l2
            len2 = l2.shape[0]
        else:
            m2 = torch.tensor([0], dtype=torch.float, device=y_true.device)
            len2 = 0

        # Similarly, compare negative batch elements against (subsampled) positive elements
        if ln_neg>0 :
            pos_expand = epoch_pos.view(-1,1).expand(-1, ln_neg).reshape(-1)
            neg_expand = neg.repeat(epoch_pos.shape[0])

            diff3 = neg_expand - pos_expand + self.gamma
            l3 = diff3[diff3>0]
            m3 = l3*l3
            len3 = l3.shape[0]
        else:
            m3 = torch.tensor([0], dtype=torch.float, device=y_true.device)
            len3=0

        if (torch.sum(m2)+torch.sum(m3))!=0 :
          res2 = torch.sum(m2)/max_pos+torch.sum(m3)/max_neg
          #code.interact(local=dict(globals(), **locals()))
        else:
          res2 = torch.sum(m2)+torch.sum(m3)

        res2 = torch.where(torch.isnan(res2), torch.zeros_like(res2), res2)

        return res2

    def epoch_update_gamma(self, y_true, y_pred, delta=2):
        """
        Calculate gamma from last epoch's targets and predictions.
        Gamma is updated at the end of each epoch.
        y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
        y_pred: `Tensor` . Predictions.
        """
        DELTA = delta
        SUB_SAMPLE_SIZE = 2000.0
        pos = y_pred[y_true.nonzero()]
        neg = y_pred[(1 - y_true).nonzero()]
          # yo pytorch, no boolean tensors or operators ? Wassap ?
        # subsample the training set for performance
        cap_pos = pos.shape[0]
        cap_neg = neg.shape[0]
        pos = pos[torch.rand_like(pos) < SUB_SAMPLE_SIZE / cap_pos]
        neg = neg[torch.rand_like(neg) < SUB_SAMPLE_SIZE / cap_neg]
        ln_pos = pos.shape[0]
        ln_neg = neg.shape[0]
        pos_expand = pos.view(-1, 1).expand(-1, ln_neg).reshape(-1)
        neg_expand = neg.repeat(ln_pos)
        diff = neg_expand - pos_expand
        ln_All = diff.shape[0]
        Lp = diff[diff > 0] # because we're taking positive diffs, we got pos and neg flipped.
        ln_Lp = Lp.shape[0]-1
        diff_neg = -1.0 * diff[diff < 0]
        diff_neg = diff_neg.sort()[0]
        ln_neg = diff_neg.shape[0] - 1
        ln_neg = max([ln_neg, 0])
        left_wing = int(ln_Lp * DELTA)
        left_wing = max([0, left_wing])
        left_wing = min([ln_neg, left_wing])
        default_gamma=torch.tensor(0.2, dtype=torch.float, device=y_true.device)
        if diff_neg.shape[0] > 0:
          self.gamma = diff_neg[left_wing]
        else:
          self.gamma = default_gamma # default=torch.tensor(0.2, dtype=torch.float).cuda() #zoink


class MarginFocalLoss(nn.Module):
    
    def __init__(self):
        super(MarginFocalLoss, self).__init__()

    def forward(self, logit, truth):
        weight_pos = 2
        weight_neg = 1
        gamma = 2
        margin = 0.2
        em = np.exp(margin)
        logit = logit.view(-1)
        truth = truth.view(-1)
        log_pos = -F.logsigmoid(logit)
        log_neg = -F.logsigmoid(-logit)
        log_prob = truth * log_pos + (1-truth) * log_neg
        prob = torch.exp(-log_prob)
        margin = torch.log(em + (1-em) * prob)
        weight = truth * weight_pos + (1-truth) * weight_neg
        loss = margin + weight * (1 - prob) ** gamma * log_prob
        loss = loss.mean()
        return loss

