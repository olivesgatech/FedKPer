import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["weighted_loss"]


class weighted_loss(nn.Module):

    def __init__(self):
        super(weighted_loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        #print('============> Beta param: ', self.beta)
        self.tau = 1
        self.lam_max = 10

    def forward(self, logits, targets, dg_logits):


        ce_loss = self.CE(logits, targets)

        global_ce = self.CE(dg_logits, targets)

        beta = 1/(global_ce + 1e-8)
        beta = torch.clamp(beta, max=self.lam_max)

        kl_loss = self.kl_loss(logits, dg_logits)
        loss = ce_loss + beta * kl_loss

        return loss, kl_loss.item(), beta, ce_loss.item()

    def kl_loss(self, logits, dg_logits):

        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        with torch.no_grad():
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = (self.tau ** 2) * self.KLDiv(pred_probs, dg_probs)

        return loss
