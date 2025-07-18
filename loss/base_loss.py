import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from . import LOSS
import random


def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits


def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()


@LOSS.register_module
class SCLLoss(nn.Module):
    def __init__(self, lam=1, temperature=0.1):
        super(SCLLoss, self).__init__()
        self.loss = self.scl_loss
        self.lam = lam
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_local_batch_size = None

    def scl_loss(self, feats, labels):
        # Normalize global features
        global_features = F.normalize(feats, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(global_features, global_features.T) / self.temperature

        # Create positive pairs mask
        positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        # Create negative pairs mask (1 - positive_mask)
        negative_mask = 1.0 - positive_mask

        # Compute log-softmax over the similarity matrix
        log_prob = F.log_softmax(similarity_matrix, dim=-1)

        # Compute positive loss (maximize similarity for positive pairs)
        positive_loss = -(positive_mask * log_prob).sum(dim=-1) / (positive_mask.sum(dim=-1) + 1e-6)

        # Compute negative loss (minimize similarity for negative pairs)
        negative_loss = -(negative_mask * log_prob).sum(dim=-1) / (negative_mask.sum(dim=-1) + 1e-6)

        # Combine positive and negative losses
        loss = positive_loss.mean() + negative_loss.mean()
        return loss

    def forward(self, x, labels):
        return self.loss(x, labels) * self.lam


@LOSS.register_module
class DenseLoss(nn.Module):
    def __init__(self, lam=1, temperature=0.1):
        super(DenseLoss, self).__init__()
        self.loss = self.densecl
        self.lam = lam
        self.temperature = temperature

    def densecl(self, q_b, k_b, q_grid, k_grid, labels):
        # Normalize features
        q_b = F.normalize(q_b, p=2, dim=1)  # (b, c, h, w)
        k_b = F.normalize(k_b, p=2, dim=1)  # (b, c, h, w)
        q_grid = F.normalize(q_grid, p=2, dim=1)  # (b, c, h, w)
        k_grid = F.normalize(k_grid, p=2, dim=1)  # (b, c, h*w)

        # Flatten the spatial dimensions
        q_b_flat = q_b.view(q_b.size(0), q_b.size(1), -1)  # (b, c, h*w)
        k_b_flat = k_b.view(k_b.size(0), k_b.size(1), -1)  # (b, c, h*w)
        similarity_matrix = torch.einsum('bci,bcj->bij', q_b_flat, k_b_flat)  # (b, h*w, h*w)

        # Get the index of the most similar features between q_b and k_b
        max_sim_idx = torch.argmax(similarity_matrix, dim=-1)  # (b, h*w)

        # k = 3
        # b, c, h, w = q_b.size()
        # q_b_flat = q_b.view(b, c, -1).transpose(1, 2)   # (b, h*w, c)

        # k_b_unfold = F.unfold(k_b, kernel_size=3, padding=1)  # (b, c*3*3, h*w)
        # k_b_unfold = k_b_unfold.view(b, c, 3 * 3, h * w).permute(0, 3, 1, 2)   # (b, h*w, c, 3*3)

        # similarity_local = torch.einsum('bkc,bkci->bki', q_b_flat, k_b_unfold)  # (b, h*w, 9)
        # max_sim_idx = torch.argmax(similarity_local, dim=-1)  # (b, h*w)

        # Flatten q_grid and k_grid for grid-level comparison
        q_grid_flat = q_grid.view(q_grid.size(0), q_grid.size(1), -1)  # (b, c, h*w)
        k_grid_flat = k_grid.view(k_grid.size(0), k_grid.size(1), -1)  # (b, c, h*w)

        # Calculate the positive similarity for the same class
        pos_sim = F.cosine_similarity(
            q_grid_flat,
            k_grid_flat.gather(2, max_sim_idx.unsqueeze(1).expand(-1, q_grid.size(1), -1)),
            dim=1
        )  # (b, h*w)

        # Apply temperature scaling to the positive similarity
        pos_sim = pos_sim / self.temperature

        # Prepare to store negative similarities (one negative sample per batch element)
        neg_sim_list = []

        for i in range(q_b.size(0)):
            # Get indices of all different-class samples
            neg_indices = torch.where(labels != labels[i].item())[0]

            # If no negative samples are available, skip this sample
            if len(neg_indices) == 0:
                return torch.tensor(0.0, device=q_b.device)

            # Randomly select one negative sample from different-class samples
            rand_idx = torch.randint(0, len(neg_indices), (1,))
            neg_k_grid_flat = k_grid_flat[neg_indices[rand_idx]]  # (1, c, h*w)

            # Compute the cosine similarity between q_grid_flat[i] and the selected negative sample
            neg_sim = F.cosine_similarity(q_grid_flat[i].unsqueeze(0), neg_k_grid_flat, dim=1)  # (1, h*w)

            # Store the negative similarity
            neg_sim_list.append(neg_sim)

        # Concatenate all negative similarities
        neg_sim = torch.cat(neg_sim_list, dim=0)  # (b, h*w)
        neg_sim = neg_sim / self.temperature  # Apply temperature scaling

        # Contrastive loss (InfoNCE Loss)
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-6))

        return loss.mean()

    def forward(self, q_b, k_b, q_grid, k_grid, labels):
        if not isinstance(q_b, list):
            q_b = [q_b]
            k_b = [k_b]
            q_grid = [q_grid]
            k_grid = [k_grid]

        loss = [self.densecl(q, k, q_g, k_g, labels) for q, k, q_g, k_g in zip(q_b, k_b, q_grid, k_grid)]
        loss = sum(loss) / len(loss)
        return loss * self.lam


@LOSS.register_module
class L1Loss(nn.Module):
    def __init__(self, lam=1):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')
        self.lam = lam

    def forward(self, input1, input2):
        input1 = input1 if isinstance(input1, list) else [input1]
        input2 = input2 if isinstance(input2, list) else [input2]
        loss = 0
        for in1, in2 in zip(input1, input2):
            loss += self.loss(in1, in2) * self.lam
        return loss


@LOSS.register_module
class L2Loss(nn.Module):
    def __init__(self, lam=1):
        super(L2Loss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')
        self.lam = lam

    def forward(self, input1, input2):
        input1 = input1 if isinstance(input1, list) else [input1]
        input2 = input2 if isinstance(input2, list) else [input2]
        loss = 0
        for in1, in2 in zip(input1, input2):
            loss += self.loss(in1, in2) * self.lam
        return loss


@LOSS.register_module
class CosLoss(nn.Module):
    def __init__(self, avg=True, flat=True, lam=1):
        super(CosLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity()
        self.lam = lam
        self.avg = avg
        self.flat = flat

    def forward(self, input1, input2):
        input1 = input1 if isinstance(input1, list) else [input1]
        input2 = input2 if isinstance(input2, list) else [input2]
        loss = 0
        for in1, in2 in zip(input1, input2):
            if self.flat:
                loss += (1 - self.cos_sim(in1.contiguous().view(in1.shape[0], -1),
                                          in2.contiguous().view(in2.shape[0], -1))).mean() * self.lam
            else:
                loss += (1 - self.cos_sim(in1.contiguous(), in2.contiguous())).mean() * self.lam
        return loss / len(input1) if self.avg else loss


@LOSS.register_module
class KLLoss(nn.Module):
    def __init__(self, lam=1):
        super(KLLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='mean')
        self.lam = lam

    def forward(self, input1, input2):
        # real, pred
        # teacher, student
        input1 = input1 if isinstance(input1, list) else [input1]
        input2 = input2 if isinstance(input2, list) else [input2]
        loss = 0
        for in1, in2 in zip(input1, input2):
            in1 = in1.permute(0, 2, 3, 1)
            in2 = in2.permute(0, 2, 3, 1)
            loss += self.loss(F.log_softmax(in2, dim=-1), F.softmax(in1, dim=-1)) * self.lam
        return loss


@LOSS.register_module
class LPIPSLoss(nn.Module):
    def __init__(self, lam=1):
        super(LPIPSLoss, self).__init__()
        self.loss = None
        self.lam = lam

    def forward(self, input1, input2):
        pass


@LOSS.register_module
class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True, lam=1):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
        self.lam = lam

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        loss *= self.lam
        return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        l = max_val - min_val
    else:
        l = val_range

    padd = window_size // 2
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    c1 = (0.01 * l) ** 2
    c2 = (0.03 * l) ** 2

    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret, ssim_map


@LOSS.register_module
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None, lam=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size).cuda()

        self.lam = lam

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        s_score, ssim_map = ssim(img1, img2, window=window, window_size=self.window_size,
                                 size_average=self.size_average)
        loss = (1.0 - s_score) * self.lam
        return loss


@LOSS.register_module
class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()

    def forward(self, input):
        loss = torch.fft.fft2(input).abs().mean()
        return loss


@LOSS.register_module
class SumLoss(nn.Module):
    def __init__(self, lam=1):
        super(SumLoss, self).__init__()
        self.lam = lam

    def forward(self, input1, input2):
        input1 = input1 if isinstance(input1, list) else [input1]
        input2 = input2 if isinstance(input2, list) else [input2]
        loss = 0
        for in1, in2 in zip(input1, input2):
            loss += (in1 + in2) * self.lam
        return loss


@LOSS.register_module
class CSUMLoss(nn.Module):
    def __init__(self, lam=1):
        super(CSUMLoss, self).__init__()
        self.lam = lam

    def forward(self, input):
        loss = 0
        for instance in input:
            _, _, h, w = instance.shape
            loss += torch.sum(instance) / (h * w) * self.lam
        return loss


@LOSS.register_module
class FFocalLoss(nn.Module):
    def __init__(self, lam=1, alpha=-1, gamma=4, reduction="mean"):
        super(FFocalLoss, self).__init__()
        self.lam = lam
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean() * self.lam
        elif self.reduction == "sum":
            loss = loss.sum() * self.lam

        return loss


@LOSS.register_module
class SegmentCELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, mask, pred):
        bsz, _, h, w = pred.size()
        pred = pred.view(bsz, 2, -1)
        mask = mask.view(bsz, -1).long()
        return self.criterion(pred, mask)
