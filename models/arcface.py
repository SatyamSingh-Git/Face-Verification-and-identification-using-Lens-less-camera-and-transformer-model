import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcMarginProduct(nn.Module):
    r"""Sub-center ArcFace with K sub-centers per class.

    When K=1, this is identical to standard ArcFace.
    When K>1, each class has K prototype vectors; the closest one is used
    for the angular margin, making the model robust to noisy samples.

    Args:
        in_features: size of each input sample (embedding dim)
        out_features: number of classes
        s: feature scale (default 64.0)
        m: angular margin (default 0.50)
        K: number of sub-centers per class (default 3)
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.50, K=3, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.K = K

        # K sub-centers per class: weight shape = (out_features * K, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(out_features * K, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # Normalize input and weight
        cosine_all = F.linear(F.normalize(input), F.normalize(self.weight))
        # cosine_all shape: (B, out_features * K)

        if self.K > 1:
            # Reshape to (B, out_features, K) and take the max over sub-centers
            cosine_all = cosine_all.view(-1, self.out_features, self.K)
            cosine, _ = cosine_all.max(dim=2)  # (B, out_features)
        else:
            cosine = cosine_all

        if label is None:
            # Inference: return scaled cosine (no margin)
            return cosine * self.s

        # Training: apply angular margin
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot label
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Apply margin only to the target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
