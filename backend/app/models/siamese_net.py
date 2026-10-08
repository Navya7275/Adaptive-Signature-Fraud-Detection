"""
Siamese CNN for signature embedding and verification.

Architecture:
  - 4 convolutional blocks (Conv → BatchNorm → ReLU → MaxPool)
  - Global Average Pooling
  - FC layers → 128-dim embedding
  - Two branches share identical weights

Training uses Contrastive Loss:
  - Genuine pairs (same writer): pull embeddings together
  - Forged pairs (different writer): push embeddings apart

At inference: cosine similarity between two embeddings = verification score.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.config import EMBEDDING_DIM, MARGIN


class ConvBlock(nn.Module):
    """Conv2d → GroupNorm → ReLU → MaxPool

    GroupNorm instead of BatchNorm: BatchNorm lets a Siamese encoder
    encode batch-relative statistics that vanish in eval mode (running
    stats), collapsing all eval embeddings together while train loss
    looks fine. GroupNorm is batch-independent — train and eval behave
    identically, so the encoder must learn input-dependent features.
    """
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SqueezeExcite(nn.Module):
    """
    Channel attention: learns which feature channels matter for this
    image (stroke curvature, endings, flourishes...) and re-weights
    them. Cheap (~2*ch²/r params) and effective on sparse stroke images.
    """
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // r),
            nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ResidualBlock(nn.Module):
    """(Conv → GN → ReLU → Conv → GN → SE) + skip connection."""
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, ch),
        )
        self.se = SqueezeExcite(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.se(self.body(x)))


class SignatureEncoder(nn.Module):
    """
    CNN encoder: signature image (1, 150, 220) → embedding (128,)
    Custom architecture trained fully from scratch: downsampling conv
    stages with residual + squeeze-excite attention blocks at each scale
    (~2M parameters).
    """
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),       # (32, 75, 110)
            ConvBlock(32, 64),      # (64, 37, 55)
            ResidualBlock(64),
            ConvBlock(64, 128),     # (128, 18, 27)
            ResidualBlock(128),
            ConvBlock(128, 256),    # (256, 9, 13)
            ResidualBlock(256),
        )
        # Global MAX pooling — signatures are sparse strokes on empty
        # background; average pooling dilutes stroke features into the
        # background and collapses all embeddings together. Max pooling
        # keeps the strongest stroke response per channel.
        self.gap = nn.AdaptiveMaxPool2d(1)  # (256, 1, 1)
        # NO dropout in the embedding head. With contrastive loss the
        # network can minimize the objective by amplifying dropout noise:
        # every pair lands at distance margin/2 from randomness alone,
        # without ever using the input — and at eval time (dropout off)
        # all embeddings collapse to a constant. Regularization comes
        # from data augmentation and weight decay instead.
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # NOT normalized here. Contrastive loss operates on raw embeddings
        # (the SigNet recipe): on a unit sphere the max distance is 2.0,
        # so a margin near sqrt(2) demands forged pairs be near-antipodal —
        # geometrically infeasible for many writers at once, and training
        # settles into a constant-distance blob (loss = 0.5*(margin/2)^2).
        # Normalization happens at inference (get_embedding) for cosine.
        return x


class SiameseNetwork(nn.Module):
    """
    Two-branch Siamese network with shared encoder, plus a decision head.

    The decision head classifies |e1 - e2| (on the normalized manifold)
    as genuine/forged and is trained jointly with the contrastive loss.
    Classification gradients flow even when metric shaping is slow —
    this stabilizes contrastive training — and because it reads
    normalized embeddings it can also score the app's stored embeddings.
    """
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.encoder = SignatureEncoder(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, img1, img2):
        emb1 = self.encoder(img1)
        emb2 = self.encoder(img2)
        return emb1, emb2

    def classify(self, emb1, emb2):
        """Logit for 'forged' from a pair of (raw) embeddings."""
        e1 = F.normalize(emb1, p=2, dim=1)
        e2 = F.normalize(emb2, p=2, dim=1)
        return self.classifier(torch.abs(e1 - e2))

    def get_embedding(self, img):
        """Extract L2-normalized embedding for a single image (inference)."""
        return F.normalize(self.encoder(img), p=2, dim=1)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese training.

    label = 0: genuine pair (same writer) → pull together
    label = 1: forged pair (different writer) → push apart

    L = (1-Y) * 0.5 * D² + Y * 0.5 * max(0, margin - D)²
    """
    def __init__(self, margin=MARGIN, hard_fraction=0.25):
        super().__init__()
        self.margin = margin
        # Hard-example mining: the hardest `hard_fraction` of each batch
        # (near-miss forgeries, drifted genuines) is counted twice. Most
        # pairs are easy and teach nothing late in training — the extra
        # weight focuses learning on skilled forgeries that almost pass.
        self.hard_fraction = hard_fraction

    def forward(self, emb1, emb2, label):
        # SHAPES MATTER: dist must be (B,) to match label (B,).
        # With keepdim=True dist is (B,1) and broadcasting against (B,)
        # silently produces a (B,B) matrix — every distance paired with
        # every label — which scrambles supervision into a constant
        # pull-AND-push whose optimum is dist = margin/2 for all pairs.
        dist = F.pairwise_distance(emb1, emb2)   # (B,)
        label = label.view(-1)                   # (B,)
        # label: 0=genuine pair, 1=impostor pair
        loss = (1 - label) * 0.5 * dist.pow(2) + \
               label * 0.5 * F.relu(self.margin - dist).pow(2)

        k = max(1, int(loss.numel() * self.hard_fraction))
        hard = loss.topk(k).values
        return loss.mean() + hard.mean()


def cosine_similarity_score(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two embeddings.
    Returns a float in [0, 1] where 1 = identical.
    """
    sim = F.cosine_similarity(emb1, emb2, dim=1)
    # Clamp to [0, 1] since embeddings are L2-normalized
    score = torch.clamp(sim, 0.0, 1.0)
    return float(score.item())