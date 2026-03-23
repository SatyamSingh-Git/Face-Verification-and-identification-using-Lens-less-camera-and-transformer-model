import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os


class HybridResNetTransformer(nn.Module):
    def __init__(self, in_channels=15, embed_dim=512, depth=4, num_heads=8, out_dim=768, backbone='resnet18'):
        super(HybridResNetTransformer, self).__init__()

        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.backbone_name = backbone

        # 1. Pretrained ResNet Backbone (with offline fallback)
        WEIGHT_FILES = {
            'resnet18': 'resnet18-f37072fd.pth',
            'resnet34': 'resnet34-b627a593.pth',
        }
        resnet_factory = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
        }
        if backbone not in resnet_factory:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from {list(resnet_factory.keys())}")

        try:
            resnet = resnet_factory[backbone](pretrained=True)
        except Exception:
            resnet = resnet_factory[backbone](pretrained=False)
            local_weights = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                          WEIGHT_FILES[backbone])
            if os.path.exists(local_weights):
                resnet.load_state_dict(torch.load(local_weights, map_location='cpu'))
                print(f'  [INFO] Loaded {backbone} weights from local file: {local_weights}')
            else:
                print(f'  [WARN] No pretrained {backbone} weights found. Training from scratch.')


        # Modify conv1 to accept 15 channels (5 DCT subbands x 3ch)
        old_conv = resnet.conv1
        self.conv1 = nn.Conv2d(in_channels, old_conv.out_channels,
                               kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                               padding=old_conv.padding, bias=old_conv.bias is not None)
        with torch.no_grad():
            self.conv1.weight.data = old_conv.weight.data.repeat(1, 5, 1, 1) / 5.0
            if old_conv.bias is not None:
                self.conv1.bias.data = old_conv.bias.data

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 2. Positional Encoding for 7x7 = 49 tokens
        self.pos_embed = nn.Parameter(torch.randn(1, 49, embed_dim) * 0.02)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, norm=nn.LayerNorm(embed_dim)
        )

        # 4. Projection head: 512 → out_dim (768)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )

    def _extract_feats(self, x_in):
        """Core feature extraction pipeline."""
        out = self.conv1(x_in)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # (B, 512, 7, 7)

        # Tokenize: flatten spatial dims
        out = out.flatten(2).transpose(1, 2)  # (B, 49, 512)
        out = out + self.pos_embed

        # Transformer
        out = self.transformer_encoder(out)

        # Global Mean Pooling
        out = out.mean(dim=1)  # (B, 512)

        # Project to higher dim
        out = self.projection(out)  # (B, out_dim)

        # L2 Normalize
        out = F.normalize(out, p=2, dim=1)
        return out

    def forward(self, x, tta=False):
        """
        x: (B, 15, 64, 64)
        Returns: (B, out_dim) L2-normalized embedding
        """
        # Resize to 224x224
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        if not tta:
            return self._extract_feats(x)
        else:
            import torchvision.transforms.functional as TF

            feat_orig = self._extract_feats(x)
            feat_flip = self._extract_feats(torch.flip(x, dims=[3]))
            feat_blur = self._extract_feats(TF.gaussian_blur(x, kernel_size=[3, 3]))

            # Center crop 90% and resize back
            crop_size = int(224 * 0.9)  # 201
            x_crop = TF.center_crop(x, [crop_size, crop_size])
            x_crop = F.interpolate(x_crop, size=(224, 224), mode='bilinear', align_corners=False)
            feat_crop = self._extract_feats(x_crop)

            # Brightness shift +10%
            feat_bright = self._extract_feats(x * 1.1)

            feat_avg = (feat_orig + feat_flip + feat_blur + feat_crop + feat_bright) / 5.0
            return F.normalize(feat_avg, p=2, dim=1)

    def freeze_backbone_early_layers(self):
        """Freeze conv1, bn1, layer1, layer2 — call at start of training."""
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters — call after warmup epochs."""
        for param in self.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    from arcface import ArcMarginProduct
    net = HybridResNetTransformer(embed_dim=512, out_dim=768)
    metric_fc = ArcMarginProduct(768, 87, K=3)

    x = torch.randn(2, 15, 64, 64)
    labels = torch.randint(0, 87, (2,))

    features = net(x)
    print("Features shape:", features.size())  # Expected: (2, 768)

    output = metric_fc(features, labels)
    print("ArcFace Output shape:", output.size())  # Expected: (2, 87)
    print("Model Parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))
    print("ArcFace Parameters:", sum(p.numel() for p in metric_fc.parameters() if p.requires_grad))
