import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class HybridResNetTransformer(nn.Module):
    def __init__(self, in_channels=15, embed_dim=512, depth=4, num_heads=8):
        super(HybridResNetTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        
        # 1. Pretrained ResNet18 Backbone
        # Use models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) for modern torchvision, 
        # or pretrained=True for older.
        resnet = models.resnet18(pretrained=True)
        
        # Modify the first conv layer to accept 15 channels instead of 3
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
        self.pos_embed = nn.Parameter(torch.randn(1, 49, embed_dim))
        
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
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth, norm=nn.LayerNorm(embed_dim))

    def forward(self, x, tta=False):
        """
        x: (B, 15, 64, 64)
        """
        # Resize to 224x224
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        def extract_feats(x_in):
            # ResNet Feature Extraction
            out = self.conv1(x_in)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out) # (B, 512, 7, 7)
            
            # Tokenization
            out = out.flatten(2).transpose(1, 2) # (B, 49, 512)
            out = out + self.pos_embed
            
            # Transformer
            out = self.transformer_encoder(out)
            
            # Global Mean Pooling
            out = out.mean(dim=1) # (B, 512)
            
            # L2 Normalize
            out = F.normalize(out, p=2, dim=1)
            return out
            
        if not tta:
            return extract_feats(x)
        else:
            # TTA: Original, Flipped, Blurred
            feat_orig = extract_feats(x)
            feat_flip = extract_feats(torch.flip(x, dims=[3]))
            
            import torchvision.transforms.functional as TF
            feat_blur = extract_feats(TF.gaussian_blur(x, kernel_size=[3, 3]))
            
            # Average and re-normalize
            feat_avg = (feat_orig + feat_flip + feat_blur) / 3.0
            return F.normalize(feat_avg, p=2, dim=1)

if __name__ == '__main__':
    from arcface import ArcMarginProduct
    net = HybridResNetTransformer(embed_dim=512)
    metric_fc = ArcMarginProduct(512, 87)
    
    x = torch.randn(2, 15, 64, 64)
    labels = torch.randint(0, 87, (2,))
    
    features = net(x)
    print("Features extracted shape:", features.size())
    
    output = metric_fc(features, labels)
    print("ArcFace Output size:", output.size())
    print("Hybrid Model Parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))
