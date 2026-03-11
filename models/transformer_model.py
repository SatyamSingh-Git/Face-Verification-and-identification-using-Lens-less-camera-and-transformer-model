import torch
import torch.nn as nn
from torch.autograd import Variable

class CNNStem(nn.Module):
    def __init__(self, in_channels):
        super(CNNStem, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        return self.conv(x)

class DCT_ViT(nn.Module):
    def __init__(self, in_channels=3, num_classes=87, embed_dim=256, depth=6, num_heads=8, seq_length=16, num_subbands=5):
        super(DCT_ViT, self).__init__()
        
        self.num_subbands = num_subbands
        self.embed_dim = embed_dim
        
        # CNN Stems for each subband
        self.stems = nn.ModuleList([CNNStem(in_channels) for _ in range(num_subbands)])
        
        # Patch embedding: 64 channels, 8x8 patches -> embed_dim
        self.patch_embed = nn.Conv2d(64, embed_dim, kernel_size=8, stride=8)
        
        # Positional encodings
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, seq_length, embed_dim))
        self.subband_embed = nn.Embedding(num_subbands, embed_dim)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=1024, 
            dropout=0.1, 
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        x: (B, 15, 64, 64) - The stacked 5 subbands, 3 channels each
        """
        B = x.size(0)
        
        # Split into 5 subbands
        x_subbands = [x[:, i*3:(i+1)*3, :, :] for i in range(self.num_subbands)]
        
        tokens_list = []
        for i, (stem, x_sub) in enumerate(zip(self.stems, x_subbands)):
            # CNN stem -> (B, 64, 32, 32)
            features = stem(x_sub)
            
            # Patch embedding -> (B, embed_dim, 4, 4)
            patches = self.patch_embed(features)
            
            # Flatten spatial dims -> (B, embed_dim, 16) -> (B, 16, embed_dim)
            tokens = patches.flatten(2).transpose(1, 2)
            
            # Add spatial pos embedding and subband embedding
            tokens = tokens + self.spatial_pos_embed + self.subband_embed(torch.tensor(i, device=x.device)).unsqueeze(0).unsqueeze(0)
            
            tokens_list.append(tokens)
            
        # Concat all subband tokens -> (B, 5 * 16, embed_dim)
        x_encoded = torch.cat(tokens_list, dim=1)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_encoded = torch.cat((cls_tokens, x_encoded), dim=1)
        
        # Transformer encoder
        x_encoded = self.transformer_encoder(x_encoded)
        
        # Take the output of the CLS token
        cls_out = x_encoded[:, 0, :]
        
        # Classify
        out = self.classifier(cls_out)
        return out

if __name__ == '__main__':
    net = DCT_ViT()
    x = torch.randn(2, 15, 64, 64)
    out = net(Variable(x))
    print("Output size:", out.size())
    print("Parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))
