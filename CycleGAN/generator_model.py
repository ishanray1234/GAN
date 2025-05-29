import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs)
            if down 
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True) if use_act else nn.Identity(),
        )
    
    def forward(self, x):
        return self.conv(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return x + self.block(x)
    
class Generator(nn.Module):
    def __init__(self, img_channels, num_features = 64, num_residuals=9):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )
        
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1) 
            ]
        )

        self.residuals = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
 
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features * 2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
            ]
        )

        self.last = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
    
    def forward(self, x):
        x = self.initial(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.residuals(x)
        for block in self.up_blocks:
            x = block(x)
        return torch.tanh(self.last(x))
    
def test():
    x = torch.randn((2, 3, 256, 256))  # Batch size of 1, 3 channels, 256x256 image
    model = Generator(img_channels=3, num_residuals=9)
    preds = model(x)
    print(preds.shape)  # Should output torch.Size([2, 3, 256, 256]) for a 256x256 input

if __name__ == "__main__":
    test()