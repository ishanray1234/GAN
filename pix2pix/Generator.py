import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down = True, act = 'relu', use_dropout = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect')
            if down else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2, inplace=True)
        )
        self.use_dropout = use_dropout 
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    
class Generator(nn.Module):
    def __init__(self, in_channels = 3, features = 64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        ) # 128
        
        self.down1 = Block(features, features * 2, down=True, act='lrelu', use_dropout=False) # 64
        self.down2 = Block(features * 2, features * 4, down=True, act='lrelu', use_dropout=False) # 32
        self.down3 = Block(features * 4, features * 8, down=True, act='lrelu', use_dropout=False) # 16
        self.down4 = Block(features * 8, features * 8, down=True, act='lrelu', use_dropout=False) # 8
        self.down5 = Block(features * 8, features * 8, down=True, act='lrelu', use_dropout=False) # 4
        self.down6 = Block(features * 8, features * 8, down=True, act='lrelu', use_dropout=False) # 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1, padding_mode='reflect'), # 1
            nn.ReLU()
        )

        self.up1 = Block(features * 8, features * 8, down=False, act='relu', use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act='relu', use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act='relu', use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act='relu', use_dropout=False)
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act='relu', use_dropout=False)
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act='relu', use_dropout=False)
        self.up7 = Block(features * 2 * 2, features, down=False, act='relu', use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.initial_down(x)  # 128
        d2 = self.down1(d1)        # 64
        d3 = self.down2(d2)        # 32
        d4 = self.down3(d3)        # 16
        d5 = self.down4(d4)        # 8
        d6 = self.down5(d5)        # 4
        d7 = self.down6(d6)        # 2
        bottleneck = self.bottleneck(d7)
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        return self.final_up(torch.cat([u7, d1], dim=1))

def test():
    N, in_channels, H, W = 8, 3, 256, 256
    x = torch.randn((N, in_channels, H, W))
    model = Generator(in_channels, features=64)
    output = model(x)
    print(output.shape)  # Expected output shape: (N, in_channels, H, W)

if __name__ == "__main__":
    test()
    # Expected output shape: (8, 3, 256, 256) for the default in_channels = 3
    # This means the model should output an image of the same size as the input