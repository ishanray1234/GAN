import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channel_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: channel_img x 64 x 64
            nn.Conv2d(channel_img, features_d, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, kernel_size=4, stride=2, padding=1),  # 16x16
            self._block(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1),  # 8x8
            self._block(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),  # 1x1
            nn.Sigmoid()  # Output a probability
        )
        
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module): 
    def __init__(self, z_dim, channel_img, features_g):
        super(Generator, self).__init__() 
        self.gen = nn.Sequential(
            # input: z_dim x 1 x 1
            self._block(z_dim, features_g * 16, kernel_size=4, stride=1, padding=0),  # 4x4
            self._block(features_g * 16, features_g * 8, kernel_size=4, stride=2, padding=1),  # 8x8
            self._block(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1),  # 16x16
            self._block(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ConvTranspose2d(features_g * 2, channel_img, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.Tanh()  # Output image in range [-1, 1]
        )
        
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.gen(x) 
    
def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W  = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    weights_init(disc)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator output shape mismatch"

    gen = Generator(z_dim, in_channels, 8)
    weights_init(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator output shape mismatch"
