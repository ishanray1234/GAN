import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ) 

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim): 
        super().__init__() 
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 3e-4
z_dim = 64
image_dim = 784  # 28x28 images flattened
batch_size = 32
num_epochs = 50

gen = Generator(z_dim, image_dim).to(device)
disc = Discriminator(image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
disc_optimizer = optim.Adam(disc.parameters(), lr=lr)
gen_optimizer = optim.Adam(gen.parameters(), lr=lr) 
criterion = nn.BCELoss()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])
mnist = datasets.MNIST(root='dataset/', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)
writer_fake = SummaryWriter(f"logs/Vanilla_GAN/fake")
writer_real = SummaryWriter(f"logs/Vanilla_GAN/real")
step = 0

for epoch in range(num_epochs): 
    for idx, (real, _) in enumerate(train_loader):
        real = real.view(-1, image_dim).to(device)
        batch_size = real.shape[0]
        
        ## Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn((batch_size, z_dim)).to(device)
        fake = gen(noise)
        
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        
        lossD = (lossD_real + lossD_fake) / 2
          
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        disc_optimizer.step()
        
        ## Train Generator: maximize log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        
        gen.zero_grad()
        lossG.backward()
        gen_optimizer.step()
        
        if idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {idx}/{len(train_loader)} Loss D: {lossD:.4f}, Loss G: {lossG:.4f}")
            with torch.no_grad():
                fake = gen(fixed_noise).view(-1, 1, 28, 28)
                real = real.reshape(-1, 1, 28, 28)
                real_grid = torchvision.utils.make_grid(real, normalize=True)
                fake_grid = torchvision.utils.make_grid(fake, normalize=True)
                writer_real.add_image("Real", real_grid, global_step=step)
                writer_fake.add_image("Fake", fake_grid, global_step=step)
            step += 1
    print(f"Epoch [{epoch}/{num_epochs}] completed.")

  