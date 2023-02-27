import torch
import torch.nn as nn
from models.encoders.psp_encoders import GradualStyleEncoder, BackboneEncoderUsingLastLayerIntoWPlus
from models.stylegan2.model import Generator
device = "cuda"
x = torch.randn(1, 3, 256, 256).to(device)

class SE(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = GradualStyleEncoder(50, mode='ir_se')
        self.dec = Generator(1024, style_dim=512, n_mlp=8)
    def forward(self, images):
        latents = self.enc(images)
        truncation = 0.7
        trunc = self.dec.mean_latent(4096).detach()
        trunc.requires_grad = False
        recon_img, _ = self.dec([latents],
                                 input_is_latent=True,
                                 truncation=truncation,
                                 truncation_latent=trunc,
                                 randomize_noise=False)
        return recon_img
net = SE().to(device)
y = net(x)
print(y.shape)