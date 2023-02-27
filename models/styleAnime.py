"""
This file defines the core research contribution
"""
import matplotlib

matplotlib.use('Agg')
import torch

from torch import nn
from models.encoders import styleAnime_encoders
from models.stylegan2.model import Generator
from configs.paths_config import model_paths


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

class StyleAnime(nn.Module):

    def __init__(self, opts):
        super(StyleAnime, self).__init__()
        self.set_opts(opts)
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(256, 512, 8)
        #self.decoder = Generator(1024, 512, 8)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()
        

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = styleAnime_encoders.GradualStyleEncoder_Both(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading StyleAnime from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            # if input to encoder is not an RGB image, do not load the input layer weights
            if self.opts.label_nc != 0:
                encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            print('self.opts.stylegan_weights')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            if self.opts.learn_in_w:
                self.__load_latent_avg(self.opts)
            else:
                self.__load_latent_avg(self.opts)

    def forward(self, x, y, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None, ref=False):
        if not ref:
            if input_code:
                codes = x
            else:
                codes, codes_style = self.encoder(x, y)
                if self.opts.start_from_latent_avg:
                    if self.opts.learn_in_w:
                        codes = codes + self.latent_avg.repeat(1,codes.shape[1], 1)
                        codes_style = codes_style + self.latent_avg.repeat(codes_style.shape[0], 1)
                    else:
                        codes = codes + self.latent_avg.repeat(1, codes.shape[1], 1)
                        codes_style = codes_style + self.latent_avg.repeat(1, codes_style.shape[1], 1)

                codes = torch.cat((codes, codes_style), dim=1)
            if latent_mask is not None:
                
                for i in latent_mask:
                    if inject_latent is not None:
                        if alpha is not None:
                            codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                        else:
                            codes[:, i] = inject_latent[:, i]
                    else:
                        codes[:, i] = 0

            
            input_is_latent = not input_code
            images, result_latent = self.decoder([codes],
                                                input_is_latent=input_is_latent,
                                                randomize_noise=randomize_noise,
                                                return_latents=return_latents)
        if ref:
            images_1, result_latent_1 = self.decoder([x],
                                        input_is_latent=True,
                                        randomize_noise=randomize_noise,
                                        return_latents=return_latents)
            images = images_1
            result_latent = result_latent_1

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images, images_1

    def set_opts(self, opts):
        self.opts = opts

    
    def __load_latent_avg(self, opts):
        latents = torch.load('./pretrained_models/latent_avg.pt')
        self.latent_avg = latents[:, 0, :].squeeze()