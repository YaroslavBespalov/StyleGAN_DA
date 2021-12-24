import argparse
import sys, os
import time
from itertools import chain

from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch import Tensor

from gan.loss.stylegan import StyleGANLoss
from gan.models.stylegan import StyleGanModel
from src.examples.style_progressive import StyleDisc, StyleTransform
from gan.loss.loss_base import Loss
from gan.loss.perceptual.psp import PSPLoss
from gan.nn.stylegan.generator import Decoder, Generator, FromStyleConditionalGenerator
from gan.nn.stylegan.style_encoder import GradualStyleEncoder
from parameters.run import RuntimeParameters

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../../gans/'))
import torch
from dataset.lazy_loader import LazyLoader
from parameters.path import Paths


def jointed_loader(loader1, loader2):
    while True:
        yield next(loader1)
        yield next(loader2)

def send_images_to_tensorboard(writer, data: Tensor, name: str, iter: int, count=8, normalize=True, range=(-1, 1)):
    with torch.no_grad():
        grid = make_grid(
            data[0:count], nrow=count, padding=2, pad_value=0, normalize=normalize, range=range,
            scale_each=False)
        writer.add_image(name, grid, iter)

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    parents=[
        RuntimeParameters(),
    ]
)
args = parser.parse_args()
for k in vars(args):
    print(f"{k}: {vars(args)[k]}")

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

writer = SummaryWriter(f"{Paths.default.board()}/hm2img{int(time.time())}")

image_generator = Generator(FromStyleConditionalGenerator(256, 512)).cuda()
decoder = Decoder(image_generator)
style_enc = GradualStyleEncoder(50, 1, mode="ir", style_count=14).cuda()

style_transform = StyleTransform()
style_disc = StyleDisc().cuda()

gan_model = StyleGanModel(style_transform, StyleGANLoss(style_disc), (0.001, 0.0015))

loader = jointed_loader(LazyLoader.domain_adaptation_philips15().loader_train_inf,
                        LazyLoader.domain_adaptation_siemens3().loader_train_inf)

rec_loss = PSPLoss(id_lambda=0).cuda()
style_opt = Adam(style_enc.parameters(), lr=1e-4)
gen_opt = Adam(image_generator.parameters(), lr=0.001)

for i in range(100000):

    batch = next(loader)
    image = batch['image'].to(device)
    latent = style_enc(image)

    # ll = style_transform([latent[:, k] for k in range(latent.shape[1])])
    # D = style_disc(ll)
    # print(D.shape)
    res = StyleTransform().forward([latent[:, k] for k in range(latent.shape[1])])

    reconstructed = decoder.forward([latent[:, k] for k in range(latent.shape[1])])

    loss: Loss = rec_loss(image, image, reconstructed, latent) # восстановление изображения по стилю
    loss.minimize_step(style_opt, gen_opt)

    batch_x = next(LazyLoader.domain_adaptation_philips15().loader_train_inf)
    batch_y = next(LazyLoader.domain_adaptation_siemens3().loader_train_inf)
    image_x, image_y = batch_x['image'].to(device), batch_y['image'].to(device)
    latent_x, latent_y = style_enc(image_x), style_enc(image_y).detach()

    fake_style = style_transform([latent_x[:, k] for k in range(latent_x.shape[1])])
    real_style = style_transform([latent_y[:, k] for k in range(latent_y.shape[1])])

    gan_model.discriminator_train([real_style], [fake_style])
    gan_model.generator_loss([hm_ref], [hm_pred]).__mul__(coefs["obratno"]) \
        .minimize_step(gan_model_obratno.optimizer.opt_min)


    if i % 10 == 0:
        print(loss.item())

    if i % 100 == 0:
        print(i)
        with torch.no_grad():
            writer.add_scalar("loss_slavika", loss.item(), i)
            send_images_to_tensorboard(writer, reconstructed, "REC", i)
            send_images_to_tensorboard(writer, image, "REAL", i)


