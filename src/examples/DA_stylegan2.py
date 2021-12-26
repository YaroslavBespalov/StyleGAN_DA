import argparse
import sys, os
import time

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../../gans/'))
sys.path.append(os.path.join(sys.path[0], '../../src/'))
from itertools import chain

from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch import Tensor, nn

from examples.autoencoder_penalty import DecoderPenalty, EncoderPenalty
from gan.loss.stylegan import StyleGANLoss
from gan.models.stylegan import StyleGanModel
from examples.style_progressive import StyleDisc, StyleTransform
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

starting_model_number = 0

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

writer = SummaryWriter(f"{Paths.default.board()}/StyleGAN_DA{int(time.time())}")

# weights = torch.load(
#     f'{Paths.default.models()}/StyleGAN_DA_{str(starting_model_number).zfill(6)}.pt',
#     map_location="cpu"
# )


image_generator = Generator(FromStyleConditionalGenerator(256, 512)).cuda()

decoder = Decoder(image_generator).cuda()
# decoder.load_state_dict(weights['dec'])

style_enc = GradualStyleEncoder(50, 1, mode="ir", style_count=14).cuda()
# style_enc.load_state_dict(weights['enc'])


style_transform = StyleTransform().cuda()
# style_transform.load_state_dict(weights['st_trfm'])

style_disc = StyleDisc().cuda()
# style_disc.load_state_dict(weights['st_disc'])

gan_model = StyleGanModel(style_transform, StyleGANLoss(style_disc), (0.001, 0.0015))

loader = jointed_loader(LazyLoader.domain_adaptation_philips15().loader_train_inf,
                        LazyLoader.domain_adaptation_siemens3().loader_train_inf)

rec_loss = PSPLoss(id_lambda=0).cuda()
style_opt = Adam(style_enc.parameters(), lr=2e-5)
gen_opt = Adam(image_generator.parameters(), lr=0.001)

dec_pen = DecoderPenalty(10)
enc_pen = EncoderPenalty(10)

for i in range(100000):

    batch = next(loader)
    image = batch['image'].to(device)
    latent = style_enc(image)

    # res = StyleTransform().forward([latent[:, k] for k in range(latent.shape[1])])

    reconstructed = decoder.forward([latent[:, k] for k in range(latent.shape[1])])

    loss: Loss = rec_loss(image, image, reconstructed, latent) # восстановление изображения по стилю
    loss.minimize_step(style_opt, gen_opt)

    if i % 10 == 0:
        latent = style_enc(image).detach()
        latent.requires_grad_(True)
        reconstructed = decoder.forward([latent[:, k] for k in range(latent.shape[1])])
        dec_pen(reconstructed, [latent]).minimize_step(gen_opt)

        image.requires_grad_(True)
        latent = style_enc(image)
        enc_pen(latent, [image]).minimize_step(style_opt)


    batch_x = next(LazyLoader.domain_adaptation_philips15().loader_train_inf)
    batch_y = next(LazyLoader.domain_adaptation_siemens3().loader_train_inf)
    image_x, image_y = batch_x['image'].to(device), batch_y['image'].to(device)
    latent_x, latent_y = style_enc(image_x).detach(), style_enc(image_y).detach()

    fake_style = style_transform(latent_x)
    real_style = latent_y

    gan_model.discriminator_train([real_style], [fake_style.detach()])
    Loss(
        gan_model.generator_loss([real_style], [fake_style]).to_tensor() +
        nn.L1Loss()(fake_style, real_style) * 10
     ).minimize_step(gan_model.optimizer.opt_min)


    if i % 10 == 0:
        print(loss.item())
        writer.add_scalar("loss_slavika", loss.item(), i)

    if i % 100 == 0:
        print(i)
        with torch.no_grad():

            reconstructed = decoder.forward([latent_y[:, k] for k in range(latent_y.shape[1])])
            send_images_to_tensorboard(writer, reconstructed, "Y REC", i)

            send_images_to_tensorboard(writer, image_x, "X", i)

            send_images_to_tensorboard(writer, image_y, "Y", i)

            fake_image = decoder.forward([fake_style[:, k] for k in range(fake_style.shape[1])])
            send_images_to_tensorboard(writer, fake_image, "X -> Y", i)

    if i % 2000 == 0 and i > 0:
        torch.save(
            {
                'dec': decoder.state_dict(),
                'enc': style_enc.state_dict(),
                'st_disc': style_disc.state_dict(),
                'st_trfm': style_transform.state_dict()
            },
            f'{Paths.default.models()}/StyleGAN_DA_{str(i + starting_model_number).zfill(6)}.pt',
        )

    if i == 20001:
        break
