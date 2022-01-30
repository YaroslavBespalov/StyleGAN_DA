import argparse
import sys, os
import time
import json

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../../gans/'))
sys.path.append(os.path.join(sys.path[0], '../../src/'))
from itertools import chain

from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch import Tensor, nn

from examples.autoencoder_penalty import DecoderPenalty, EncoderPenalty
from optim.accumulator import Accumulator
from gan.loss.stylegan import StyleGANLoss
from gan.models.stylegan import StyleGanModel, CondStyleGanModel
from examples.style_progressive import StyleDisc, StyleTransform, Noise2Style, ConditionalStyleTransform
from gan.loss.loss_base import Loss
from gan.loss.perceptual.psp import PSPLoss
from gan.nn.stylegan.generator import Decoder, Generator, FromStyleConditionalGenerator
from gan.nn.stylegan.discriminator import Discriminator, CondBinaryDiscriminator
from gan.nn.stylegan.style_encoder import GradualStyleEncoder
from parameters.run import RuntimeParameters

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../../gans/'))
import torch
from dataset.lazy_loader import LazyLoader
from parameters.path import Paths


def jointed_loader(loader1, loader2):
    while True:
        yield 0, next(loader1)
        yield 1, next(loader2)

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

starting_model_number = 100000

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

writer = SummaryWriter(f"{Paths.default.board()}/StyleGAN_COND_DA_Celeba_{int(time.time())}")
#
weights = torch.load(
    f'{Paths.default.models()}/StyleGAN_COND_DA_Celeba_{str(starting_model_number).zfill(6)}.pt',
    map_location="cpu"
)
#{str(starting_model_number).zfill(6)}.pt',
# e -> s,cond -> s -> I -> D(I, cond)
# s, cond -> s -> (8, 14, 512)


image_generator = Generator(FromStyleConditionalGenerator(256, 512)).cuda()
decoder = Decoder(image_generator).cuda()
decoder.load_state_dict(weights["dec"])

noise_to_style = Noise2Style().cuda()
noise_to_style.load_state_dict(weights["n2s"])

style_transform = ConditionalStyleTransform().cuda()
style_transform.load_state_dict(weights["st_trfm"])

image_disc = CondBinaryDiscriminator(size=256).cuda()
image_disc.load_state_dict(weights["image_disc"])

gan_model = CondStyleGanModel(decoder, StyleGANLoss(image_disc), (0.001/2, 0.0015/2))

loader = jointed_loader(LazyLoader.celeba().loader, #[8,3,256,256]
                        LazyLoader.metfaces().loader)


gan_accumulator = Accumulator(gan_model.generator, decay=0.99, write_every=100)

noise_to_style_opt = Adam(noise_to_style.parameters(), lr=0.001/2)
style_transform_opt = Adam(style_transform.parameters(), lr=0.001/2)

for i in range(200001):

    coefs = json.load(open(os.path.join(sys.path[0], "../parameters/loss_params.json")))

    cond, batch = next(loader)
    cond = torch.ones(8, dtype=torch.int32).cuda() * cond
    real_image = batch.to(device)

    B = real_image.shape[0]
    s0 = noise_to_style(B)

    s1 = style_transform(s0, cond)
    s2 = style_transform(s0, 1 - cond)
    fake_image = decoder([s1[:, k] for k in range(s1.shape[1])])

    fcond = cond.type(torch.float32)
    gan_model.discriminator_train([real_image], [fake_image.detach()], [fcond])

    loss: Loss = Loss(
        gan_model.generator_loss([real_image], [fake_image], [fcond]).to_tensor() +
        nn.L1Loss()(s1, s2) * coefs["L1_coef"]
    )  # восстановление изображения по стилю
    loss.minimize_step(gan_model.optimizer.opt_min, style_transform_opt, noise_to_style_opt)

    gan_accumulator.step(i)


    if i % 10 == 0:
        print(loss.item())
        # writer.add_scalar("Styles_L1_Loss", style_l1.item(), i)
        writer.add_scalar("Loss", loss.item(), i)
        l1_dict = {f"style_{k}": nn.L1Loss()(s1[:, k], s2[:, k]) for k in range(14)}
        for k, v in l1_dict.items():
            writer.add_scalar(f'styles/{k}', v, i)

    if i % 100 == 0:
        print(i)
        with torch.no_grad():

            fake_image = decoder([s1[:, k] for k in range(s1.shape[1])])
            send_images_to_tensorboard(writer, fake_image, "FAKE 1", i)
            fake_image = decoder([s2[:, k] for k in range(s2.shape[1])])
            send_images_to_tensorboard(writer, fake_image, "FAKE 2", i)

            send_images_to_tensorboard(writer, real_image, "REAL", i)


    if i % 30000 == 0 and i > 0:
        torch.save(
            {
                'n2s': noise_to_style.state_dict(),
                'dec': decoder.state_dict(),
                'st_trfm': style_transform.state_dict(),
                'image_disc': image_disc.state_dict()
            },
            f'{Paths.default.models()}/StyleGAN_COND_DA_Celeba_{str(i + starting_model_number).zfill(6)}.pt',
        )

    # if i == 20001:
    #     break
