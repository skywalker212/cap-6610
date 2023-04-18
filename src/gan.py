import argparse
import os

import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch import nn
from torch.autograd import Variable


parser = argparse.ArgumentParser(description="GAN MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=30,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--no-mps", action="store_true", default=False, help="disables macOS GPU training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--nz",
    type=int,
    default=100,
    metavar="N",
    help="dimensionality of the latent space",
)
parser.add_argument(
    "--lr", 
    type=float, 
    default=0.001, 
    metavar="0.XYZW",
    help="adam: learning rate"
)
parser.add_argument(
    "--b1", 
    type=float, 
    default=0.5, 
    metavar="0.X",
    help="adam: decay of first order momentum of gradient"
)
parser.add_argument(
    "--b2", 
    type=float, 
    default=0.9999, 
    metavar="0.XYZW",
    help="adam: decay of second order momentum of gradient"
)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

# set manual seed to make outputs consistant
torch.manual_seed(args.seed)

if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
curr_dirname = os.path.dirname(__file__)
data_directory = os.path.join(curr_dirname, "data")
results_directory_name = "results/gan"
os.makedirs(results_directory_name, exist_ok=True)
results_directory = os.path.join(curr_dirname, results_directory_name)
# TODO: normalize the images
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        data_directory, train=True, download=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])
    ),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)

img_shape = (1, 28, 28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # create a reusable block with #in_feat input features and #out_feat output features
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(args.nz, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # create a reusable block with #in_feat input features and #out_feat output features
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(int(np.prod(img_shape)), 512),
            *block(512, 256),
            *block(256, 128),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        out = self.model(img_flat)
        return out

# loss function
adversarial_loss = torch.nn.BCELoss()

# initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# optimizers
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(1, args.epochs+1):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)

        # Adversarial ground truths
        valid = torch.ones([real_imgs.size(0), 1], device=device)
        fake = torch.zeros([real_imgs.size(0), 1], device=device)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_generator.zero_grad()

        # Sample noise as generator input
        z = torch.randn(real_imgs.shape[0], args.nz, device=device)

        # Generate a batch of images
        gen_imgs = generator(z)

        # classify images using discriminator
        classifications = discriminator(gen_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(classifications, valid)

        g_loss.backward()
        optimizer_generator.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_discriminator.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        # detaching since the backprop does not need to run on the generator
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        # take the average of loss against generated images and loss against real images
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_discriminator.step()

        if i % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tDiscriminator Loss: {:.6f}\tGenerator Loss: {:.6f}".format(
                    epoch,
                    i * len(real_imgs),
                    len(dataloader.dataset),
                    100.0 * i / len(dataloader),
                    d_loss.item(),
                    g_loss.item()
                )
            )
    with torch.no_grad():
        n_images = 64
        sample = torch.randn(n_images, args.nz).to(device)
        sample = generator(sample).cpu()
        save_image(
            sample.view(n_images, *img_shape),
            results_directory + "/sample_" + str(epoch) + ".png",
        )