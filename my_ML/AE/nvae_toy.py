
IN_KAGGLE = False
if IN_KAGGLE:
    out_dir = '/kaggle/working'
    DATA_DIR = '/kaggle/input/celeba-dataset/img_align_celeba' # or '../input'
    DATA_DIR = '/kaggle/input/animefacedataset' # or '../input'
else:
    out_dir = r'C:\Users\i_hat\Desktop\bastl\py\deep_larn\NVAE'
    DATA_DIR = r'C:\Users\i_hat\Desktop\losable\celeba\img_align_celeba'
    DATA_DIR = r'C:\Users\i_hat\Desktop\losable\anime_face_400'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm
from torch.nn import functional as F
import numpy as np
from functools import reduce
import os
from glob import glob
import cv2
import robust_loss_pytorch # pip install git+https://github.com/jonbarron/robust_loss_pytorch
import os
import matplotlib.pyplot as plt


def add_sn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return spectral_norm(m)
    else:
        return m


def reparameterize(mu, std):
    z = torch.randn_like(mu) * std + mu
    return z


def create_grid(h, w, device):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)


def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def random_uniform_like(tensor, min_val, max_val):
    return (max_val - min_val) * torch.rand_like(tensor) + min_val


def sample_from_discretized_mix_logistic(y, img_channels=3, log_scale_min=-7.):
    """

    :param y: Tensor, shape=(batch_size, 3 * num_mixtures * img_channels, height, width),
    :return: Tensor: sample in range of [-1, 1]
    """

    # unpack parameters, [batch_size, num_mixtures * img_channels, height, width] x 3
    logit_probs, means, log_scales = y.chunk(3, dim=1)

    temp = random_uniform_like(logit_probs, min_val=1e-5, max_val=1. - 1e-5)
    temp = logit_probs - torch.log(-torch.log(temp))

    ones = torch.eye(means.size(1) // img_channels, dtype=means.dtype, device=means.device)

    sample = []
    for logit_prob, mean, log_scale, tmp in zip(logit_probs.chunk(img_channels, dim=1),
                                                means.chunk(img_channels, dim=1),
                                                log_scales.chunk(img_channels, dim=1),
                                                temp.chunk(img_channels, dim=1)):
        # (batch_size, height, width)
        argmax = torch.max(tmp, dim=1)[1]
        B, H, W = argmax.shape

        one_hot = ones.index_select(0, argmax.flatten())
        one_hot = one_hot.view(B, H, W, mean.size(1)).permute(0, 3, 1, 2).contiguous()

        # (batch_size, 1, height, width)
        mean = torch.sum(mean * one_hot, dim=1)
        log_scale = torch.clamp_max(torch.sum(log_scale * one_hot, dim=1), log_scale_min)

        u = random_uniform_like(mean, min_val=1e-5, max_val=1. - 1e-5)
        x = mean + torch.exp(log_scale) * (torch.log(u) - torch.log(1 - u))
        sample.append(x)

    # (batch_size, img_channels, height, width)
    sample = torch.stack(sample, dim=1)

    return sample


def recon(output, target):
    """
    recon loss
    :param output: Tensor. shape = (B, C, H, W)
    :param target: Tensor. shape = (B, C, H, W)
    :return:
    """

    # Treat q(x|z) as Norm distribution
    # loss = F.mse_loss(output, target)

    # Treat q(x|z) as Bernoulli distribution.
    loss = F.binary_cross_entropy(output, target)
    return loss


def kl(mu, log_var):
    """
    kl loss with standard norm distribute
    :param mu:
    :param log_var:
    :return:
    """
    loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=[1, 2, 3])
    return torch.mean(loss, dim=0)


def kl_2(delta_mu, delta_log_var, mu, log_var):
    var = torch.exp(log_var)
    delta_var = torch.exp(delta_log_var)

    loss = -0.5 * torch.sum(1 + delta_log_var - delta_mu ** 2 / var - delta_var, dim=[1, 2, 3])
    return torch.mean(loss, dim=0)


def log_sum_exp(x):
    """

    :param x: Tensor. shape = (batch_size, num_mixtures, height, width)
    :return:
    """

    m2 = torch.max(x, dim=1, keepdim=True)[0]
    m = m2.unsqueeze(1)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=1))


def discretized_mix_logistic_loss(y_hat: torch.Tensor, y: torch.Tensor, num_classes=256, log_scale_min=-7.0):
    """Discretized mix of logistic distributions loss.

    Note that it is assumed that input is scaled to [-1, 1]



    :param y_hat: Tensor. shape=(batch_size, 3 * num_mixtures * img_channels, height, width), predict output.
    :param y: Tensor. shape=(batch_size, img_channels, height, width), Target.
    :return: Tensor loss
    """

    # unpack parameters, [batch_size, num_mixtures * img_channels, height, width] x 3
    logit_probs, means, log_scales = y_hat.chunk(3, dim=1)
    log_scales = torch.clamp_max(log_scales, log_scale_min)

    num_mixtures = y_hat.size(1) // y.size(1) // 3

    B, C, H, W = y.shape
    y = y.unsqueeze(1).repeat(1, num_mixtures, 1, 1, 1).permute(0, 2, 1, 3, 4).reshape(B, -1, H, W)

    centered_y = y - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    log_pdf_mid = min_in - log_scales - 2. * F.softplus(mid_in)

    log_probs = torch.where(y < -0.999, log_cdf_plus,
                            torch.where(y > 0.999, log_one_minus_cdf_min,
                                        torch.where(cdf_delta > 1e-5, torch.clamp_max(cdf_delta, 1e-12),
                                                    log_pdf_mid - np.log((num_classes - 1) / 2))))

    # (batch_size, num_mixtures * img_channels, height, width)
    log_probs = log_probs + F.softmax(log_probs, dim=1)

    log_probs = [log_sum_exp(log_prob) for log_prob in log_probs.chunk(y.size(1), dim=1)]
    log_probs = reduce(lambda a, b: a + b, log_probs)

    return -torch.sum(log_probs)


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self._seq = nn.Sequential(

            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=1),
            nn.BatchNorm2d(out_channel // 2), Swish(),
            nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel), Swish()
        )

    def forward(self, x):
        return self._seq(x)


class EncoderBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        modules = []
        for i in range(len(channels) - 1):
            modules.append(ConvBlock(channels[i], channels[i + 1]))

        self.modules_list = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x


class Encoder(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock([3, z_dim // 16, z_dim // 8]),  # (16, 16)
            EncoderBlock([z_dim // 8, z_dim // 4, z_dim // 2]),  # (4, 4)
            EncoderBlock([z_dim // 2, z_dim]),  # (2, 2)
        ])

        self.encoder_residual_blocks = nn.ModuleList([
            EncoderResidualBlock(z_dim // 8),
            EncoderResidualBlock(z_dim // 2),
            EncoderResidualBlock(z_dim),
        ])

        self.condition_x = nn.Sequential(
            Swish(),
            nn.Conv2d(z_dim, z_dim * 2, kernel_size=1)
        )

    def forward(self, x):
        xs = []
        last_x = x
        for e, r in zip(self.encoder_blocks, self.encoder_residual_blocks):
            x = r(e(x))
            last_x = x
            xs.append(x)

        mu, log_var = self.condition_x(last_x).chunk(2, dim=1)

        return mu, log_var, xs[:-1][::-1]


class UpsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self._seq = nn.Sequential(

            nn.ConvTranspose2d(in_channel,
                               out_channel,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel), Swish(),
        )

    def forward(self, x):
        return self._seq(x)


class DecoderBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        modules = []
        for i in range(len(channels) - 1):
            modules.append(UpsampleBlock(channels[i], channels[i + 1]))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x


class Decoder(nn.Module):

    def __init__(self, z_dim):
        super().__init__()

        # Input channels = z_channels * 2 = x_channels + z_channels
        # Output channels = z_channels
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock([z_dim * 2, z_dim // 2]),  # 2x upsample
            DecoderBlock([z_dim, z_dim // 4, z_dim // 8]),  # 4x upsample
            DecoderBlock([z_dim // 4, z_dim // 16, z_dim // 32])  # 4x uplsampe
        ])
        self.decoder_residual_blocks = nn.ModuleList([
            DecoderResidualBlock(z_dim // 2, n_group=4),
            DecoderResidualBlock(z_dim // 8, n_group=2),
            DecoderResidualBlock(z_dim // 32, n_group=1)
        ])

        # p(z_l | z_(l-1))
        self.condition_z = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(z_dim // 2),
                Swish(),
                nn.Conv2d(z_dim // 2, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim // 8),
                Swish(),
                nn.Conv2d(z_dim // 8, z_dim // 4, kernel_size=1)
            )
        ])

        # p(z_l | x, z_(l-1))
        self.condition_xz = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(z_dim),
                nn.Conv2d(z_dim, z_dim // 2, kernel_size=1),
                Swish(),
                nn.Conv2d(z_dim // 2, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim // 4),
                nn.Conv2d(z_dim // 4, z_dim // 8, kernel_size=1),
                Swish(),
                nn.Conv2d(z_dim // 8, z_dim // 4, kernel_size=1)
            )
        ])

        self.recon = nn.Sequential(
            ResidualBlock(z_dim // 32),
            nn.Conv2d(z_dim // 32, 3, kernel_size=1),
        )

        self.zs = []

    def forward(self, z, xs=None, mode="random", freeze_level=-1):
        """

        :param z: shape. = (B, z_dim, map_h, map_w)
        :return:
        """

        B, D, map_h, map_w = z.shape

        # The init h (hidden state), can be replace with learned param, but it didn't work much
        decoder_out = torch.zeros(B, D, map_h, map_w, device=z.device, dtype=z.dtype)

        kl_losses = []
        if freeze_level != -1 and len(self.zs) == 0 :
            self.zs.append(z)

        for i in range(len(self.decoder_residual_blocks)):

            z_sample = torch.cat([decoder_out, z], dim=1)
            decoder_out = self.decoder_residual_blocks[i](self.decoder_blocks[i](z_sample))

            if i == len(self.decoder_residual_blocks) - 1:
                break

            mu, log_var = self.condition_z[i](decoder_out).chunk(2, dim=1)

            if xs is not None:
                delta_mu, delta_log_var = self.condition_xz[i](torch.cat([xs[i], decoder_out], dim=1)) \
                    .chunk(2, dim=1)
                kl_losses.append(kl_2(delta_mu, delta_log_var, mu, log_var))
                mu = mu + delta_mu
                log_var = log_var + delta_log_var

            if mode == "fix" and i < freeze_level:
                if len(self.zs) < freeze_level + 1:
                    z = reparameterize(mu, 0)
                    self.zs.append(z)
                else:
                    z = self.zs[i + 1]
            elif mode == "fix":
                z = reparameterize(mu, 0 if i == 0 else torch.exp(0.5 * log_var))
            else:
                z = reparameterize(mu, torch.exp(0.5 * log_var))

            map_h *= 2 ** (len(self.decoder_blocks[i].channels) - 1)
            map_w *= 2 ** (len(self.decoder_blocks[i].channels) - 1)

        x_hat = torch.sigmoid(self.recon(decoder_out))

        return x_hat, kl_losses


class ImageFolderDataset(Dataset):

    def __init__(self, image_dir, img_dim):
        self.img_paths = glob(os.path.join(image_dir, "*.jpg"))
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, c = image.shape
        if h > w:
            top_h = int((h - w) / 2)
            image = image[top_h:top_h + w]
        else:
            left_w = int((w - h) / 2)
            image = image[:, left_w:left_w + h]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        image = image / 255.

        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

    def __len__(self):
        return len(self.img_paths)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim), Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            SELayer(dim))

    def forward(self, x):
        return x + 0.1 * self._seq(x)


class EncoderResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.seq = nn.Sequential(

            nn.Conv2d(dim, dim, kernel_size=5, padding=2),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim), Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            SELayer(dim))

    def forward(self, x):
        return x + 0.1 * self.seq(x)


class DecoderResidualBlock(nn.Module):

    def __init__(self, dim, n_group):
        super().__init__()

        self._seq = nn.Sequential(
            nn.Conv2d(dim, n_group * dim, kernel_size=1),
            nn.BatchNorm2d(n_group * dim), Swish(),
            nn.Conv2d(n_group * dim, n_group * dim, kernel_size=5, padding=2, groups=n_group),
            nn.BatchNorm2d(n_group * dim), Swish(),
            nn.Conv2d(n_group * dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            SELayer(dim))

    def forward(self, x):
        return x + 0.1 * self._seq(x)


class NVAE(nn.Module):

    def __init__(self, z_dim, img_dim):
        super().__init__()

        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

        self.adaptive_loss = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims=1, float_dtype=np.float32, device="cpu")

    def forward(self, x):
        """

        :param x: Tensor. shape = (B, C, H, W)
        :return:
        """

        mu, log_var, xs = self.encoder(x)

        # (B, D_Z)
        z = reparameterize(mu, torch.exp(0.5 * log_var))

        decoder_output, losses = self.decoder(z, xs)

        # Treat p(x|z) as discretized_mix_logistic distribution cost so much, this is an alternative way
        # witch combine multi distribution.
        recon_loss = torch.mean(self.adaptive_loss.lossfun(
            torch.mean(F.binary_cross_entropy(decoder_output, x, reduction='none'), dim=[1, 2, 3])[:, None]))

        kl_loss = kl(mu, log_var)

        return decoder_output, recon_loss, [kl_loss] + losses


class WarmupKLLoss:

    def __init__(self, init_weights, steps,
                 M_N=0.005,
                 eta_M_N=1e-5,
                 M_N_decay_step=3000):
        """
        预热KL损失，先对各级别的KL损失进行预热，预热完成后，对M_N的值进行衰减,所有衰减策略采用线性衰减
        :param init_weights: 各级别 KL 损失的初始权重
        :param steps: 各级别KL损失从初始权重增加到1所需的步数
        :param M_N: 初始M_N值
        :param eta_M_N: 最小M_N值
        :param M_N_decay_step: 从初始M_N值到最小M_N值所需的衰减步数
        """
        self.init_weights = init_weights
        self.M_N = M_N
        self.eta_M_N = eta_M_N
        self.M_N_decay_step = M_N_decay_step
        self.speeds = [(1. - w) / s for w, s in zip(init_weights, steps)]
        self.steps = np.cumsum(steps)
        self.stage = 0
        self._ready_start_step = 0
        self._ready_for_M_N = False
        self._M_N_decay_speed = (self.M_N - self.eta_M_N) / self.M_N_decay_step

    def _get_stage(self, step):
        while True:

            if self.stage > len(self.steps) - 1:
                break

            if step <= self.steps[self.stage]:
                return self.stage
            else:
                self.stage += 1

        return self.stage

    def get_loss(self, step, losses):
        loss = 0.
        stage = self._get_stage(step)

        for i, l in enumerate(losses):
            # Update weights
            if i == stage:
                speed = self.speeds[stage]
                t = step if stage == 0 else step - self.steps[stage - 1]
                w = min(self.init_weights[i] + speed * t, 1.)
            elif i < stage:
                w = 1.
            else:
                w = self.init_weights[i]

            # 如果所有级别的KL损失的预热都已完成
            if self._ready_for_M_N == False and i == len(losses) - 1 and w == 1.:
                # 准备M_N的衰减
                self._ready_for_M_N = True
                self._ready_start_step = step
            l = losses[i] * w
            loss += l

        if self._ready_for_M_N:
            M_N = max(self.M_N - self._M_N_decay_speed *
                      (step - self._ready_start_step), self.eta_M_N)
        else:
            M_N = self.M_N

        return M_N * loss



epochs = 500
batch_size = 64 #128
pretrained_weights = None


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = NVAE(z_dim=512, img_dim=(64, 64))


train_ds = ImageFolderDataset(DATA_DIR, img_dim=64)
train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)#, num_workers=8)


# apply Spectral Normalization
model.apply(add_sn)
model.to(device)

if pretrained_weights:
    model.load_state_dict(torch.load(pretrained_weights, map_location=device), strict=False)

warmup_kl = WarmupKLLoss(init_weights=[1., 1. / 2, 1. / 8],
                            steps=[4500, 3000, 1500],
                            M_N=batch_size / len(train_ds),
                            eta_M_N=5e-6,
                            M_N_decay_step=36000)
print('M_N=', warmup_kl.M_N, 'ETA_M_N=', warmup_kl.eta_M_N)

optimizer = torch.optim.Adamax(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-4)

step = 0
for epoch in range(epochs):
    model.train()

    n_true = 0
    total_size = 0
    total_loss = 0
    for i, image in enumerate(train_dataloader):
        optimizer.zero_grad()

        image = image.to(device)
        image_recon, recon_loss, kl_losses = model(image)
        kl_loss = warmup_kl.get_loss(step, kl_losses)
        loss = recon_loss + kl_loss
        print("\r---- [Epoch %d/%d, Step %d/%d] loss: %.6f----" % (epoch, epochs, i, len(train_dataloader), loss.item()))

        loss.backward()
        optimizer.step()
        step += 1

        if step != 0 and step % 100 == 0:
            with torch.no_grad():
                z = torch.randn((1, 512, 2, 2)).to(device)
                gen_img, _ = model.decoder(z)
                gen_img = gen_img.permute(0, 2, 3, 1)
                gen_img = gen_img[0].cpu().numpy() * 255
                gen_img = gen_img.astype(np.uint8)
                plt.savefig(os.path.join(out_dir, f"ae_ckpt_%d_%.6f.png" % (epoch, total_loss)))
    scheduler.step()
    torch.save(model.state_dict(), os.path.join(out_dir, f"ae_ckpt_%d_%.6f.pth" % (epoch, loss.item())))
    model.eval()
    with torch.no_grad():
        z = torch.randn((1, 512, 2, 2)).to(device)
        gen_img, _ = model.decoder(z)
        gen_img = gen_img.permute(0, 2, 3, 1)
        gen_img = gen_img[0].cpu().numpy() * 255
        gen_img = gen_img.astype(np.uint8)
        plt.savefig(os.path.join(out_dir, f"ae_ckpt_%d_%.6f.png" % (epoch, total_loss)))











# class FourierMapping(nn.Module):
#
#     def __init__(self, dims, seed):
#         super().__init__()
#         np.random.seed(seed)
#         B = np.random.randn(*dims) * 10
#         np.random.seed(None)
#         self.B = torch.tensor(B, dtype=torch.float32)
#
#     def forward(self, x):
#         x = input_mapping(x, self.B.to(x.device))
#         return x
