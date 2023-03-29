
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.utils import make_grid, save_image

from scipy import stats
from skimage.io import imread
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import imageio

from functools import reduce
from collections import defaultdict
from timeit import default_timer
import os, re, json, math, random, ast, shutil, abc, sys
import subprocess, hashlib, zipfile, glob, tarfile
import configparser
import logging
import argparse


TEST_LOSSES_FILE = "test_losses.log"
METRICS_FILENAME = "metrics.log"
METRIC_HELPERS_FILE = "metric_helpers.pth"
TRAIN_LOSSES_LOGFILE = "train_losses.log"
MODELS = ["Burgess"]
LOSSES = ["VAE", "betaH", "betaB", "factor", "btcvae"]
RECON_DIST = ["bernoulli", "laplace", "gaussian"]
CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]
TRAIN_FILE = "train_losses.log"
DECIMAL_POINTS = 3
GIF_FILE = "training.gif"
PLOT_NAMES = dict(generate_samples="samples.png",
                  data_samples="data_samples.png",
                  reconstruct="reconstruct.png",
                  traversals="traversals.png",
                  reconstruct_traverse="reconstruct_traverse.png",
                  gif_traversals="posterior_traversals.gif",)
DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1
DATASETS_DICT = {"mnist": "MNIST",
                 "fashion": "FashionMNIST",
                 "dsprites": "DSprites",
                 "celeba": "CelebA",
                 "chairs": "Chairs"}
DATASETS = list(DATASETS_DICT.keys())
FPS_GIF = 12
MODEL_FILENAME = "model.pt"
META_FILENAME = "specs.json"



# TO-DO: clean n_data and device
def get_loss_f(loss_name, **kwargs_parse):
    """Return the correct loss function given the argparse arguments."""
    kwargs_all = dict(rec_dist=kwargs_parse["rec_dist"],
                      steps_anneal=kwargs_parse["reg_anneal"])
    if loss_name == "betaH":
        return BetaHLoss(beta=kwargs_parse["betaH_B"], **kwargs_all)
    elif loss_name == "VAE":
        return BetaHLoss(beta=1, **kwargs_all)
    elif loss_name == "betaB":
        return BetaBLoss(C_init=kwargs_parse["betaB_initC"],
                         C_fin=kwargs_parse["betaB_finC"],
                         gamma=kwargs_parse["betaB_G"],
                         **kwargs_all)
    elif loss_name == "factor":
        return FactorKLoss(kwargs_parse["device"],
                           gamma=kwargs_parse["factor_G"],
                           disc_kwargs=dict(latent_dim=kwargs_parse["latent_dim"]),
                           optim_kwargs=dict(lr=kwargs_parse["lr_disc"], betas=(0.5, 0.9)),
                           **kwargs_all)
    elif loss_name == "btcvae":
        return BtcvaeLoss(kwargs_parse["n_data"],
                          alpha=kwargs_parse["btcvae_A"],
                          beta=kwargs_parse["btcvae_B"],
                          gamma=kwargs_parse["btcvae_G"],
                          **kwargs_all)
    else:
        assert loss_name not in LOSSES
        raise ValueError("Uknown loss : {}".format(loss_name))


class BaseLoss(abc.ABC):
    """
    Base class for losses.

    Parameters
    ----------
    record_loss_every: int, optional
        Every how many steps to recorsd the loss.

    rec_dist: {"bernoulli", "gaussian", "laplace"}, optional
        Reconstruction distribution istribution of the likelihood on the each pixel.
        Implicitely defines the reconstruction loss. Bernoulli corresponds to a
        binary cross entropy (bse), Gaussian corresponds to MSE, Laplace
        corresponds to L1.

    steps_anneal: nool, optional
        Number of annealing steps where gradually adding the regularisation.
    """

    def __init__(self, record_loss_every=50, rec_dist="bernoulli", steps_anneal=0):
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every
        self.rec_dist = rec_dist
        self.steps_anneal = steps_anneal

    @abc.abstractmethod
    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Shape : (batch_size, n_chan,
            height, width).

        recon_data : torch.Tensor
            Reconstructed data. Shape : (batch_size, n_chan, height, width).

        latent_dist : tuple of torch.tensor
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, log_var) each of shape : (batch_size, latent_dim).

        is_train : bool
            Whether currently in train mode.

        storer : dict
            Dictionary in which to store important variables for vizualisation.

        kwargs:
            Loss specific arguments
        """

    def _pre_call(self, is_train, storer):
        if is_train:
            self.n_train_steps += 1

        if not is_train or self.n_train_steps % self.record_loss_every == 1:
            storer = storer
        else:
            storer = None

        return storer


class BetaHLoss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """

    def __init__(self, beta=4, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer)
        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if is_train else 1)
        loss = rec_loss + anneal_reg * (self.beta * kl_loss)

        if storer is not None:
            storer['loss'].append(loss.item())

        return loss


class BetaBLoss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    C_init : float, optional
        Starting annealed capacity C.

    C_fin : float, optional
        Final annealed capacity C.

    gamma : float, optional
        Weight of the KL divergence term.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Burgess, Christopher P., et al. "Understanding disentangling in
        $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
    """

    def __init__(self, C_init=0., C_fin=20., gamma=100., **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.C_init = C_init
        self.C_fin = C_fin

    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer)

        C = (linear_annealing(self.C_init, self.C_fin, self.n_train_steps, self.steps_anneal)
             if is_train else self.C_fin)

        loss = rec_loss + self.gamma * (kl_loss - C).abs()

        if storer is not None:
            storer['loss'].append(loss.item())

        return loss


class FactorKLoss(BaseLoss):
    """
    Compute the Factor-VAE loss as per Algorithm 2 of [1]

    Parameters
    ----------
    device : torch.device

    gamma : float, optional
        Weight of the TC loss term. `gamma` in the paper.

    discriminator : disvae.discriminator.Discriminator

    optimizer_d : torch.optim

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).
    """

    def __init__(self, device,
                 gamma=10.,
                 disc_kwargs={},
                 optim_kwargs=dict(lr=5e-5, betas=(0.5, 0.9)),
                 **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.device = device
        self.discriminator = Discriminator(**disc_kwargs).to(self.device)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), **optim_kwargs)

    def __call__(self, *args, **kwargs):
        raise ValueError("Use `call_optimize` to also train the discriminator")

    def call_optimize(self, data, model, optimizer, storer):
        storer = self._pre_call(model.training, storer)

        # factor-vae split data into two batches. In the paper they sample 2 batches
        batch_size = data.size(dim=0)
        half_batch_size = batch_size // 2
        data = data.split(half_batch_size)
        data1 = data[0]
        data2 = data[1]

        # Factor VAE Loss
        recon_batch, latent_dist, latent_sample1 = model(data1)
        rec_loss = _reconstruction_loss(data1, recon_batch,
                                        storer=storer,
                                        distribution=self.rec_dist)

        kl_loss = _kl_normal_loss(*latent_dist, storer)

        d_z = self.discriminator(latent_sample1)
        # We want log(p_true/p_false). If not using logisitc regression but softmax
        # then p_true = exp(logit_true) / Z; p_false = exp(logit_false) / Z
        # so log(p_true/p_false) = logit_true - logit_false
        tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()
        # with sigmoid (not good results) should be `tc_loss = (2 * d_z.flatten()).mean()`

        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if model.training else 1)
        vae_loss = rec_loss + kl_loss + anneal_reg * self.gamma * tc_loss

        if storer is not None:
            storer['loss'].append(vae_loss.item())
            storer['tc_loss'].append(tc_loss.item())

        if not model.training:
            # don't backprop if evaluating
            return vae_loss

        # Compute VAE gradients
        optimizer.zero_grad()
        vae_loss.backward(retain_graph=True)

        # Discriminator Loss
        # Get second sample of latent distribution
        latent_sample2 = model.sample_latent(data2)
        z_perm = _permute_dims(latent_sample2).detach()
        d_z_perm = self.discriminator(z_perm)

        # Calculate total correlation loss
        # for cross entropy the target is the index => need to be long and says
        # that it's first output for d_z and second for perm
        ones = torch.ones(half_batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros_like(ones)
        d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_perm, ones))
        # with sigmoid would be :
        # d_tc_loss = 0.5 * (self.bce(d_z.flatten(), ones) + self.bce(d_z_perm.flatten(), 1 - ones))

        # TO-DO: check ifshould also anneals discriminator if not becomes too good ???
        #d_tc_loss = anneal_reg * d_tc_loss

        # Compute discriminator gradients
        self.optimizer_d.zero_grad()
        d_tc_loss.backward()

        # Update at the end (since pytorch 1.5. complains if update before)
        optimizer.step()
        self.optimizer_d.step()

        if storer is not None:
            storer['discrim_loss'].append(d_tc_loss.item())

        return vae_loss


class BtcvaeLoss(BaseLoss):
    """
    Compute the decomposed KL loss with either minibatch weighted sampling or
    minibatch stratified sampling according to [1]

    Parameters
    ----------
    n_data: int
        Number of data in the training set

    alpha : float
        Weight of the mutual information term.

    beta : float
        Weight of the total correlation term.

    gamma : float
        Weight of the dimension-wise KL term.

    is_mss : bool
        Whether to use minibatch stratified sampling instead of minibatch
        weighted sampling.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(self, n_data, alpha=1., beta=6., gamma=1., is_mss=True, **kwargs):
        super().__init__(**kwargs)
        self.n_data = n_data
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.is_mss = is_mss  # minibatch stratified sampling

    def __call__(self, data, recon_batch, latent_dist, is_train, storer,
                 latent_sample=None):
        storer = self._pre_call(is_train, storer)
        batch_size, latent_dim = latent_sample.shape

        rec_loss = _reconstruction_loss(data, recon_batch,
                                        storer=storer,
                                        distribution=self.rec_dist)
        log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample,
                                                                             latent_dist,
                                                                             self.n_data,
                                                                             is_mss=self.is_mss)
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qzi - log_pz).mean()

        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if is_train else 1)

        # total loss
        loss = rec_loss + (self.alpha * mi_loss +
                           self.beta * tc_loss +
                           anneal_reg * self.gamma * dw_kl_loss)

        if storer is not None:
            storer['loss'].append(loss.item())
            storer['mi_loss'].append(mi_loss.item())
            storer['tc_loss'].append(tc_loss.item())
            storer['dw_kl_loss'].append(dw_kl_loss.item())
            # computing this for storing and comparaison purposes
            _ = _kl_normal_loss(*latent_dist, storer)

        return loss


def _reconstruction_loss(data, recon_data, distribution="bernoulli", storer=None):
    """
    Calculates the per image reconstruction loss for a batch of data. I.e. negative
    log likelihood.

    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Shape : (batch_size, n_chan,
        height, width).

    recon_data : torch.Tensor
        Reconstructed data. Shape : (batch_size, n_chan, height, width).

    distribution : {"bernoulli", "gaussian", "laplace"}
        Distribution of the likelihood on the each pixel. Implicitely defines the
        loss Bernoulli corresponds to a binary cross entropy (bse) loss and is the
        most commonly used. It has the issue that it doesn't penalize the same
        way (0.1,0.2) and (0.4,0.5), which might not be optimal. Gaussian
        distribution corresponds to MSE, and is sometimes used, but hard to train
        ecause it ends up focusing only a few pixels that are very wrong. Laplace
        distribution corresponds to L1 solves partially the issue of MSE.

    storer : dict
        Dictionary in which to store important variables for vizualisation.

    Returns
    -------
    loss : torch.Tensor
        Per image cross entropy (i.e. normalized per batch but not pixel and
        channel)
    """
    batch_size, n_chan, height, width = recon_data.size()
    is_colored = n_chan == 3

    if distribution == "bernoulli":
        loss = F.binary_cross_entropy(recon_data, data, reduction="sum")
    elif distribution == "gaussian":
        # loss in [0,255] space but normalized by 255 to not be too big
        loss = F.mse_loss(recon_data * 255, data * 255, reduction="sum") / 255
    elif distribution == "laplace":
        # loss in [0,255] space but normalized by 255 to not be too big but
        # multiply by 255 and divide 255, is the same as not doing anything for L1
        loss = F.l1_loss(recon_data, data, reduction="sum")
        loss = loss * 3  # emperical value to give similar values than bernoulli => use same hyperparam
        loss = loss * (loss != 0)  # masking to avoid nan
    else:
        assert distribution not in RECON_DIST
        raise ValueError("Unkown distribution: {}".format(distribution))

    loss = loss / batch_size

    if storer is not None:
        storer['recon_loss'].append(loss.item())

    return loss


def _kl_normal_loss(mean, logvar, storer=None):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, latent_dim) where
        D is dimension of distribution.

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        latent_dim)

    storer : dict
        Dictionary in which to store important variables for vizualisation.
    """
    latent_dim = mean.size(1)
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()

    if storer is not None:
        storer['kl_loss'].append(total_kl.item())
        for i in range(latent_dim):
            storer['kl_loss_' + str(i)].append(latent_kl[i].item())

    return total_kl


def _permute_dims(latent_sample):
    """
    Implementation of Algorithm 1 in ref [1]. Randomly permutes the sample from
    q(z) (latent_dist) across the batch for each of the latent dimensions (mean
    and log_var).

    Parameters
    ----------
    latent_sample: torch.Tensor
        sample from the latent dimension using the reparameterisation trick
        shape : (batch_size, latent_dim).

    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).

    """
    perm = torch.zeros_like(latent_sample)
    batch_size, dim_z = perm.size()

    for z in range(dim_z):
        pi = torch.randperm(batch_size).to(latent_sample.device)
        perm[:, z] = latent_sample[pi, z]

    return perm


def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


# Batch TC specific
# TO-DO: test if mss is better!
def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx


def get_activation_name(activation):
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation

    mapper = {nn.LeakyReLU: "leaky_relu", nn.ReLU: "relu", nn.Tanh: "tanh",
              nn.Sigmoid: "sigmoid", nn.Softmax: "sigmoid"}
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k

    raise ValueError("Unkown given activation type : {}".format(activation))


def get_gain(activation):
    """Given an object of `torch.nn.modules.activation` or an activation name
    return the correct gain."""
    if activation is None:
        return 1

    activation_name = get_activation_name(activation)

    param = None if activation_name != "leaky_relu" else activation.negative_slope
    gain = nn.init.calculate_gain(activation_name, param)

    return gain


def linear_init(layer, activation="relu"):
    """Initialize a linear layer.
    Args:
        layer (nn.Linear): parameters to initialize.
        activation (`torch.nn.modules.activation` or str, optional) activation that
            will be used on the `layer`.
    """
    x = layer.weight

    if activation is None:
        return nn.init.xavier_uniform_(x)

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity='leaky_relu')
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity='relu')
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))


def weights_init(module):
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        # TO-DO: check litterature
        linear_init(module)
    elif isinstance(module, nn.Linear):
        linear_init(module)


def create_safe_directory(directory, logger=None):
    """Create a directory and archive the previous one if already existed."""
    if os.path.exists(directory):
        if logger is not None:
            warn = "Directory {} already exists. Archiving it to {}.zip"
            logger.warning(warn.format(directory, directory))
        shutil.make_archive(directory, 'zip', directory)
        shutil.rmtree(directory)
    os.makedirs(directory)


def set_seed(seed):
    """Set all random seeds."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        # if want pure determinism could uncomment below: but slower
        # torch.backends.cudnn.deterministic = True


def get_device(is_gpu=True):
    """Return the correct device"""
    return torch.device("cuda" if torch.cuda.is_available() and is_gpu
                        else "cpu")


def get_model_device(model):
    """Return the device on which a model is."""
    return next(model.parameters()).device


def get_n_param(model):
    """Return the number of parameters."""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nParams = sum([np.prod(p.size()) for p in model_parameters])
    return nParams


def update_namespace_(namespace, dictionnary):
    """Update an argparse namespace in_place."""
    vars(namespace).update(dictionnary)


def get_config_section(filenames, section):
    """Return a dictionnary of the section of `.ini` config files. Every value
    int the `.ini` will be litterally evaluated, such that `l=[1,"as"]` actually
    returns a list.
    """
    parser = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    parser.optionxform = str
    files = parser.read(filenames)
    if len(files) == 0:
        raise ValueError("Config files not found: {}".format(filenames))
    dict_session = dict(parser[section])
    dict_session = {k: ast.literal_eval(v) for k, v in dict_session.items()}
    return dict_session


def check_bounds(value, type=float, lb=-float("inf"), ub=float("inf"),
                 is_inclusive=True, name="value"):
    """Argparse bound checker"""
    value = type(value)
    is_in_bound = lb <= value <= ub if is_inclusive else lb < value < ub
    if not is_in_bound:
        raise argparse.ArgumentTypeError("{}={} outside of bounds ({},{})".format(name, value, lb, ub))
    return value


class FormatterNoDuplicate(argparse.ArgumentDefaultsHelpFormatter):
    """Formatter overriding `argparse.ArgumentDefaultsHelpFormatter` to show
    `-e, --epoch EPOCH` instead of `-e EPOCH, --epoch EPOCH`

    Note
    ----
    - code modified from cPython: https://github.com/python/cpython/blob/master/Lib/argparse.py
    """

    def _format_action_invocation(self, action):
        # no args given
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            metavar, = self._metavar_formatter(action, default)(1)
            return metavar
        else:
            parts = []
            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)
            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    # don't store the DEFAULT
                    parts.append('%s' % (option_string))
                # store DEFAULT for the last one
                parts[-1] += ' %s' % args_string
            return ', '.join(parts)


class Evaluator:
    """
    Class to handle training of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    loss_f: disvae.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, loss_f,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 is_progress_bar=True):

        self.device = device
        self.loss_f = loss_f
        self.model = model.to(self.device)
        self.logger = logger
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger.info("Testing Device: {}".format(self.device))

    def __call__(self, data_loader, is_metrics=False, is_losses=True):
        """Compute all test losses.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        is_metrics: bool, optional
            Whether to compute and store the disentangling metrics.

        is_losses: bool, optional
            Whether to compute and store the test losses.
        """
        start = default_timer()
        is_still_training = self.model.training
        self.model.eval()

        metric, losses = None, None
        if is_metrics:
            self.logger.info('Computing metrics...')
            metrics = self.compute_metrics(data_loader)
            self.logger.info('Losses: {}'.format(metrics))
            save_metadata(metrics, self.save_dir, filename=METRICS_FILENAME)

        if is_losses:
            self.logger.info('Computing losses...')
            losses = self.compute_losses(data_loader)
            self.logger.info('Losses: {}'.format(losses))
            save_metadata(losses, self.save_dir, filename=TEST_LOSSES_FILE)

        if is_still_training:
            self.model.train()

        self.logger.info('Finished evaluating after {:.1f} min.'.format((default_timer() - start) / 60))

        return metric, losses

    def compute_losses(self, dataloader):
        """Compute all test losses.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        storer = defaultdict(list)
        for data, _ in dataloader:
            data = data.to(self.device)

            try:
                recon_batch, latent_dist, latent_sample = self.model(data)
                _ = self.loss_f(data, recon_batch, latent_dist, self.model.training,
                                storer, latent_sample=latent_sample)
            except ValueError:
                # for losses that use multiple optimizers (e.g. Factor)
                _ = self.loss_f.call_optimize(data, self.model, None, storer)

            losses = {k: sum(v) / len(dataloader) for k, v in storer.items()}
            return losses

    def compute_metrics(self, dataloader):
        """Compute all the metrics.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        try:
            lat_sizes = dataloader.dataset.lat_sizes
            lat_names = dataloader.dataset.lat_names
        except AttributeError:
            raise ValueError("Dataset needs to have known true factors of variations to compute the metric. This does not seem to be the case for {}".format(type(dataloader.__dict__["dataset"]).__name__))

        self.logger.info("Computing the empirical distribution q(z|x).")
        samples_zCx, params_zCx = self._compute_q_zCx(dataloader)
        len_dataset, latent_dim = samples_zCx.shape

        self.logger.info("Estimating the marginal entropy.")
        # marginal entropy H(z_j)
        H_z = self._estimate_latent_entropies(samples_zCx, params_zCx)

        # conditional entropy H(z|v)
        samples_zCx = samples_zCx.view(*lat_sizes, latent_dim)
        params_zCx = tuple(p.view(*lat_sizes, latent_dim) for p in params_zCx)
        H_zCv = self._estimate_H_zCv(samples_zCx, params_zCx, lat_sizes, lat_names)

        H_z = H_z.cpu()
        H_zCv = H_zCv.cpu()

        # I[z_j;v_k] = E[log \sum_x q(z_j|x)p(x|v_k)] + H[z_j] = - H[z_j|v_k] + H[z_j]
        mut_info = - H_zCv + H_z
        sorted_mut_info = torch.sort(mut_info, dim=1, descending=True)[0].clamp(min=0)

        metric_helpers = {'marginal_entropies': H_z, 'cond_entropies': H_zCv}
        mig = self._mutual_information_gap(sorted_mut_info, lat_sizes, storer=metric_helpers)
        aam = self._axis_aligned_metric(sorted_mut_info, storer=metric_helpers)

        metrics = {'MIG': mig.item(), 'AAM': aam.item()}
        torch.save(metric_helpers, os.path.join(self.save_dir, METRIC_HELPERS_FILE))

        return metrics

    def _mutual_information_gap(self, sorted_mut_info, lat_sizes, storer=None):
        """Compute the mutual information gap as in [1].

        References
        ----------
           [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
           autoencoders." Advances in Neural Information Processing Systems. 2018.
        """
        # difference between the largest and second largest mutual info
        delta_mut_info = sorted_mut_info[:, 0] - sorted_mut_info[:, 1]
        # NOTE: currently only works if balanced dataset for every factor of variation
        # then H(v_k) = - |V_k|/|V_k| log(1/|V_k|) = log(|V_k|)
        H_v = torch.from_numpy(lat_sizes).float().log()
        mig_k = delta_mut_info / H_v
        mig = mig_k.mean()  # mean over factor of variations

        if storer is not None:
            storer["mig_k"] = mig_k
            storer["mig"] = mig

        return mig

    def _axis_aligned_metric(self, sorted_mut_info, storer=None):
        """Compute the proposed axis aligned metrics."""
        numerator = (sorted_mut_info[:, 0] - sorted_mut_info[:, 1:].sum(dim=1)).clamp(min=0)
        aam_k = numerator / sorted_mut_info[:, 0]
        aam_k[torch.isnan(aam_k)] = 0
        aam = aam_k.mean()  # mean over factor of variations

        if storer is not None:
            storer["aam_k"] = aam_k
            storer["aam"] = aam

        return aam

    def _compute_q_zCx(self, dataloader):
        """Compute the empiricall disitribution of q(z|x).

        Parameter
        ---------
        dataloader: torch.utils.data.DataLoader
            Batch data iterator.

        Return
        ------
        samples_zCx: torch.tensor
            Tensor of shape (len_dataset, latent_dim) containing a sample of
            q(z|x) for every x in the dataset.

        params_zCX: tuple of torch.Tensor
            Sufficient statistics q(z|x) for each training example. E.g. for
            gaussian (mean, log_var) each of shape : (len_dataset, latent_dim).
        """
        len_dataset = len(dataloader.dataset)
        latent_dim = self.model.latent_dim
        n_suff_stat = 2

        q_zCx = torch.zeros(len_dataset, latent_dim, n_suff_stat, device=self.device)

        n = 0
        with torch.no_grad():
            for x, label in dataloader:
                batch_size = x.size(0)
                idcs = slice(n, n + batch_size)
                q_zCx[idcs, :, 0], q_zCx[idcs, :, 1] = self.model.encoder(x.to(self.device))
                n += batch_size

        params_zCX = q_zCx.unbind(-1)
        samples_zCx = self.model.reparameterize(*params_zCX)

        return samples_zCx, params_zCX

    def _estimate_latent_entropies(self, samples_zCx, params_zCX,
                                   n_samples=10000):
        r"""Estimate :math:`H(z_j) = E_{q(z_j)} [-log q(z_j)] = E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]`
        using the emperical distribution of :math:`p(x)`.

        Note
        ----
        - the expectation over the emperical distributio is: :math:`q(z) = 1/N sum_{n=1}^N q(z|x_n)`.
        - we assume that q(z|x) is factorial i.e. :math:`q(z|x) = \prod_j q(z_j|x)`.
        - computes numerically stable NLL: :math:`- log q(z) = log N - logsumexp_n=1^N log q(z|x_n)`.

        Parameters
        ----------
        samples_zCx: torch.tensor
            Tensor of shape (len_dataset, latent_dim) containing a sample of
            q(z|x) for every x in the dataset.

        params_zCX: tuple of torch.Tensor
            Sufficient statistics q(z|x) for each training example. E.g. for
            gaussian (mean, log_var) each of shape : (len_dataset, latent_dim).

        n_samples: int, optional
            Number of samples to use to estimate the entropies.

        Return
        ------
        H_z: torch.Tensor
            Tensor of shape (latent_dim) containing the marginal entropies H(z_j)
        """
        len_dataset, latent_dim = samples_zCx.shape
        device = samples_zCx.device
        H_z = torch.zeros(latent_dim, device=device)

        # sample from p(x)
        samples_x = torch.randperm(len_dataset, device=device)[:n_samples]
        # sample from p(z|x)
        samples_zCx = samples_zCx.index_select(0, samples_x).view(latent_dim, n_samples)

        mini_batch_size = 10
        samples_zCx = samples_zCx.expand(len_dataset, latent_dim, n_samples)
        mean = params_zCX[0].unsqueeze(-1).expand(len_dataset, latent_dim, n_samples)
        log_var = params_zCX[1].unsqueeze(-1).expand(len_dataset, latent_dim, n_samples)
        log_N = math.log(len_dataset)
        for k in range(0, n_samples, mini_batch_size):
            # log q(z_j|x) for n_samples
            idcs = slice(k, k + mini_batch_size)
            log_q_zCx = log_density_gaussian(samples_zCx[..., idcs],
                                                mean[..., idcs],
                                                log_var[..., idcs])
            # numerically stable log q(z_j) for n_samples:
            # log q(z_j) = -log N + logsumexp_{n=1}^N log q(z_j|x_n)
            # As we don't know q(z) we appoximate it with the monte carlo
            # expectation of q(z_j|x_n) over x. => fix a single z and look at
            # proba for every x to generate it. n_samples is not used here !
            log_q_z = -log_N + torch.logsumexp(log_q_zCx, dim=0, keepdim=False)
            # H(z_j) = E_{z_j}[- log q(z_j)]
            # mean over n_samples (i.e. dimesnion 1 because already summed over 0).
            H_z += (-log_q_z).sum(1)


        H_z /= n_samples

        return H_z

    def _estimate_H_zCv(self, samples_zCx, params_zCx, lat_sizes, lat_names):
        """Estimate conditional entropies :math:`H[z|v]`."""
        latent_dim = samples_zCx.size(-1)
        len_dataset = reduce((lambda x, y: x * y), lat_sizes)
        H_zCv = torch.zeros(len(lat_sizes), latent_dim, device=self.device)
        for i_fac_var, (lat_size, lat_name) in enumerate(zip(lat_sizes, lat_names)):
            idcs = [slice(None)] * len(lat_sizes)
            for i in range(lat_size):
                self.logger.info("Estimating conditional entropies for the {}th value of {}.".format(i, lat_name))
                idcs[i_fac_var] = i
                # samples from q(z,x|v)
                samples_zxCv = samples_zCx[idcs].contiguous().view(len_dataset // lat_size,
                                                                   latent_dim)
                params_zxCv = tuple(p[idcs].contiguous().view(len_dataset // lat_size, latent_dim)
                                    for p in params_zCx)

                H_zCv[i_fac_var] += self._estimate_latent_entropies(samples_zxCv, params_zxCv
                                                                    ) / lat_size
        return H_zCv


# ALL encoders should be called Enccoder<Model>
def get_encoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Encoder{}".format(model_type))


class EncoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Encoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(EncoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar


class Discriminator(nn.Module):
    def __init__(self,
                 neg_slope=0.2,
                 latent_dim=10,
                 hidden_units=1000):
        """Discriminator proposed in [1].

        Parameters
        ----------
        neg_slope: float
            Hyperparameter for the Leaky ReLu

        latent_dim : int
            Dimensionality of latent variables.

        hidden_units: int
            Number of hidden units in the MLP

        Model Architecture
        ------------
        - 6 layer multi-layer perceptron, each with 1000 hidden units
        - Leaky ReLu activations
        - Output 2 logits

        References:
            [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
            arXiv preprint arXiv:1802.05983 (2018).

        """
        super(Discriminator, self).__init__()

        # Activation parameters
        self.neg_slope = neg_slope
        self.leaky_relu = nn.LeakyReLU(self.neg_slope, True)

        # Layer parameters
        self.z_dim = latent_dim
        self.hidden_units = hidden_units
        # theoretically 1 with sigmoid but gives bad results => use 2 and softmax
        out_units = 2

        # Fully connected layers
        self.lin1 = nn.Linear(self.z_dim, hidden_units)
        self.lin2 = nn.Linear(hidden_units, hidden_units)
        self.lin3 = nn.Linear(hidden_units, hidden_units)
        self.lin4 = nn.Linear(hidden_units, hidden_units)
        self.lin5 = nn.Linear(hidden_units, hidden_units)
        self.lin6 = nn.Linear(hidden_units, out_units)

        self.reset_parameters()

    def forward(self, z):

        # Fully connected layers with leaky ReLu activations
        z = self.leaky_relu(self.lin1(z))
        z = self.leaky_relu(self.lin2(z))
        z = self.leaky_relu(self.lin3(z))
        z = self.leaky_relu(self.lin4(z))
        z = self.leaky_relu(self.lin5(z))
        z = self.lin6(z)

        return z

    def reset_parameters(self):
        self.apply(weights_init)


# ALL decoders should be called Decoder<Model>
def get_decoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Decoder{}".format(model_type))


class DecoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(DecoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x


def save_model(model, directory, metadata=None, filename=MODEL_FILENAME):
    """
    Save a model and corresponding metadata.

    Parameters
    ----------
    model : nn.Module
        Model.

    directory : str
        Path to the directory where to save the data.

    metadata : dict
        Metadata to save.
    """
    device = next(model.parameters()).device
    model.cpu()

    if metadata is None:
        # save the minimum required for loading
        metadata = dict(img_size=model.img_size, latent_dim=model.latent_dim,
                        model_type=model.model_type)

    save_metadata(metadata, directory)

    path_to_model = os.path.join(directory, filename)
    torch.save(model.state_dict(), path_to_model)

    model.to(device)  # restore device


def load_metadata(directory, filename=META_FILENAME):
    """Load the metadata of a training directory.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.
    """
    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata) as metadata_file:
        metadata = json.load(metadata_file)

    return metadata


def save_metadata(metadata, directory, filename=META_FILENAME, **kwargs):
    """Load the metadata of a training directory.

    Parameters
    ----------
    metadata:
        Object to save

    directory: string
        Path to folder where to save model. For example './experiments/mnist'.

    kwargs:
        Additional arguments to `json.dump`
    """
    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata, 'w') as f:
        json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)


def load_model(directory, is_gpu=True, filename=MODEL_FILENAME):
    """Load a trained model.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.

    is_gpu : bool
        Whether to load on GPU is available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and is_gpu
                          else "cpu")

    path_to_model = os.path.join(directory, MODEL_FILENAME)

    metadata = load_metadata(directory)
    img_size = metadata["img_size"]
    latent_dim = metadata["latent_dim"]
    model_type = metadata["model_type"]

    path_to_model = os.path.join(directory, filename)
    model = _get_model(model_type, img_size, latent_dim, device, path_to_model)
    return model


def load_checkpoints(directory, is_gpu=True):
    """Load all chechpointed models.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.

    is_gpu : bool
        Whether to load on GPU .
    """
    checkpoints = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            results = re.search(r'.*?-([0-9].*?).pt', filename)
            if results is not None:
                epoch_idx = int(results.group(1))
                model = load_model(root, is_gpu=is_gpu, filename=filename)
                checkpoints.append((epoch_idx, model))

    return checkpoints


def _get_model(model_type, img_size, latent_dim, device, path_to_model):
    """ Load a single model.

    Parameters
    ----------
    model_type : str
        The name of the model to load. For example Burgess.
    img_size : tuple
        Tuple of the number of pixels in the image width and height.
        For example (32, 32) or (64, 64).
    latent_dim : int
        The number of latent dimensions in the bottleneck.

    device : str
        Either 'cuda' or 'cpu'
    path_to_device : str
        Full path to the saved model on the device.
    """
    model = init_specific_model(model_type, img_size, latent_dim).to(device)
    # works with state_dict to make it independent of the file structure
    model.load_state_dict(torch.load(path_to_model), strict=False)
    model.eval()

    return model


def numpy_serialize(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def save_np_arrays(arrays, directory, filename):
    """Save dictionary of arrays in json file."""
    save_metadata(arrays, directory, filename=filename, default=numpy_serialize)


def load_np_arrays(directory, filename):
    """Load dictionary of arrays from json file."""
    arrays = load_metadata(directory, filename=filename)
    return {k: np.array(v) for k, v in arrays.items()}


def matrix_log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian for all combination of bacth pairs of
    `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.

    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).

    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).

    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).

    batch_size: int
        number of training images in the batch
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)


def log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian.

    Parameters
    ----------
    x: torch.Tensor or np.ndarray or float
        Value at which to compute the density.

    mu: torch.Tensor or np.ndarray or float
        Mean.

    logvar: torch.Tensor or np.ndarray or float
        Log variance.
    """
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    return log_density


def log_importance_weight_matrix(batch_size, dataset_size):
    """
    Calculates a log importance weight matrix

    Parameters
    ----------
    batch_size: int
        number of training images in the batch

    dataset_size: int
    number of training images in the dataset
    """
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()


def init_specific_model(model_type, img_size, latent_dim):
    """Return an instance of a VAE with encoder and decoder from `model_type`."""
    model_type = model_type.lower().capitalize()
    if model_type not in MODELS:
        err = "Unkown model_type={}. Possible values: {}"
        raise ValueError(err.format(model_type, MODELS))

    encoder = get_encoder(model_type)
    decoder = get_decoder(model_type)
    model = VAE(img_size, encoder, decoder, latent_dim)
    model.model_type = model_type  # store to help reloading
    return model


class VAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, latent_dim):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(VAE, self).__init__()

        if list(img_size[1:]) not in [[32, 32], [64, 64]]:
            raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(img_size, self.latent_dim)
        self.decoder = decoder(img_size, self.latent_dim)

        self.reset_parameters()

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        return reconstruct, latent_dist, latent_sample

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample


def get_samples(dataset, num_samples, idcs=[]):
    """ Generate a number of samples from the dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset.

    num_samples : int, optional
        The number of samples to load from the dataset

    idcs : list of ints, optional
        List of indices to of images to put at the begning of the samples.
    """
    data_loader = get_dataloaders(dataset,
                                  batch_size=1,
                                  shuffle=idcs is None)

    idcs += random.sample(range(len(data_loader.dataset)), num_samples - len(idcs))
    samples = torch.stack([data_loader.dataset[i][0] for i in idcs], dim=0)
    print("Selected idcs: {}".format(idcs))

    return samples


def sort_list_by_other(to_sort, other, reverse=True):
    """Sort a list by an other."""
    return [el for _, el in sorted(zip(other, to_sort), reverse=reverse)]


# TO-DO: clean
def read_loss_from_file(log_file_path, loss_to_fetch):
    """ Read the average KL per latent dimension at the final stage of training from the log file.
        Parameters
        ----------
        log_file_path : str
            Full path and file name for the log file. For example 'experiments/custom/losses.log'.

        loss_to_fetch : str
            The loss type to search for in the log file and return. This must be in the exact form as stored.
    """
    EPOCH = "Epoch"
    LOSS = "Loss"

    logs = pd.read_csv(log_file_path)
    df_last_epoch_loss = logs[logs.loc[:, EPOCH] == logs.loc[:, EPOCH].max()]
    df_last_epoch_loss = df_last_epoch_loss.loc[df_last_epoch_loss.loc[:, LOSS].str.startswith(loss_to_fetch), :]
    df_last_epoch_loss.loc[:, LOSS] = df_last_epoch_loss.loc[:, LOSS].str.replace(loss_to_fetch, "").astype(int)
    df_last_epoch_loss = df_last_epoch_loss.sort_values(LOSS).loc[:, "Value"]
    return list(df_last_epoch_loss)


def add_labels(input_image, labels):
    """Adds labels next to rows of an image.

    Parameters
    ----------
    input_image : image
        The image to which to add the labels
    labels : list
        The list of labels to plot
    """
    new_width = input_image.width + 100
    new_size = (new_width, input_image.height)
    new_img = Image.new("RGB", new_size, color='white')
    new_img.paste(input_image, (0, 0))
    draw = ImageDraw.Draw(new_img)

    for i, s in enumerate(labels):
        draw.text(xy=(new_width - 100 + 0.005,
                      int((i / len(labels) + 1 / (2 * len(labels))) * input_image.height)),
                  text=s,
                  fill=(0, 0, 0))

    return new_img


def make_grid_img(tensor, **kwargs):
    """Converts a tensor to a grid of images that can be read by imageio.

    Notes
    -----
    * from in https://github.com/pytorch/vision/blob/master/torchvision/utils.py

    Parameters
    ----------
    tensor (torch.Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
        or a list of images all of the same size.

    kwargs:
        Additional arguments to `make_grid_img`.
    """
    grid = make_grid(tensor, **kwargs)
    img_grid = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    img_grid = img_grid.to('cpu', torch.uint8).numpy()
    return img_grid


def get_image_list(image_file_name_list):
    image_list = []
    for file_name in image_file_name_list:
        image_list.append(Image.open(file_name))
    return image_list


def arr_im_convert(arr, convert="RGBA"):
    """Convert an image array."""
    return np.asarray(Image.fromarray(arr).convert(convert))


def plot_grid_gifs(filename, grid_files, pad_size=7, pad_values=255):
    """Take a grid of gif files and merge them in order with padding."""
    grid_gifs = [[imageio.mimread(f) for f in row] for row in grid_files]
    n_per_gif = len(grid_gifs[0][0])

    # convert all to RGBA which is the most general => can merge any image
    imgs = [concatenate_pad([concatenate_pad([arr_im_convert(gif[i], convert="RGBA")
                                              for gif in row], pad_size, pad_values, axis=1)
                             for row in grid_gifs], pad_size, pad_values, axis=0)
            for i in range(n_per_gif)]

    imageio.mimsave(filename, imgs, fps=FPS_GIF)


def concatenate_pad(arrays, pad_size, pad_values, axis=0):
    """Concatenate lsit of array with padding inbetween."""
    pad = np.ones_like(arrays[0]).take(indices=range(pad_size), axis=axis) * pad_values

    new_arrays = [pad]
    for arr in arrays:
        new_arrays += [arr, pad]
    new_arrays += [pad]
    return np.concatenate(new_arrays, axis=axis)


class Visualizer():
    def __init__(self, model, dataset, model_dir,
                 save_images=True,
                 loss_of_interest=None,
                 display_loss_per_dim=False,
                 max_traversal=0.475,  # corresponds to ~2 for standard normal
                 upsample_factor=1):
        """
        Visualizer is used to generate images of samples, reconstructions,
        latent traversals and so on of the trained model.

        Parameters
        ----------
        model : disvae.vae.VAE

        dataset : str
            Name of the dataset.

        model_dir : str
            The directory that the model is saved to and where the images will
            be stored.

        save_images : bool, optional
            Whether to save images or return a tensor.

        loss_of_interest : str, optional
            The loss type (as saved in the log file) to order the latent dimensions by and display.

        display_loss_per_dim : bool, optional
            if the loss should be included as text next to the corresponding latent dimension images.

        max_traversal: float, optional
            The maximum displacement induced by a latent traversal. Symmetrical
            traversals are assumed. If `m>=0.5` then uses absolute value traversal,
            if `m<0.5` uses a percentage of the distribution (quantile).
            E.g. for the prior the distribution is a standard normal so `m=0.45` c
            orresponds to an absolute value of `1.645` because `2m=90%%` of a
            standard normal is between `-1.645` and `1.645`. Note in the case
            of the posterior, the distribution is not standard normal anymore.

        upsample_factor : floar, optional
            Scale factor to upsample the size of the tensor
        """
        self.model = model
        self.device = next(self.model.parameters()).device
        self.latent_dim = self.model.latent_dim
        self.max_traversal = max_traversal
        self.save_images = save_images
        self.model_dir = model_dir
        self.dataset = dataset
        self.upsample_factor = upsample_factor
        if loss_of_interest is not None:
            self.losses = read_loss_from_file(os.path.join(self.model_dir, TRAIN_FILE),
                                              loss_of_interest)

    def _get_traversal_range(self, mean=0, std=1):
        """Return the corresponding traversal range in absolute terms."""
        max_traversal = self.max_traversal

        if max_traversal < 0.5:
            max_traversal = (1 - 2 * max_traversal) / 2  # from 0.45 to 0.05
            max_traversal = stats.norm.ppf(max_traversal, loc=mean, scale=std)  # from 0.05 to -1.645

        # symmetrical traversals
        return (-1 * max_traversal, max_traversal)

    def _traverse_line(self, idx, n_samples, data=None):
        """Return a (size, latent_size) latent sample, corresponding to a traversal
        of a latent variable indicated by idx.

        Parameters
        ----------
        idx : int
            Index of continuous dimension to traverse. If the continuous latent
            vector is 10 dimensional and idx = 7, then the 7th dimension
            will be traversed while all others are fixed.

        n_samples : int
            Number of samples to generate.

        data : torch.Tensor or None, optional
            Data to use for computing the posterior. Shape (N, C, H, W). If
            `None` then use the mean of the prior (all zeros) for all other dimensions.
        """
        if data is None:
            # mean of prior for other dimensions
            samples = torch.zeros(n_samples, self.latent_dim)
            traversals = torch.linspace(*self._get_traversal_range(), steps=n_samples)

        else:
            if data.size(0) > 1:
                raise ValueError("Every value should be sampled from the same posterior, but {} datapoints given.".format(data.size(0)))

            with torch.no_grad():
                post_mean, post_logvar = self.model.encoder(data.to(self.device))
                samples = self.model.reparameterize(post_mean, post_logvar)
                samples = samples.cpu().repeat(n_samples, 1)
                post_mean_idx = post_mean.cpu()[0, idx]
                post_std_idx = torch.exp(post_logvar / 2).cpu()[0, idx]

            # travers from the gaussian of the posterior in case quantile
            traversals = torch.linspace(*self._get_traversal_range(mean=post_mean_idx,
                                                                   std=post_std_idx),
                                        steps=n_samples)

        for i in range(n_samples):
            samples[i, idx] = traversals[i]

        return samples

    def _save_or_return(self, to_plot, size, filename, is_force_return=False):
        """Create plot and save or return it."""
        to_plot = F.interpolate(to_plot, scale_factor=self.upsample_factor)

        if size[0] * size[1] != to_plot.shape[0]:
            raise ValueError("Wrong size {} for datashape {}".format(size, to_plot.shape))

        # `nrow` is number of images PER row => number of col
        kwargs = dict(nrow=size[1], pad_value=(1 - get_background(self.dataset)))
        if self.save_images and not is_force_return:
            filename = os.path.join(self.model_dir, filename)
            save_image(to_plot, filename, **kwargs)
        else:
            return make_grid_img(to_plot, **kwargs)

    def _decode_latents(self, latent_samples):
        """Decodes latent samples into images.

        Parameters
        ----------
        latent_samples : torch.autograd.Variable
            Samples from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        latent_samples = latent_samples.to(self.device)
        return self.model.decoder(latent_samples).cpu()

    def generate_samples(self, size=(8, 8)):
        """Plot generated samples from the prior and decoding.

        Parameters
        ----------
        size : tuple of ints, optional
            Size of the final grid.
        """
        prior_samples = torch.randn(size[0] * size[1], self.latent_dim)
        generated = self._decode_latents(prior_samples)
        return self._save_or_return(generated.data, size, PLOT_NAMES["generate_samples"])

    def data_samples(self, data, size=(8, 8)):
        """Plot samples from the dataset

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        size : tuple of ints, optional
            Size of the final grid.
        """
        data = data[:size[0] * size[1], ...]
        return self._save_or_return(data, size, PLOT_NAMES["data_samples"])

    def reconstruct(self, data, size=(8, 8), is_original=True, is_force_return=False):
        """Generate reconstructions of data through the model.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        size : tuple of ints, optional
            Size of grid on which reconstructions will be plotted. The number
            of rows should be even when `is_original`, so that upper
            half contains true data and bottom half contains reconstructions.contains

        is_original : bool, optional
            Whether to exclude the original plots.

        is_force_return : bool, optional
            Force returning instead of saving the image.
        """
        if is_original:
            if size[0] % 2 != 0:
                raise ValueError("Should be even number of rows when showing originals not {}".format(size[0]))
            n_samples = size[0] // 2 * size[1]
        else:
            n_samples = size[0] * size[1]

        with torch.no_grad():
            originals = data.to(self.device)[:n_samples, ...]
            recs, _, _ = self.model(originals)

        originals = originals.cpu()
        recs = recs.view(-1, *self.model.img_size).cpu()

        to_plot = torch.cat([originals, recs]) if is_original else recs
        return self._save_or_return(to_plot, size, PLOT_NAMES["reconstruct"],
                                    is_force_return=is_force_return)

    def traversals(self,
                   data=None,
                   is_reorder_latents=False,
                   n_per_latent=8,
                   n_latents=None,
                   is_force_return=False):
        """Plot traverse through all latent dimensions (prior or posterior) one
        by one and plots a grid of images where each row corresponds to a latent
        traversal of one latent dimension.

        Parameters
        ----------
        data : bool, optional
            Data to use for computing the latent posterior. If `None` traverses
            the prior.

        n_per_latent : int, optional
            The number of points to include in the traversal of a latent dimension.
            I.e. number of columns.

        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.

        is_reorder_latents : bool, optional
            If the latent dimensions should be reordered or not

        is_force_return : bool, optional
            Force returning instead of saving the image.
        """
        n_latents = n_latents if n_latents is not None else self.model.latent_dim
        latent_samples = [self._traverse_line(dim, n_per_latent, data=data)
                          for dim in range(self.latent_dim)]
        decoded_traversal = self._decode_latents(torch.cat(latent_samples, dim=0))

        if is_reorder_latents:
            n_images, *other_shape = decoded_traversal.size()
            n_rows = n_images // n_per_latent
            decoded_traversal = decoded_traversal.reshape(n_rows, n_per_latent, *other_shape)
            decoded_traversal = sort_list_by_other(decoded_traversal, self.losses)
            decoded_traversal = torch.stack(decoded_traversal, dim=0)
            decoded_traversal = decoded_traversal.reshape(n_images, *other_shape)

        decoded_traversal = decoded_traversal[range(n_per_latent * n_latents), ...]

        size = (n_latents, n_per_latent)
        sampling_type = "prior" if data is None else "posterior"
        filename = "{}_{}".format(sampling_type, PLOT_NAMES["traversals"])

        return self._save_or_return(decoded_traversal.data, size, filename,
                                    is_force_return=is_force_return)

    def reconstruct_traverse(self, data,
                             is_posterior=True,
                             n_per_latent=8,
                             n_latents=None,
                             is_show_text=False):
        """
        Creates a figure whith first row for original images, second are
        reconstructions, rest are traversals (prior or posterior) of the latent
        dimensions.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        n_per_latent : int, optional
            The number of points to include in the traversal of a latent dimension.
            I.e. number of columns.

        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.

        is_posterior : bool, optional
            Whether to sample from the posterior.

        is_show_text : bool, optional
            Whether the KL values next to the traversal rows.
        """
        n_latents = n_latents if n_latents is not None else self.model.latent_dim

        reconstructions = self.reconstruct(data[:2 * n_per_latent, ...],
                                           size=(2, n_per_latent),
                                           is_force_return=True)
        traversals = self.traversals(data=data[0:1, ...] if is_posterior else None,
                                     is_reorder_latents=True,
                                     n_per_latent=n_per_latent,
                                     n_latents=n_latents,
                                     is_force_return=True)

        concatenated = np.concatenate((reconstructions, traversals), axis=0)
        concatenated = Image.fromarray(concatenated)

        if is_show_text:
            losses = sorted(self.losses, reverse=True)[:n_latents]
            labels = ['orig', 'recon'] + ["KL={:.4f}".format(l) for l in losses]
            concatenated = add_labels(concatenated, labels)

        filename = os.path.join(self.model_dir, PLOT_NAMES["reconstruct_traverse"])
        concatenated.save(filename)

    def gif_traversals(self, data, n_latents=None, n_per_gif=15):
        """Generates a grid of gifs of latent posterior traversals where the rows
        are the latent dimensions and the columns are random images.

        Parameters
        ----------
        data : bool
            Data to use for computing the latent posteriors. The number of datapoint
            (batchsize) will determine the number of columns of the grid.

        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.

        n_per_gif : int, optional
            Number of images per gif (number of traversals)
        """
        n_images, _, _, width_col = data.shape
        width_col = int(width_col * self.upsample_factor)
        all_cols = [[] for c in range(n_per_gif)]
        for i in range(n_images):
            grid = self.traversals(data=data[i:i + 1, ...], is_reorder_latents=True,
                                   n_per_latent=n_per_gif, n_latents=n_latents,
                                   is_force_return=True)

            height, width, c = grid.shape
            padding_width = (width - width_col * n_per_gif) // (n_per_gif + 1)

            # split the grids into a list of column images (and removes padding)
            for j in range(n_per_gif):
                all_cols[j].append(grid[:, [(j + 1) * padding_width + j * width_col + i
                                            for i in range(width_col)], :])

        pad_values = (1 - get_background(self.dataset)) * 255
        all_cols = [concatenate_pad(cols, pad_size=2, pad_values=pad_values, axis=1)
                    for cols in all_cols]

        filename = os.path.join(self.model_dir, PLOT_NAMES["gif_traversals"])
        imageio.mimsave(filename, all_cols, fps=FPS_GIF)


class GifTraversalsTraining:
    """Creates a Gif of traversals by generating an image at every training epoch.

    Parameters
    ----------
    model : disvae.vae.VAE

    dataset : str
        Name of the dataset.

    model_dir : str
        The directory that the model is saved to and where the images will
        be stored.

    is_reorder_latents : bool, optional
        If the latent dimensions should be reordered or not

    n_per_latent : int, optional
        The number of points to include in the traversal of a latent dimension.
        I.e. number of columns.

    n_latents : int, optional
        The number of latent dimensions to display. I.e. number of rows. If `None`
        uses all latents.

    kwargs:
        Additional arguments to `Visualizer`
    """

    def __init__(self, model, dataset, model_dir,
                 is_reorder_latents=False,
                 n_per_latent=10,
                 n_latents=None,
                 **kwargs):
        self.save_filename = os.path.join(model_dir, GIF_FILE)
        self.visualizer = Visualizer(model, dataset, model_dir,
                                     save_images=False, **kwargs)

        self.images = []
        self.is_reorder_latents = is_reorder_latents
        self.n_per_latent = n_per_latent
        self.n_latents = n_latents if n_latents is not None else model.latent_dim

    def __call__(self):
        """Generate the next gif image. Should be called after each epoch."""
        cached_training = self.visualizer.model.training
        self.visualizer.model.eval()
        img_grid = self.visualizer.traversals(data=None,  # GIF from prior
                                              is_reorder_latents=self.is_reorder_latents,
                                              n_per_latent=self.n_per_latent,
                                              n_latents=self.n_latents)
        self.images.append(img_grid)
        if cached_training:
            self.visualizer.model.train()

    def save_reset(self):
        """Saves the GIF and resets the list of images. Call at the end of training."""
        imageio.mimsave(self.save_filename, self.images, fps=FPS_GIF)
        self.images = []


def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def get_img_size(dataset):
    """Return the correct image size."""
    return get_dataset(dataset).img_size


def get_background(dataset):
    """Return the image background color."""
    return get_dataset(dataset).background_color


def get_dataloaders(dataset, root=None, shuffle=True, pin_memory=True,
                    batch_size=128, logger=logging.getLogger(__name__), **kwargs):
    """A generic data loader

    Parameters
    ----------
    dataset : {"mnist", "fashion", "dsprites", "celeba", "chairs"}
        Name of the dataset to load

    root : str
        Path to the dataset root. If `None` uses the default one.

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    Dataset = get_dataset(dataset)
    dataset = Dataset(logger=logger) if root is None else Dataset(root=root, logger=logger)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      **kwargs)


class DisentangledDataset(Dataset, abc.ABC):
    """Base Class for disentangled VAE datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], logger=logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        pass


class DSprites(DisentangledDataset):
    """DSprites Dataset from [1].

    Disentanglement test Sprites dataset.Procedurally generated 2D shapes, from 6
    disentangled latent factors. This dataset uses 6 latents, controlling the color,
    shape, scale, rotation and position of a sprite. All possible variations of
    the latents are present. Ordering along dimension 1 is fixed and can be mapped
    back to the exact latent values that generated that image. Pixel outputs are
    different. No noise added.

    Notes
    -----
    - Link : https://github.com/deepmind/dsprites-dataset/
    - hard coded metadata because issue with python 3 loading of python 2

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick,
        M., ... & Lerchner, A. (2017). beta-vae: Learning basic visual concepts
        with a constrained variational framework. In International Conference
        on Learning Representations.

    """
    urls = {"train": "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"}
    files = {"train": "dsprite_train.npz"}
    lat_names = ('shape', 'scale', 'orientation', 'posX', 'posY')
    lat_sizes = np.array([3, 6, 40, 32, 32])
    img_size = (1, 64, 64)
    background_color = COLOUR_BLACK
    lat_values = {'posX': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                    0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                    0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                    0.5483871, 0.58064516, 0.61290323, 0.64516129,
                                    0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774, 0.90322581,
                                    0.93548387, 0.96774194, 1.]),
                  'posY': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                    0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                    0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                    0.5483871, 0.58064516, 0.61290323, 0.64516129,
                                    0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774, 0.90322581,
                                    0.93548387, 0.96774194, 1.]),
                  'scale': np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
                  'orientation': np.array([0., 0.16110732, 0.32221463, 0.48332195,
                                           0.64442926, 0.80553658, 0.96664389, 1.12775121,
                                           1.28885852, 1.44996584, 1.61107316, 1.77218047,
                                           1.93328779, 2.0943951, 2.25550242, 2.41660973,
                                           2.57771705, 2.73882436, 2.89993168, 3.061039,
                                           3.22214631, 3.38325363, 3.54436094, 3.70546826,
                                           3.86657557, 4.02768289, 4.1887902, 4.34989752,
                                           4.51100484, 4.67211215, 4.83321947, 4.99432678,
                                           5.1554341, 5.31654141, 5.47764873, 5.63875604,
                                           5.79986336, 5.96097068, 6.12207799, 6.28318531]),
                  'shape': np.array([1., 2., 3.]),
                  'color': np.array([1.])}

    def __init__(self, root=os.path.join(DIR, '../data/dsprites/'), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        dataset_zip = np.load(self.train_data)
        self.imgs = dataset_zip['imgs']
        self.lat_values = dataset_zip['latents_values']

    def download(self):
        """Download the dataset."""
        os.makedirs(self.root)
        subprocess.check_call(["curl", "-L", type(self).urls["train"],
                               "--output", self.train_data])

    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        # stored image have binary and shape (H x W) so multiply by 255 to get pixel
        # values + add dimension
        sample = np.expand_dims(self.imgs[idx] * 255, axis=-1)

        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        sample = self.transforms(sample)

        lat_value = self.lat_values[idx]
        return sample, lat_value


class CelebA(DisentangledDataset):
    """CelebA Dataset from [1].

    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset
    with more than 200K celebrity images, each with 40 attribute annotations.
    The images in this dataset cover large pose variations and background clutter.
    CelebA has large diversities, large quantities, and rich annotations, including
    10,177 number of identities, and 202,599 number of face images.

    Notes
    -----
    - Link : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face
        attributes in the wild. In Proceedings of the IEEE international conference
        on computer vision (pp. 3730-3738).

    """
    urls = {"train": "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"}
    files = {"train": "img_align_celeba"}
    img_size = (3, 64, 64)
    background_color = COLOUR_WHITE

    def __init__(self, root=os.path.join(DIR, '../data/celeba'), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        self.imgs = glob.glob(self.train_data + '/*')

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.root, 'celeba.zip')
        os.makedirs(self.root)
        subprocess.check_call(["curl", "-L", type(self).urls["train"],
                               "--output", save_path])

        hash_code = '00d2c5bc6d35e252742224ab0c1e8fcb'
        assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
            '{} file is corrupted.  Remove the file and try again.'.format(save_path)

        with zipfile.ZipFile(save_path) as zf:
            self.logger.info("Extracting CelebA ...")
            zf.extractall(self.root)

        os.remove(save_path)

        self.logger.info("Resizing CelebA ...")
        preprocess(self.train_data, size=type(self).img_size[1:])

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        placeholder :
            Placeholder value as their are no targets.
        """
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = imread(img_path)

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)

        # no label so return 0 (note that can't return None because)
        # dataloaders requires so
        return img, 0


class Chairs(datasets.ImageFolder):
    """Chairs Dataset from [1].

    Notes
    -----
    - Link : https://www.di.ens.fr/willow/research/seeing3Dchairs

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Aubry, M., Maturana, D., Efros, A. A., Russell, B. C., & Sivic, J. (2014).
        Seeing 3d chairs: exemplar part-based 2d-3d alignment using a large dataset
        of cad models. In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 3762-3769).

    """
    urls = {"train": "https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar"}
    files = {"train": "chairs_64"}
    img_size = (1, 64, 64)
    background_color = COLOUR_WHITE

    def __init__(self, root=os.path.join(DIR, '../data/chairs'),
                 logger=logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose([transforms.Grayscale(),
                                              transforms.ToTensor()])
        self.logger = logger

        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

        super().__init__(self.train_data, transform=self.transforms)

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.root, 'chairs.tar')
        os.makedirs(self.root)
        subprocess.check_call(["curl", type(self).urls["train"],
                               "--output", save_path])

        self.logger.info("Extracting Chairs ...")
        tar = tarfile.open(save_path)
        tar.extractall(self.root)
        tar.close()
        os.rename(os.path.join(self.root, 'rendered_chairs'), self.train_data)

        os.remove(save_path)

        self.logger.info("Preprocessing Chairs ...")
        preprocess(os.path.join(self.train_data, '*/*'),  # root/*/*/*.png structure
                   size=type(self).img_size[1:],
                   center_crop=(400, 400))


class MNIST(datasets.MNIST):
    """Mnist wrapper. Docs: `datasets.MNIST.`"""
    img_size = (1, 32, 32)
    background_color = COLOUR_BLACK

    def __init__(self, root=os.path.join(DIR, '../data/mnist'), **kwargs):
        super().__init__(root,
                         train=True,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(32),
                             transforms.ToTensor()
                         ]))


class FashionMNIST(datasets.FashionMNIST):
    """Fashion Mnist wrapper. Docs: `datasets.FashionMNIST.`"""
    img_size = (1, 32, 32)
    background_color = COLOUR_BLACK

    def __init__(self, root=os.path.join(DIR, '../data/fashionMnist'), **kwargs):
        super().__init__(root,
                         train=True,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(32),
                             transforms.ToTensor()
                         ]))


def preprocess(root, size=(64, 64), img_format='JPEG', center_crop=None):
    """Preprocess a folder of images.

    Parameters
    ----------
    root : string
        Root directory of all images.

    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.

    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.

    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, '*' + ext))

    for img_path in imgs:
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)


class Trainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    optimizer: torch.optim.Optimizer

    loss_f: disvae.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    gif_visualizer : viz.Visualizer, optional
        Gif Visualizer that should return samples at every epochs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, optimizer, loss_f,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 gif_visualizer=None,
                 is_progress_bar=True):

        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.losses_logger = LossesLogger(os.path.join(self.save_dir, TRAIN_LOSSES_LOGFILE))
        self.gif_visualizer = gif_visualizer
        self.logger.info("Training Device: {}".format(self.device))

    def __call__(self, data_loader,
                 epochs=10,
                 checkpoint_every=10):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epochs: int, optional
            Number of epochs to train the model for.

        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.
        """
        start = default_timer()
        self.model.train()
        for epoch in range(epochs):
            storer = defaultdict(list)
            mean_epoch_loss = self._train_epoch(data_loader, storer, epoch)
            self.logger.info('Epoch: {} Average loss per image: {:.2f}'.format(epoch + 1,
                                                                               mean_epoch_loss))
            self.losses_logger.log(epoch, storer)

            if self.gif_visualizer is not None:
                self.gif_visualizer()

            if epoch % checkpoint_every == 0:
                save_model(self.model, self.save_dir,
                           filename="model-{}.pt".format(epoch))

        if self.gif_visualizer is not None:
            self.gif_visualizer.save_reset()

        self.model.eval()

        delta_time = (default_timer() - start) / 60
        self.logger.info('Finished training after {:.1f} min.'.format(delta_time))

    def _train_epoch(self, data_loader, storer, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        storer: dict
            Dictionary in which to store important variables for vizualisation.

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
            Mean loss per image
        """
        epoch_loss = 0.
        kwargs = dict(desc="Epoch {}".format(epoch + 1), leave=False,
                      disable=not self.is_progress_bar)
        for _, (data, _) in enumerate(data_loader):
            iter_loss = self._train_iteration(data, storer)
            epoch_loss += iter_loss

        mean_epoch_loss = epoch_loss / len(data_loader)
        return mean_epoch_loss

    def _train_iteration(self, data, storer):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).

        storer: dict
            Dictionary in which to store important variables for vizualisation.
        """
        batch_size, channel, height, width = data.size()
        data = data.to(self.device)

        try:
            recon_batch, latent_dist, latent_sample = self.model(data)
            loss = self.loss_f(data, recon_batch, latent_dist, self.model.training,
                               storer, latent_sample=latent_sample)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        except ValueError:
            # for losses that use multiple optimizers (e.g. Factor)
            loss = self.loss_f.call_optimize(data, self.model, self.optimizer, storer)

        return loss.item()


class LossesLogger(object):
    """Class definition for objects to write data to log files in a
    form which is then easy to be plotted.
    """

    def __init__(self, file_path_name):
        """ Create a logger to store information for plotting. """
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)

        self.logger = logging.getLogger("losses_logger")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)

        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def log(self, epoch, losses_storer):
        """Write to the log file """
        for k, v in losses_storer.items():
            log_string = ",".join(str(item) for item in [epoch, k, mean(v)])
            self.logger.debug(log_string)


def mean(l):
    """Compute the mean of a list"""
    return sum(l) / len(l)


def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")

    description = "PyTorch implementation and evaluation of disentangled Variational AutoEncoders and metrics."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('name', type=str,
                         help="Name of the model for storing and loading purposes.")
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log_level'], choices=LOG_LEVELS)
    general.add_argument('--no-progress-bar', action='store_true',
                         default=default_config['no_progress_bar'],
                         help='Disables progress bar.')
    general.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when have one.')
    general.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                         help='Random seed. Can be `None` for stochastic behavior.')

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('--checkpoint-every',
                          type=int, default=default_config['checkpoint_every'],
                          help='Save a checkpoint of the trained model every n epoch.')
    training.add_argument('-d', '--dataset', help="Path to training data.",
                          default=default_config['dataset'], choices=DATASETS)
    training.add_argument('-x', '--experiment',
                          default=default_config['experiment'], choices=EXPERIMENTS,
                          help='Predefined experiments to run. If not `custom` this will overwrite some other arguments.')
    training.add_argument('-e', '--epochs', type=int,
                          default=default_config['epochs'],
                          help='Maximum number of epochs to run for.')
    training.add_argument('-b', '--batch-size', type=int,
                          default=default_config['batch_size'],
                          help='Batch size for training.')
    training.add_argument('--lr', type=float, default=default_config['lr'],
                          help='Learning rate.')

    # Model Options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=MODELS,
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim', type=int,
                       default=default_config['latent_dim'],
                       help='Dimension of the latent variable.')
    model.add_argument('-l', '--loss',
                       default=default_config['loss'], choices=LOSSES,
                       help="Type of VAE loss function to use.")
    model.add_argument('-r', '--rec-dist', default=default_config['rec_dist'],
                       choices=RECON_DIST,
                       help="Form of the likelihood ot use for each pixel.")
    model.add_argument('-a', '--reg-anneal', type=float,
                       default=default_config['reg_anneal'],
                       help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")

    # Loss Specific Options
    betaH = parser.add_argument_group('BetaH specific parameters')
    betaH.add_argument('--betaH-B', type=float,
                       default=default_config['betaH_B'],
                       help="Weight of the KL (beta in the paper).")

    betaB = parser.add_argument_group('BetaB specific parameters')
    betaB.add_argument('--betaB-initC', type=float,
                       default=default_config['betaB_initC'],
                       help="Starting annealed capacity.")
    betaB.add_argument('--betaB-finC', type=float,
                       default=default_config['betaB_finC'],
                       help="Final annealed capacity.")
    betaB.add_argument('--betaB-G', type=float,
                       default=default_config['betaB_G'],
                       help="Weight of the KL divergence term (gamma in the paper).")

    factor = parser.add_argument_group('factor VAE specific parameters')
    factor.add_argument('--factor-G', type=float,
                        default=default_config['factor_G'],
                        help="Weight of the TC term (gamma in the paper).")
    factor.add_argument('--lr-disc', type=float,
                        default=default_config['lr_disc'],
                        help='Learning rate of the discriminator.')

    btcvae = parser.add_argument_group('beta-tcvae specific parameters')
    btcvae.add_argument('--btcvae-A', type=float,
                        default=default_config['btcvae_A'],
                        help="Weight of the MI term (alpha in the paper).")
    btcvae.add_argument('--btcvae-G', type=float,
                        default=default_config['btcvae_G'],
                        help="Weight of the dim-wise KL term (gamma in the paper).")
    btcvae.add_argument('--btcvae-B', type=float,
                        default=default_config['btcvae_B'],
                        help="Weight of the TC term (beta in the paper).")

    # Learning options
    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--is-eval-only', action='store_true',
                            default=default_config['is_eval_only'],
                            help='Whether to only evaluate using precomputed model `name`.')
    evaluation.add_argument('--is-metrics', action='store_true',
                            default=default_config['is_metrics'],
                            help="Whether to compute the disentangled metrcics. Currently only possible with `dsprites` as it is the only dataset with known true factors of variations.")
    evaluation.add_argument('--no-test', action='store_true',
                            default=default_config['no_test'],
                            help="Whether not to compute the test losses.`")
    evaluation.add_argument('--eval-batchsize', type=int,
                            default=default_config['eval_batchsize'],
                            help='Batch size for evaluation.')

    args = parser.parse_args(args_to_parse)
    if args.experiment != 'custom':
        if args.experiment not in ADDITIONAL_EXP:
            # update all common sections first
            model, dataset = args.experiment.split("_")
            common_data = get_config_section([CONFIG_FILE], "Common_{}".format(dataset))
            update_namespace_(args, common_data)
            common_model = get_config_section([CONFIG_FILE], "Common_{}".format(model))
            update_namespace_(args, common_model)

        try:
            experiments_config = get_config_section([CONFIG_FILE], args.experiment)
            update_namespace_(args, experiments_config)
        except KeyError as e:
            if args.experiment in ADDITIONAL_EXP:
                raise e  # only reraise if didn't use common section

    return args




args = parse_arguments(sys.argv[1:])

formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                "%H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(args.log_level.upper())
stream = logging.StreamHandler()
stream.setLevel(args.log_level.upper())
stream.setFormatter(formatter)
logger.addHandler(stream)

set_seed(args.seed)
device = get_device(is_gpu=not args.no_cuda)
exp_dir = os.path.join(RES_DIR, args.name)
logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))

if not args.is_eval_only:

    create_safe_directory(exp_dir, logger=logger)

    if args.loss == "factor":
        logger.info("FactorVae needs 2 batches per iteration. To replicate this behavior while being consistent, we double the batch size and the the number of epochs.")
        args.batch_size *= 2
        args.epochs *= 2

    # PREPARES DATA
    train_loader = get_dataloaders(args.dataset,
                                    batch_size=args.batch_size,
                                    logger=logger)
    logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))

    # PREPARES MODEL
    args.img_size = get_img_size(args.dataset)  # stores for metadata
    model = init_specific_model(args.model_type, args.img_size, args.latent_dim)
    logger.info('Num parameters in model: {}'.format(get_n_param(model)))

    # TRAINS
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = model.to(device)  # make sure trainer and viz on same device
    gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir)
    loss_f = get_loss_f(args.loss,
                        n_data=len(train_loader.dataset),
                        device=device,
                        **vars(args))
    trainer = Trainer(model, optimizer, loss_f,
                        device=device,
                        logger=logger,
                        save_dir=exp_dir,
                        is_progress_bar=not args.no_progress_bar,
                        gif_visualizer=gif_visualizer)
    trainer(train_loader,
            epochs=args.epochs,
            checkpoint_every=args.checkpoint_every,)

    # SAVE MODEL AND EXPERIMENT INFORMATION
    save_model(trainer.model, exp_dir, metadata=vars(args))

if args.is_metrics or not args.no_test:
    model = load_model(exp_dir, is_gpu=not args.no_cuda)
    metadata = load_metadata(exp_dir)
    # TO-DO: currently uses train datatset
    test_loader = get_dataloaders(metadata["dataset"],
                                    batch_size=args.eval_batchsize,
                                    shuffle=False,
                                    logger=logger)
    loss_f = get_loss_f(args.loss,
                        n_data=len(test_loader.dataset),
                        device=device,
                        **vars(args))
    evaluator = Evaluator(model, loss_f,
                            device=device,
                            logger=logger,
                            save_dir=exp_dir,
                            is_progress_bar=not args.no_progress_bar)

    evaluator(test_loader, is_metrics=args.is_metrics, is_losses=not args.no_test)
