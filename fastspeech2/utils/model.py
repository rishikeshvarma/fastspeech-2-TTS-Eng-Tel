import os
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim

# [modified] resolve vocoder files relative to this package, not the working directory
HIFIGAN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hifigan")


def load_checkpoint(path, device):
    """torch.load with weights_only=True, also accepting the numpy scalars that
    checkpoints written before the optimizer fix carry in their optimizer state."""
    allowed = [np._core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location=device, weights_only=True)


def export_checkpoint(src, dst):
    """[new] Model weights only, stored as float16: a training checkpoint (418 MB with
    the Adam state) becomes ~70 MB, under GitHub's 100 MB file limit. The pitch/energy
    bin edges and the positional table stay float32; load_state_dict casts the rest
    back to the model's float32 parameters."""
    model = load_checkpoint(src, "cpu")["model"]
    model = {
        k: v.half() if v.is_floating_point() and "bins" not in k and "position_enc" not in k else v
        for k, v in model.items()
    }
    torch.save({"model": model, "source": os.path.basename(src)}, dst)


def get_model(restore_step, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(restore_step),
        )
        ckpt = load_checkpoint(ckpt_path, device)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, restore_step
        )
        if restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_(False)
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    # [modified] only HiFi-GAN is kept; MelGAN came from torch.hub at runtime
    assert name == "HiFi-GAN", "only the bundled HiFi-GAN vocoder is supported"
    with open(os.path.join(HIFIGAN_DIR, "config.json"), "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load(
        os.path.join(HIFIGAN_DIR, "generator_{}.pth.tar".format(speaker)),
        map_location=device,
    )
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
