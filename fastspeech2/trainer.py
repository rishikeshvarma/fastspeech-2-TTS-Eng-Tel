"""Training loop, adapted from the reference train.py.

[modified]
  * single GPU (no nn.DataParallel), optional bf16 autocast for newer GPUs
  * losses are also written to CSV files so the loss-curve plots can be regenerated
    without TensorBoard; the training CSV holds the mean over each log window
  * DataLoader workers and a fixed random seed are configurable
  * the best validation checkpoint is kept as best.pth.tar in addition to periodic ones
"""

import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset
from evaluate import evaluate

LOSS_NAMES = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]


def _append_csv(path, step, values):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["step"] + LOSS_NAMES)
        writer.writerow([step] + ["{:.6f}".format(v) for v in values])


def save_checkpoint(model, optimizer, path):
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer._optimizer.state_dict()},
        path,
    )


def main(restore_step, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = train_config.get("seed", 1234)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    num_workers = train_config.get("num_workers", 0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    # Prepare model
    model, optimizer = get_model(restore_step, configs, device, train=True)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)
    use_amp = train_config.get("use_amp", False) and device.type == "cuda"

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)
    train_csv = os.path.join(train_config["path"]["log_path"], "train_loss.csv")
    val_csv = os.path.join(train_config["path"]["log_path"], "val_loss.csv")
    ckpt_path = train_config["path"]["ckpt_path"]

    # Training
    step = restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    best_val = float("inf")
    window = []
    t0 = time.time()

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    output = model(*(batch[2:]))
                output = tuple(o.float() if torch.is_floating_point(o) else o for o in output)

                # Cal Loss
                losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                window.append([l.item() for l in losses])

                if step % log_step == 0:
                    losses = list(np.mean(window, axis=0))
                    window = []
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )
                    message3 = ", {:.1f} steps/s".format(log_step / (time.time() - t0))
                    t0 = time.time()

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")
                    _append_csv(train_csv, step, losses)

                    outer_bar.write(message1 + message2 + message3)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message, val_losses = evaluate(
                        model, step, configs, val_logger, vocoder, return_losses=True
                    )
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    _append_csv(val_csv, step, val_losses)
                    outer_bar.write(message)
                    if val_losses[0] < best_val:
                        best_val = val_losses[0]
                        save_checkpoint(model, optimizer, os.path.join(ckpt_path, "best.pth.tar"))

                    model.train()

                if step % save_step == 0 or step == total_step:
                    save_checkpoint(
                        model, optimizer, os.path.join(ckpt_path, "{}.pth.tar".format(step))
                    )

                if step == total_step:
                    return
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1
