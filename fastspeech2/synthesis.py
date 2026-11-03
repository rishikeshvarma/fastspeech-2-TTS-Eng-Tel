"""[new] Single-sentence synthesis plus the variance-adaptor plots required by the assignment.

The reference synthesize.py only saved a mel-spectrogram figure. Here the predictions of
the variance adaptor are returned per phoneme (duration in frames, F0 in Hz, energy) so
they can be plotted against the phone and word sequence and written out as JSON for the
report.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.io import wavfile

from text import text_to_sequence
from utils.model import vocoder_infer


def load_stats(preprocess_config):
    with open(
        os.path.join(preprocess_config["path"]["metadata_path"], "stats.json")
    ) as f:
        return json.load(f)


def synthesize_phones(
    model, vocoder, configs, words, speaker_id, device, controls=(1.0, 1.0, 1.0)
):
    """words: list of (word, [phones]); a word "sp" with phones ["sp"] is a pause."""
    preprocess_config, model_config, _ = configs
    phones = [p for _, ps in words for p in ps]
    word_index = [i for i, (_, ps) in enumerate(words) for _ in ps]
    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
    sequence = np.array(text_to_sequence("{" + " ".join(phones) + "}", cleaners))
    assert len(sequence) == len(phones), "some phones are not in the symbol table"

    speakers = torch.tensor([speaker_id], dtype=torch.long, device=device)
    texts = torch.from_numpy(sequence).long().unsqueeze(0).to(device)
    src_lens = torch.tensor([len(sequence)], device=device)
    p_control, e_control, d_control = controls

    with torch.no_grad():
        output = model(
            speakers,
            texts,
            src_lens,
            len(sequence),
            p_control=p_control,
            e_control=e_control,
            d_control=d_control,
        )

    mel_len = output[9][0].item()
    mel = output[1][0, :mel_len]  # post-net output, (T, n_mels)
    hop = preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav = vocoder_infer(
        mel.transpose(0, 1).unsqueeze(0),
        vocoder,
        model_config,
        preprocess_config,
        lengths=[mel_len * hop],
    )[0]

    stats = load_stats(preprocess_config)
    pitch_mean, pitch_std = stats["pitch"][2:]
    energy_mean, energy_std = stats["energy"][2:]
    n = len(sequence)
    return {
        "words": [w for w, _ in words],
        "phones": phones,
        "word_index": word_index,
        "duration_frames": output[5][0, :n].cpu().numpy().astype(int).tolist(),
        "log_duration": output[4][0, :n].float().cpu().numpy().tolist(),
        "pitch_hz": (output[2][0, :n].float().cpu().numpy() * pitch_std + pitch_mean).tolist(),
        "energy": (output[3][0, :n].float().cpu().numpy() * energy_std + energy_mean).tolist(),
        "mel": mel.float().cpu().numpy().T,
        "wav": wav,
        "sampling_rate": preprocess_config["preprocessing"]["audio"]["sampling_rate"],
        "hop_length": hop,
    }


def _word_spans(result):
    """(start_phone, end_phone, label) for every non-pause word."""
    spans = []
    idx = result["word_index"]
    for w in range(len(result["words"])):
        members = [i for i, k in enumerate(idx) if k == w]
        if members and result["words"][w] != "sp":
            spans.append((members[0], members[-1] + 1, result["words"][w]))
    return spans


def _phone_times(result):
    sec = result["hop_length"] / result["sampling_rate"]
    d = np.array(result["duration_frames"])
    starts = np.concatenate([[0], np.cumsum(d)[:-1]]) * sec
    return starts, d * sec


def plot_duration(result, title, path):
    phones = result["phones"]
    d = np.array(result["duration_frames"])
    ms = d * result["hop_length"] / result["sampling_rate"] * 1000
    fig, ax = plt.subplots(figsize=(min(max(8, 0.32 * len(phones)), 22), 4.2))
    colors = ["#9aa5b1" if p == "sp" else "#2a6fdb" for p in phones]
    ax.bar(np.arange(len(phones)), ms, color=colors, width=0.8)
    ax.set_xticks(np.arange(len(phones)))
    ax.set_xticklabels(phones, rotation=90, fontsize=8)
    ax.set_ylabel("predicted duration (ms)")
    ax.set_xlim(-0.6, len(phones) - 0.4)
    top = max(ms.max(), 1) * 1.18
    ax.set_ylim(0, top)
    for i, (s, e, w) in enumerate(_word_spans(result)):
        ax.axvline(s - 0.5, color="#cccccc", lw=0.8, zorder=0)
        ax.text((s + e - 1) / 2, top * 0.97, w, ha="center", va="top", fontsize=8,
                rotation=30 if len(w) > 8 else 0)
    total = ms.sum() / 1000
    ax.set_title("{}  |  total {:.2f} s, {} phones".format(title, total, len(phones)), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_contour(result, values, ylabel, color, title, path):
    starts, lens = _phone_times(result)
    mel = result["mel"]
    sec = result["hop_length"] / result["sampling_rate"]
    total = mel.shape[1] * sec
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(min(max(9, 0.9 * total * 4), 22), 5.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2]},
    )
    ax0.imshow(mel, origin="lower", aspect="auto", extent=[0, total, 0, mel.shape[0]], cmap="magma")
    ax0.set_ylabel("mel bin")
    ax0.set_title(title, fontsize=10)

    values = np.array(values)
    for s, l, v, p in zip(starts, lens, values, result["phones"]):
        if l <= 0:
            continue
        ax1.hlines(v, s, s + l, color="#9aa5b1" if p == "sp" else color, lw=2.2)
    voiced = [(s + l / 2, v) for s, l, v, p in zip(starts, lens, values, result["phones"]) if l > 0 and p != "sp"]
    if voiced:
        xs, ys = zip(*voiced)
        ax1.plot(xs, ys, color=color, lw=0.8, alpha=0.5)
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel("time (s)")
    lo, hi = values.min(), values.max()
    pad = (hi - lo) * 0.25 + 1e-3
    ax1.set_ylim(lo - pad, hi + pad)
    for s, e, w in _word_spans(result):
        x0 = starts[s]
        x1 = starts[e - 1] + lens[e - 1]
        for ax in (ax0, ax1):
            ax.axvline(x0, color="white" if ax is ax0 else "#cccccc", lw=0.6, alpha=0.7)
        ax1.text((x0 + x1) / 2, hi + pad * 0.8, w, ha="center", va="top", fontsize=8)
    ax1.set_xlim(0, total)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pitch(result, title, path):
    _plot_contour(result, result["pitch_hz"], "predicted F0 (Hz)", "#d6452b", title, path)


def plot_energy(result, title, path):
    _plot_contour(result, result["energy"], "predicted energy", "#7b2cbf", title, path)


def save_outputs(result, sid, text, results_dir):
    """Write <sid>.wav, the three plots and a JSON of per-phone predictions."""
    audio_dir = os.path.join(results_dir, "test_generated_audios")
    plot_dir = os.path.join(results_dir, "variance_adaptor_plots")
    for sub in ["duration", "pitch", "energy"]:
        os.makedirs(os.path.join(plot_dir, sub), exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    wavfile.write(os.path.join(audio_dir, "{}.wav".format(sid)), result["sampling_rate"], result["wav"])
    title = "{}: predicted by the variance adaptor".format(sid)
    plot_duration(result, title, os.path.join(plot_dir, "duration", "{}.png".format(sid)))
    plot_pitch(result, title, os.path.join(plot_dir, "pitch", "{}.png".format(sid)))
    plot_energy(result, title, os.path.join(plot_dir, "energy", "{}.png".format(sid)))

    record = {k: result[k] for k in ["words", "phones", "word_index", "duration_frames", "pitch_hz", "energy"]}
    record["id"] = sid
    record["text"] = text
    record["duration_seconds"] = len(result["wav"]) / result["sampling_rate"]
    with open(os.path.join(plot_dir, "{}.json".format(sid)), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
