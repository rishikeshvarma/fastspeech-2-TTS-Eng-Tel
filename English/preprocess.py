"""English (Svarah) data preparation, forced alignment and feature extraction.

    python preprocess.py prepare    # select 5 h train + 1 h val, write wav/lab + MFA dictionary
    python preprocess.py align      # align with MFA's pretrained english_us_arpa model
    python preprocess.py features   # mel, pitch, energy, duration -> train.txt / val.txt
    python preprocess.py all

Changes from the reference implementation for Svarah:
  * the corpus is read straight from the Hugging Face parquet shard(s) and resampled to
    22.05 kHz for the universal HiFi-GAN vocoder
  * Svarah has no speaker-id column; a speaker key is derived from the speaker metadata
    (see speaker_key) and the model is trained multi-speaker
  * the train/val split is stratified per speaker (every validation speaker is also
    seen in training, so its speaker embedding is learnt) and grouped by sentence so the
    same prompt never appears in both splits
  * the MFA dictionary is the reference LibriSpeech lexicon restricted to corpus words,
    with g2p_en pronunciations added for out-of-vocabulary words
"""

import argparse
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fastspeech2"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import librosa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from tqdm import tqdm  # noqa: E402

from configuration import load_config  # noqa: E402
from selection import speech_seconds  # noqa: E402
from text import english  # noqa: E402

META = [
    "gender", "age-group", "primary_language", "native_place_state",
    "native_place_district", "highest_qualification", "job_category", "occupation_domain",
]
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")


def speaker_key(meta):
    """Stable short id for a speaker, built from the metadata columns.

    The file names do not identify speakers (their first field takes 2,938 values, one
    per recording session). The eight metadata columns take 115 distinct values against
    the 117 speakers reported for Svarah, so at most two pairs of speakers are merged.
    """
    raw = "|".join(str(meta.get(k, "")) for k in META)
    return "{}_{}".format(str(meta.get("gender", "x"))[:1].lower() or "x", hashlib.md5(raw.encode()).hexdigest()[:8])


def _shards(corpus_path):
    return sorted(Path(corpus_path, "data").glob("*.parquet"))


def scan(corpus_path, sel):
    """Collect text, speaker and durations; "speech" excludes edge silence."""
    rows = []
    for shard in _shards(corpus_path):
        pf = pq.ParquetFile(shard)
        cols = ["audio_filepath", "text"] + META
        for b, batch in enumerate(tqdm(pf.iter_batches(batch_size=64, columns=cols), desc="scan " + shard.name)):
            data = batch.to_pydict()
            for j in range(batch.num_rows):
                audio = data["audio_filepath"][j]
                info = sf.info(io.BytesIO(audio["bytes"]))
                seconds = info.frames / info.samplerate
                speech = None
                if sel["min_seconds"] <= seconds <= sel["max_seconds"] * 1.5:
                    speech = speech_seconds(audio["bytes"], sel["trim_top_db"])
                meta = {k: data[k][j] for k in META}
                rows.append({
                    "key": "{}:{}:{}".format(shard.name, b, j),
                    "path": audio.get("path"),
                    "speaker": speaker_key(meta),
                    "text": (data["text"][j] or "").strip(),
                    "seconds": seconds,
                    "speech": speech,
                    "sr": info.samplerate,
                })
    return rows


def clean_text(text):
    """Normalised, lower-case word sequence for the aligner (None if unusable)."""
    if re.search(r"[\[\]<>{}#*_@/\\|=+%&$()]", text):
        return None  # annotation tags / symbols whose spoken form is ambiguous
    if re.search(r"\d{5,}|\d\s*[-–]\s*\d", text):
        return None  # phone/account numbers are read digit by digit, ranges as "x to y"
    norm = english.normalize(text)
    norm = norm.replace("<A>", "a").replace("<P>", "p").replace("<M>", "m")
    words = WORD_RE.findall(norm.lower().replace("’", "'"))
    if not words or re.search(r"\d", norm):
        return None
    return " ".join(words)


def select(rows, sel):
    usable = []
    for r in rows:
        if r["speech"] is None or not sel["min_seconds"] <= r["speech"] <= sel["max_seconds"]:
            continue
        words = clean_text(r["text"])
        if words is None:
            continue
        r["words"] = words
        usable.append(r)

    # quotas are hours of speech (edge silence excluded)
    hours = defaultdict(float)
    for r in usable:
        hours[r["speaker"]] += r["speech"]
    total = sum(hours.values())
    scale = (sel["train_hours"] + sel["val_hours"]) * sel["margin"] * 3600 / total
    # each speaker contributes in proportion to how much usable speech it has
    target = {
        spk: {
            "val": h * min(scale, 1.0) * sel["val_hours"] / (sel["train_hours"] + sel["val_hours"]),
            "train": h * min(scale, 1.0) * sel["train_hours"] / (sel["train_hours"] + sel["val_hours"]),
        }
        for spk, h in hours.items()
    }

    groups = defaultdict(list)
    for r in usable:
        groups[r["words"]].append(r)
    keys = sorted(groups)
    random.Random(sel["seed"]).shuffle(keys)

    got = {s: defaultdict(float) for s in ["val", "train"]}
    chosen = {"train": [], "val": []}
    for k in keys:
        for split in ["val", "train"]:
            # per-speaker quotas are only ~30 s for validation, so take an utterance only
            # if that brings the speaker closer to its quota (no systematic overshoot)
            members = [
                r for r in groups[k]
                if got[split][r["speaker"]] + r["speech"] / 2 < target[r["speaker"]][split]
            ]
            if members:
                for r in members:
                    got[split][r["speaker"]] += r["speech"]
                    chosen[split].append(r)
                break
    return usable, chosen, got


def prepare(preprocess_config):
    paths = preprocess_config["path"]
    sel = preprocess_config["selection"]
    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    print("Scanning corpus ...")
    rows = scan(paths["corpus_path"], sel)
    usable, chosen, got = select(rows, sel)

    by_key = {}
    for split, items in chosen.items():
        for i, r in enumerate(sorted(items, key=lambda r: r["key"])):
            r["id"] = "{}_{}_{:04d}".format(r["speaker"], split, i)
            by_key[r["key"]] = r

    print("Writing {} utterances ...".format(len(by_key)))
    raw = Path(paths["raw_path"])
    for shard in _shards(paths["corpus_path"]):
        pf = pq.ParquetFile(shard)
        for b, batch in enumerate(tqdm(pf.iter_batches(batch_size=64, columns=["audio_filepath"]), desc=shard.name)):
            for j, audio in enumerate(batch.column("audio_filepath").to_pylist()):
                r = by_key.get("{}:{}:{}".format(shard.name, b, j))
                if r is None:
                    continue
                (raw / r["speaker"]).mkdir(parents=True, exist_ok=True)
                wav, orig_sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
                wav = librosa.resample(wav.mean(axis=1), orig_sr=orig_sr, target_sr=sr)
                # [modified] the reference scaled to exactly 32768, which overflows int16
                wav = wav / max(np.abs(wav).max(), 1e-6) * 0.95
                sf.write(raw / r["speaker"] / (r["id"] + ".wav"), wav, sr, subtype="PCM_16")
                with open(raw / r["speaker"] / (r["id"] + ".lab"), "w", encoding="utf-8") as f:
                    f.write(r["words"])

    split = {s: sorted("{}/{}".format(r["speaker"], r["id"]) for r in items) for s, items in chosen.items()}
    with open(raw / "split.json", "w", encoding="utf-8") as f:
        json.dump(split, f, indent=1)

    # MFA dictionary: reference lexicon for known words, g2p_en for the rest
    from g2p_en import G2p

    lexicon = english.read_lexicon(paths["lexicon_path"])
    g2p = G2p()
    vocab = sorted({w for items in chosen.values() for r in items for w in r["words"].split()})
    oov = []
    with open(paths["mfa_dictionary_path"], "w", encoding="utf-8") as f:
        for w in vocab:
            if w in lexicon:
                phones = lexicon[w]
            elif w in g2p.cmu:
                phones = g2p.cmu[w][0]
            else:
                phones = g2p.predict(w)
                oov.append(w)
            f.write("{}\t{}\n".format(w, " ".join(phones)))

    train_hours = {spk: h / 3600 for spk, h in got["train"].items()}
    summary = {
        "corpus_utterances": len(rows),
        "corpus_hours": sum(r["seconds"] for r in rows) / 3600,
        "corpus_speakers": len({r["speaker"] for r in rows}),
        "sample_rates": sorted({r["sr"] for r in rows}),
        "usable_utterances": len(usable),
        "usable_speech_hours": sum(r["speech"] for r in usable) / 3600,
        "selected": {
            s: {
                "utterances": len(chosen[s]),
                "speech_hours": sum(got[s].values()) / 3600,
                "recording_hours": sum(r["seconds"] for r in chosen[s]) / 3600,
                "speakers": len({r["speaker"] for r in chosen[s]}),
            }
            for s in chosen
        },
        "val_speakers_unseen_in_train": len(
            {r["speaker"] for r in chosen["val"]} - {r["speaker"] for r in chosen["train"]}
        ),
        "text_overlap_between_train_and_val": len(
            {r["words"] for r in chosen["train"]} & {r["words"] for r in chosen["val"]}
        ),
        "dictionary_words": len(vocab),
        "dictionary_words_from_g2p": len(oov),
        "default_speaker": max(train_hours, key=train_hours.get),
        "train_hours_per_speaker": dict(sorted(train_hours.items(), key=lambda x: -x[1])),
    }
    with open(raw / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "train_hours_per_speaker"}, indent=2))


def pretrained_mfa_model(name):
    """Path of a model fetched with `mfa model download acoustic <name>`."""
    root = os.environ.get("MFA_ROOT_DIR", os.path.join(os.path.expanduser("~"), "Documents", "MFA"))
    return os.path.join(root, "pretrained_models", "acoustic", name + ".zip")


def align(preprocess_config, args):
    paths = preprocess_config["path"]
    out = os.path.join(paths["preprocessed_path"], "TextGrid")
    model = pretrained_mfa_model(preprocess_config["mfa"]["acoustic_model"])
    cmd = args.mfa.split() + [
        "align", paths["raw_path"], paths["mfa_dictionary_path"], model, out,
        "--clean", "--overwrite", "-j", str(args.jobs),
        # MFA names its work folder after the corpus folder ("raw_data" for both languages)
        "--temporary_directory", os.path.join(os.path.dirname(paths["raw_path"]), "mfa_tmp"),
    ]
    print(" ".join(cmd))
    # MFA prints Unicode progress bars; without UTF-8 they crash on a Windows console
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    subprocess.run(cmd, check=True, env=env)


def features(preprocess_config, args):
    from preprocessor import Preprocessor

    Preprocessor(preprocess_config).build_from_path(num_workers=args.jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["prepare", "align", "features", "all"])
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument(
        "--mfa", default="conda run --no-capture-output -n mfa mfa",
        help="command that runs Montreal Forced Aligner",
    )
    args = parser.parse_args()
    preprocess_config, _, _ = load_config(args.config)

    if args.step in ["prepare", "all"]:
        prepare(preprocess_config)
    if args.step in ["align", "all"]:
        align(preprocess_config, args)
    if args.step in ["features", "all"]:
        features(preprocess_config, args)
