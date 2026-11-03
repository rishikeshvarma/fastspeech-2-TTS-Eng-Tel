"""Telugu data preparation, forced alignment and feature extraction.

    python preprocess.py prepare    # select 5 h train + 1 h val, write wav/lab + lexicon
    python preprocess.py align      # train a Telugu MFA model and align the corpus
    python preprocess.py features   # mel, pitch, energy, duration -> train.txt / val.txt
    python preprocess.py all

Changes from the reference implementation for Telugu:
  * the corpus is read straight from the Hugging Face parquet shards and resampled to
    22.05 kHz for the universal HiFi-GAN vocoder. The audio is really 16 kHz and about
    36.7 h long (the dataset card says 48 kHz and 8.74 h), with mostly paragraph-length
    female recordings, so only clips up to `max_seconds` are used
  * there is no pretrained Telugu aligner or lexicon, so the lexicon is generated with
    the rule-based G2P in fastspeech2/text/telugu.py and an MFA acoustic model is
    trained on the selected data itself before aligning it
  * numbers in the transcripts are expanded to Telugu words before G2P
"""

import argparse
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
from text import telugu  # noqa: E402

SPEAKERS = {0: "te_female", 1: "te_male"}
LATIN = re.compile(r"[A-Za-z]")


def _shards(corpus_path):
    return sorted(Path(corpus_path, "data").glob("train-*.parquet"))


def scan(corpus_path, sel):
    """One pass over the parquet shards collecting text, speaker and durations.

    "speech" is the duration after trimming edge silence; it is only measured for clips
    short enough to possibly pass the max_seconds filter.
    """
    rows = []
    for shard in _shards(corpus_path):
        pf = pq.ParquetFile(shard)
        batches = pf.iter_batches(batch_size=64, columns=["audio", "text", "gender"])
        for b, batch in enumerate(tqdm(batches, desc="scan " + shard.name)):
            for j, (audio, text, gender) in enumerate(zip(
                batch.column("audio").to_pylist(),
                batch.column("text").to_pylist(),
                batch.column("gender").to_pylist(),
            )):
                info = sf.info(io.BytesIO(audio["bytes"]))
                seconds = info.frames / info.samplerate
                speech = None
                if sel["min_seconds"] <= seconds <= sel["max_seconds"] * 1.5:
                    speech = speech_seconds(audio["bytes"], sel["trim_top_db"])
                rows.append({
                    "key": "{}:{}:{}".format(shard.name, b, j),
                    "speaker": SPEAKERS[gender],
                    "text": text.strip(),
                    "seconds": seconds,
                    "speech": speech,
                })
    return rows


def select(rows, sel):
    """Pick ~train_hours + ~val_hours, balanced over the two speakers.

    Utterances are grouped by their normalised sentence so that a sentence read by
    both speakers can never be in train and validation at the same time.
    """
    usable = []
    for r in rows:
        norm = telugu.normalize(r["text"])
        if LATIN.search(norm) or not telugu.words(norm):
            continue
        if r["speech"] is None or not sel["min_seconds"] <= r["speech"] <= sel["max_seconds"]:
            continue
        r["norm"] = norm
        usable.append(r)

    groups = defaultdict(list)
    for r in usable:
        key = re.sub(r"[^ఀ-౿]", "", r["norm"])
        groups[key].append(r)
    keys = sorted(groups)
    random.Random(sel["seed"]).shuffle(keys)

    # quotas are hours of speech (edge silence excluded), split evenly over the speakers
    speakers = sorted({r["speaker"] for r in usable})
    target = {
        "val": sel["val_hours"] * sel["margin"] * 3600 / len(speakers),
        "train": sel["train_hours"] * sel["margin"] * 3600 / len(speakers),
    }
    got = {s: {spk: 0.0 for spk in speakers} for s in target}
    chosen = {"train": [], "val": []}
    for k in keys:
        # a sentence group goes to one split only; recordings of it by a speaker whose
        # quota for that split is already full are left out rather than moved across
        for split in ["val", "train"]:
            members = [
                r for r in groups[k]
                if got[split][r["speaker"]] + r["speech"] / 2 < target[split]
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
            r["split"] = split
            r["id"] = "{}_{}_{:04d}".format(r["speaker"], split, i)
            by_key[r["key"]] = r

    print("Writing {} utterances ...".format(len(by_key)))
    raw = Path(paths["raw_path"])
    for spk in SPEAKERS.values():
        (raw / spk).mkdir(parents=True, exist_ok=True)
    for shard in _shards(paths["corpus_path"]):
        pf = pq.ParquetFile(shard)
        for b, batch in enumerate(tqdm(pf.iter_batches(batch_size=64, columns=["audio"]), desc=shard.name)):
            for j, audio in enumerate(batch.column("audio").to_pylist()):
                r = by_key.get("{}:{}:{}".format(shard.name, b, j))
                if r is None:
                    continue
                wav, orig_sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
                wav = librosa.resample(wav.mean(axis=1), orig_sr=orig_sr, target_sr=sr)
                # [modified] the reference scaled to exactly 32768, which overflows int16
                wav = wav / max(np.abs(wav).max(), 1e-6) * 0.95
                sf.write(raw / r["speaker"] / (r["id"] + ".wav"), wav, sr, subtype="PCM_16")
                with open(raw / r["speaker"] / (r["id"] + ".lab"), "w", encoding="utf-8") as f:
                    f.write(" ".join(telugu.words(r["norm"])))

    split = {s: sorted("{}/{}".format(r["speaker"], r["id"]) for r in items) for s, items in chosen.items()}
    with open(raw / "split.json", "w", encoding="utf-8") as f:
        json.dump(split, f, indent=1)

    # one lexicon entry per word type, generated by the rule-based G2P
    vocab = sorted({w for items in chosen.values() for r in items for w in telugu.words(r["norm"])})
    with open(paths["lexicon_path"], "w", encoding="utf-8") as f:
        for w in vocab:
            f.write("{}\t{}\n".format(w, " ".join(telugu.word_to_phones(w))))

    summary = {
        "corpus_utterances": len(rows),
        "corpus_hours": sum(r["seconds"] for r in rows) / 3600,
        "usable_utterances": len(usable),
        "usable_speech_hours": sum(r["speech"] for r in usable) / 3600,
        "selected": {
            s: {
                "utterances": len(chosen[s]),
                "speech_hours": sum(got[s].values()) / 3600,
                "recording_hours": sum(r["seconds"] for r in chosen[s]) / 3600,
                "speech_hours_per_speaker": {k: v / 3600 for k, v in got[s].items()},
            }
            for s in chosen
        },
        "lexicon_words": len(vocab),
        "text_overlap_between_train_and_val": len(
            {r["norm"] for r in chosen["train"]} & {r["norm"] for r in chosen["val"]}
        ),
    }
    with open(raw / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))


def align(preprocess_config, args):
    paths = preprocess_config["path"]
    out = os.path.join(paths["preprocessed_path"], "TextGrid")
    cmd = args.mfa.split() + [
        "train", paths["raw_path"], paths["lexicon_path"], paths["mfa_model_path"],
        "--output_directory", out, "--clean", "--overwrite", "-j", str(args.jobs),
        # with only two speakers MFA would otherwise run just two parallel jobs
        "--single_speaker",
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
