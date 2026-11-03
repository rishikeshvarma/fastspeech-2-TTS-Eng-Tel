import os
import json
import shutil
from concurrent.futures import ProcessPoolExecutor

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio

# [modified] the reference split train/val at random after feature extraction
# (`val_size` utterances). Here the split is decided earlier, per language, by
# the `prepare` step of <Language>/preprocess.py and read from <raw_path>/split.json
# so train and validation never overlap and each one has a controlled number of hours.

# [new] files copied from the work directories into <metadata_path>, which is part
# of the repository: inference needs only speakers.json and stats.json (to build the
# model and de-normalise pitch/energy), the rest documents the data selection.
METADATA_FILES = {
    "preprocessed_path": ["speakers.json", "stats.json", "split_summary.json", "train.txt", "val.txt"],
    "raw_path": ["selection_summary.json"],
}


def export_metadata(config):
    out = config["path"]["metadata_path"]
    os.makedirs(out, exist_ok=True)
    for key, names in METADATA_FILES.items():
        for name in names:
            shutil.copyfile(os.path.join(config["path"][key], name), os.path.join(out, name))
    print("Copied metadata to", out)

_worker = None


def _init_worker(config):
    global _worker
    import torch

    torch.set_num_threads(1)
    _worker = Preprocessor(config)


def _process(item):
    speaker, basename = item
    try:
        return item, _worker.process_utterance(speaker, basename)
    except Exception as e:  # a single broken file should not kill a multi-hour run
        print("skipping {}/{}: {}".format(speaker, basename, e))
        return item, None


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.max_frames = config["preprocessing"].get("max_frames", None)

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self, num_workers=4):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        with open(os.path.join(self.in_dir, "split.json"), encoding="utf-8") as f:
            split = json.load(f)  # {"train": ["spk/base", ...], "val": [...]}
        split_of = {}
        for name, items in split.items():
            for item in items:
                split_of[tuple(item.split("/"))] = name

        print("Processing Data ...")
        out = {name: [] for name in split}
        n_frames = {name: 0 for name in split}
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # [modified] speaker ids are assigned in sorted order so they are reproducible
        speakers = {s: i for i, s in enumerate(sorted({spk for spk, _ in split_of}))}

        todo = []
        for speaker, basename in sorted(split_of):
            tg_path = os.path.join(
                self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
            )
            # [modified] upstream kept going with stale pitch/energy when a TextGrid was missing
            if os.path.exists(tg_path):
                todo.append((speaker, basename))
        print("{} of {} utterances have alignments".format(len(todo), len(split_of)))

        # Compute pitch, energy, duration, and mel-spectrogram
        # [modified] parallelised over utterances
        with ProcessPoolExecutor(
            num_workers, initializer=_init_worker, initargs=(self.config,)
        ) as ex:
            for item, ret in tqdm(ex.map(_process, todo, chunksize=8), total=len(todo)):
                if ret is None:
                    continue
                info, pitch, energy, n = ret
                name = split_of[item]
                out[name].append(info)

                # statistics come from the training split only
                if name == "train":
                    if len(pitch) > 0:
                        pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                    if len(energy) > 0:
                        energy_scaler.partial_fit(energy.reshape((-1, 1)))

                n_frames[name] += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        hours = {
            name: n * self.hop_length / self.sampling_rate / 3600
            for name, n in n_frames.items()
        }
        for name in out:
            print("{}: {} utterances, {:.2f} hours".format(name, len(out[name]), hours[name]))
        with open(os.path.join(self.out_dir, "split_summary.json"), "w") as f:
            json.dump(
                {name: {"utterances": len(out[name]), "hours": hours[name]} for name in out},
                f,
                indent=2,
            )

        # Write metadata
        for name in out:
            with open(
                os.path.join(self.out_dir, "{}.txt".format(name)), "w", encoding="utf-8"
            ) as f:
                for m in sorted(out[name]):
                    f.write(m + "\n")

        export_metadata(self.config)
        return out

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Get alignments
        # [modified] MFA 2.x/3.x writes silences as empty intervals, which tgt drops by default
        textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
        tier = textgrid.get_tier_by_name("phones")
        # [modified] an "spn" interval is speech the aligner could not transcribe; the
        # reference trained on it as if it were a phone, we drop the utterance instead
        if any(t.text == "spn" for t in tier._objects):
            return None
        phone, duration, start, end = self.get_alignment(tier)
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None
        if self.max_frames is not None and sum(duration) > self.max_frames:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn", ""]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                # [modified] every pause is the single "sp" symbol the text front-ends emit
                phones.append("sp")

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
