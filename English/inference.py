"""Synthesise the English test sentences and plot the variance-adaptor predictions.

    python inference.py                                # all test sentences E1..E4, final model
    python inference.py --checkpoint best --text "..." --id custom --speaker <name>

Writes Results/English/test_generated_audios/<ID>.wav and
Results/English/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png (+ <ID>.json).
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fastspeech2"))

import torch  # noqa: E402

from configuration import load_config  # noqa: E402
from synthesis import synthesize_phones, save_outputs  # noqa: E402
from text.english import EnglishFrontend  # noqa: E402
from utils.model import get_model, get_vocoder, load_checkpoint  # noqa: E402

# Text kept exactly as given in the assignment, including the split word "rhyth m" in E3.
TEST_SENTENCES = {
    "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3": "The rhyth m of the orchestra changed unexpectedly.",
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
}


def load_model(configs, checkpoint, device):
    preprocess_config, model_config, train_config = configs
    model = get_model(0, configs, device, train=False)
    path = os.path.join(train_config["path"]["ckpt_path"], "{}.pth.tar".format(checkpoint))
    model.load_state_dict(load_checkpoint(path, device)["model"])
    return model


# Svarah speakers differ a lot in how intelligible the model makes them (Whisper WER of
# synthesised validation sentences ranges from 9 % to 75 % over the 115 voices). This
# voice had the lowest WER on 60 held-out validation sentences (9.8 %, against 20.6 %
# for the speaker with the most training audio, m_d270faea); see the report, section 5.2.
DEFAULT_SPEAKER = "m_3834f1c1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--checkpoint", default="final", help="'final' (published), 'best' or a step number")
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER, help="see metadata/speakers.json")
    parser.add_argument("--text", default=None)
    parser.add_argument("--id", default="custom")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    parser.add_argument("--results_dir", default=None, help="default: train.path.result_path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = load_config(args.config)
    preprocess_config, model_config, train_config = configs
    with open(os.path.join(preprocess_config["path"]["metadata_path"], "speakers.json")) as f:
        speaker_id = json.load(f)[args.speaker]
    print("speaker:", args.speaker)

    model = load_model(configs, args.checkpoint, device)
    vocoder = get_vocoder(model_config, device)
    frontend = EnglishFrontend(preprocess_config["path"]["lexicon_path"])

    results_dir = args.results_dir or train_config["path"]["result_path"]
    sentences = {args.id: args.text} if args.text else TEST_SENTENCES
    controls = (args.pitch_control, args.energy_control, args.duration_control)
    for sid, text in sentences.items():
        print("{}: {}".format(sid, text))
        words = frontend.text_to_words(text)
        print("    " + " ".join(p for _, ps in words for p in ps))
        result = synthesize_phones(model, vocoder, configs, words, speaker_id, device, controls)
        save_outputs(result, sid, text, results_dir)
    print("Saved to", results_dir)
