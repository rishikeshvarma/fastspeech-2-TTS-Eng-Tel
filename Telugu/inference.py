"""Synthesise the Telugu test sentences and plot the variance-adaptor predictions.

    python inference.py                                # all test sentences T1..T4, final model
    python inference.py --checkpoint 50000 --text "..." --id custom --speaker te_male

Writes Results/Telugu/test_generated_audios/<ID>.wav and
Results/Telugu/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png (+ <ID>.json).
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
from text import telugu  # noqa: E402
from utils.model import get_model, get_vocoder, load_checkpoint  # noqa: E402

# Text kept exactly as given in the assignment, including the split word "అడ విలో" in T3.
TEST_SENTENCES = {
    "T1": "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    "T2a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    "T2b": "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
    "T3": "పులి అడ విలో నిశ్శబ్దంగా నడిచింది.",
    "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
}


def load_model(configs, checkpoint, device):
    preprocess_config, model_config, train_config = configs
    model = get_model(0, configs, device, train=False)
    path = os.path.join(train_config["path"]["ckpt_path"], "{}.pth.tar".format(checkpoint))
    model.load_state_dict(load_checkpoint(path, device)["model"])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--checkpoint", default="final", help="'final' (published), 'best' or a step number")
    parser.add_argument("--speaker", default="te_female")
    parser.add_argument("--text", default=None)
    parser.add_argument("--id", default="custom")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    parser.add_argument("--results_dir", default=None, help="default: train.path.result_path")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Telugu text on a Windows console

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = load_config(args.config)
    preprocess_config, model_config, train_config = configs
    with open(os.path.join(preprocess_config["path"]["metadata_path"], "speakers.json")) as f:
        speaker_id = json.load(f)[args.speaker]

    model = load_model(configs, args.checkpoint, device)
    vocoder = get_vocoder(model_config, device)

    results_dir = args.results_dir or train_config["path"]["result_path"]
    sentences = {args.id: args.text} if args.text else TEST_SENTENCES
    controls = (args.pitch_control, args.energy_control, args.duration_control)
    for sid, text in sentences.items():
        words = telugu.text_to_words(text)
        print("{}: {}\n    {}".format(sid, telugu.normalize(text), " ".join(p for _, ps in words for p in ps)))
        result = synthesize_phones(model, vocoder, configs, words, speaker_id, device, controls)
        save_outputs(result, sid, text, results_dir)
    print("Saved to", results_dir)
