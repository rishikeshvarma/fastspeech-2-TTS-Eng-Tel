# FastSpeech 2 for English and Telugu

Two separately trained FastSpeech 2 text-to-speech models: one for Indian-accented English ([Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)) and one for Telugu ([IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu)). Both are adapted from [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), a PyTorch implementation of [FastSpeech 2 (Ren et al., 2021)](https://arxiv.org/abs/2006.04558).

The technical report, with the data selection, loss-curve analysis, per-sentence observations and variance-adaptor analysis, is in **[Report/Technical_Report.pdf](Report/Technical_Report.pdf)**.

## Results at a glance

| | English (Svarah) | Telugu (IndicTTS) |
|---|---|---|
| Training / validation speech | 5.05 h / 1.05 h (3,590 / 790 utterances) | 5.04 h / 1.02 h (2,398 / 485 utterances) |
| Speakers | 115 (about 2 min each) | 2 (2.5 h each) |
| Training | 60k steps, 2 h 00 min | 60k steps, 2 h 32 min |
| Best validation loss | 2.614 at step 13k | 1.960 at step 32k |
| Published checkpoint | step 60,000 | step 32,000 (best validation) |
| ASR error on 100 held-out sentences, synthesised vs real recordings | WER 18.2 % vs 5.7 % | CER 15.8 % vs 16.2 % |

- **Audio** for the ten test sentences: `Results/<language>/test_generated_audios/`.
- **Variance-adaptor plots** (duration, pitch and energy): `Results/<language>/variance_adaptor_plots/{duration,pitch,energy}/`. A JSON file next to each plot holds the per-phoneme values.
- **Loss plots**: `Results/<language>/training_loss.png` and `Results/<language>/validation_loss.png`.
- **Speech-recognition check**: `Results/<language>/asr_eval.json` holds the transcripts from an automatic speech recogniser and the error rates. Its WER/CER compare the transcripts of the synthesised audio with the input text, against the same check on the real recordings.

## Repository layout

```
project/
├── fastspeech2/            shared model code, ported from ming024/FastSpeech2 (see "Changes")
│   ├── model/ transformer/ audio/ hifigan/ utils/ lexicon/
│   ├── text/telugu.py      Telugu normaliser + rule-based G2P                      (new)
│   ├── text/english.py     English normaliser + lexicon/g2p with heteronyms        (new)
│   ├── preprocessor.py     mel / pitch / energy / duration extraction
│   ├── trainer.py          training loop
│   ├── synthesis.py        inference + variance-adaptor plots                      (new)
│   ├── plot_losses.py      training / validation loss plots                        (new)
│   └── asr_eval.py         intelligibility check with Whisper                      (new)
├── English/  preprocess.py  train.py  inference.py  config.yaml  checkpoints/  metadata/
├── Telugu/   preprocess.py  train.py  inference.py  config.yaml  checkpoints/  metadata/
├── Results/{English,Telugu}/
│   ├── training_loss.png  validation_loss.png  asr_eval.json
│   ├── test_generated_audios/<ID>.wav
│   └── variance_adaptor_plots/{duration,pitch,energy}/<ID>.png  (+ <ID>.json)
├── Report/Technical_Report.pdf
├── README.md
└── requirements.txt
```

Each language folder has its own entry scripts and a single `config.yaml`. That file merges the reference's `preprocess.yaml`, `model.yaml` and `train.yaml` into three sections. The model code is shared, not copied twice. `metadata/` holds what inference needs without the corpora (`speakers.json`, `stats.json`) and the exact train/validation lists (`train.txt`, `val.txt`, `selection_summary.json`, `split_summary.json`).

## Checkpoints

`English/checkpoints/final.pth.tar` (step 60,000) and `Telugu/checkpoints/final.pth.tar` (step 32,000) hold the model weights stored as float16, about 72 MB each. The training checkpoints with optimiser state are 418 MB each and are not published. `inference.py` loads `final` by default, and the published results were generated from these files.

## Setup

```bash
conda create -n fs2 python=3.10
conda activate fs2
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt     # NLTK tagger data is fetched on first English run

# Montreal Forced Aligner lives in its own environment (only needed for preprocessing)
conda create -n mfa -c conda-forge montreal-forced-aligner=3.4.3 python=3.11
conda run -n mfa mfa model download acoustic english_us_arpa
```

The CUDA 12.8 PyTorch wheels are the first ones that support RTX 50-series (Blackwell) GPUs. A newer NVIDIA driver, for example one reporting CUDA 13.x, runs them unchanged.

Data goes next to the repository folder, in `../data` (intermediate files in `../work`); the paths are set in each `config.yaml`:

```bash
huggingface-cli download SPRINGLab/IndicTTS_Telugu --repo-type dataset --local-dir ../data/IndicTTS_Telugu
huggingface-cli login     # Svarah is gated: accept its terms on Hugging Face first
huggingface-cli download ai4bharat/Svarah --repo-type dataset --local-dir ../data/Svarah
```

## Running

### Synthesis with the published models (no data needed)

```bash
cd Telugu                                  # or: cd English
python inference.py                        # the five test sentences -> Results/<language>/
python inference.py --text "..." --id my_sentence
python inference.py --speaker te_male      # Telugu voices: te_female (default), te_male
```

English uses voice `m_3834f1c1` by default. Of the 115 Svarah voices it was the most intelligible to the speech recogniser (report, section 5.2); any other id from `English/metadata/speakers.json` can be passed with `--speaker`. `--pitch_control`, `--energy_control` and `--duration_control` scale the variance-adaptor predictions.

### Full pipeline

The same commands work in both language folders:

```bash
cd Telugu                        # or: cd English
python preprocess.py prepare     # data selection, 22.05 kHz wav + transcripts, lexicon
python preprocess.py align       # Montreal Forced Aligner -> TextGrids
python preprocess.py features    # mel / pitch / energy / duration, train.txt + val.txt, metadata/
python train.py                  # 60k steps -> checkpoints/, logs/, Results/<language>/*_loss.png
python train.py --export best    # weights-only float16 copy -> checkpoints/final.pth.tar
python inference.py
```

For English the published model is exported with `python train.py --export 60000`. `python train.py --restore_step N` resumes a run, and `python train.py --plot_only` redraws the loss plots from `logs/*.csv`. TensorBoard logs are written to `logs/train` and `logs/val`.

### Intelligibility check (optional)

```bash
pip install transformers==5.20.2 jiwer==4.0.0
python fastspeech2/asr_eval.py English
python fastspeech2/asr_eval.py Telugu
```

The script transcribes the test-sentence audio with Whisper large-v3-turbo, which it downloads on first use (about 1.6 GB). It then synthesises 100 validation sentences and transcribes both these and the real recordings of the same sentences.

## Data selection in short

Full details are in the report, section 2.

- **Quotas count speech, not file length**: each clip is measured after trimming its leading and trailing silence. Only clips of 1–15 s of speech are used.
- **No sentence appears in both splits**: clips are grouped by their normalised text, the groups are shuffled with seed 1234, and each group goes entirely to validation or to training.
- **Telugu**: the selection is balanced over the two speakers, at 2.55 h of training and 0.51 h of validation speech each.
- **English**: Svarah has no speaker column, so speaker ids are derived from the eight speaker-metadata columns (115 speakers). Each speaker contributes in proportion to its usable speech. Every validation speaker is also in training.

## Changes from ming024/FastSpeech2

**Shared code**
- `text/telugu.py` (new): Telugu front-end. It normalises Unicode and expands numbers, ordinals and times into Telugu words. Rule-based G2P over aksharas (inherent vowel, virama, homorganic anusvara, visarga).
- `text/english.py` (new): English front-end. It expands times, dates, ordinals, years and numbers. Words are looked up in the LibriSpeech lexicon, then CMUdict, then g2p_en. POS tags choose between heteronym pronunciations.
- `text/symbols.py`: 46 Telugu phones added as lower-case ASCII labels; pinyin removed.
- `preprocessor.py`: changes from the reference:
  - the train/validation split comes from the selection step instead of a random split;
  - normalisation statistics are computed on the training split only;
  - speaker ids are deterministic;
  - feature extraction runs in parallel;
  - MFA 3 empty-interval silences are read correctly;
  - utterances containing `spn` are dropped;
  - every pause becomes a single `sp`;
  - a 1,300-frame limit;
  - the metadata is exported to `<language>/metadata/`.
- `trainer.py` / `evaluate.py`: single-GPU training, optional bf16 autocast, a fixed seed, CSV loss logs, and a best-validation checkpoint.
- `model/modules.py`:
  - fixed a bug where the energy predictor received the pitch control value;
  - the length regulator is vectorised;
  - tensors are created on the right device.
- `audio/*`, `utils/*`, `model/optimizer.py`:
  - work with librosa 0.11, NumPy 2 and PyTorch 2.11 (checkpoints load with `weights_only`);
  - the vocoder path is relative to the package;
  - new float16 checkpoint export.
- `synthesis.py`, `plot_losses.py`, `asr_eval.py`, `selection.py`, `configuration.py` (new): inference with plots, loss plots, ASR evaluation, speech-duration measurement, and single-file configs.

**English**: `English/preprocess.py`
- reads Svarah from parquet;
- derives speaker keys from the metadata;
- filters out transcripts with tags or ambiguous symbols;
- selects data per speaker in proportion to its usable speech;
- builds the MFA dictionary;
- aligns with the pretrained `english_us_arpa` model.

**Telugu**: `Telugu/preprocess.py`
- reads IndicTTS from parquet;
- balances the selection over the two speakers;
- generates the lexicon with the rule-based G2P;
- trains an MFA acoustic model on the selected data and aligns with it, since no pretrained Telugu aligner exists.

## Acknowledgements

The model, training and vocoder code is from [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (MIT licence, see `fastspeech2/LICENSE`). The pretrained HiFi-GAN universal vocoder is by [Kong et al.](https://github.com/jik876/hifi-gan) (MIT, see `fastspeech2/hifigan/LICENSE`). The datasets are Svarah by AI4Bharat and IndicTTS Telugu by the Indic TTS consortium (IIT Madras); the audio is not redistributed here.
