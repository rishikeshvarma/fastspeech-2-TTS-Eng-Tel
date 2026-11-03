"""Train the English FastSpeech 2 model.

    python train.py                       # from scratch
    python train.py --restore_step 20000  # resume from checkpoints/20000.pth.tar
    python train.py --plot_only           # just redraw Results/English/*_loss.png
    python train.py --export 60000        # checkpoints/60000.pth.tar -> checkpoints/final.pth.tar
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fastspeech2"))

from configuration import load_config  # noqa: E402
from plot_losses import plot_losses  # noqa: E402
from utils.model import export_checkpoint  # noqa: E402
import trainer  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--export", metavar="CHECKPOINT",
                        help="write a weights-only float16 copy of CHECKPOINT (step or 'best') to final.pth.tar")
    args = parser.parse_args()

    configs = load_config(args.config)
    train_config = configs[2]
    if args.export:
        ckpt_path = train_config["path"]["ckpt_path"]
        export_checkpoint(os.path.join(ckpt_path, "{}.pth.tar".format(args.export)),
                          os.path.join(ckpt_path, "final.pth.tar"))
        sys.exit()
    if not args.plot_only:
        trainer.main(args.restore_step, configs)
    plot_losses(train_config["path"]["log_path"], train_config["path"]["result_path"], "English")
