"""Plot training and validation loss curves for an ensemble.

Uses the EnsembleView plot methods. Note that the *published* ensemble ships one
checkpoint and a single validation value per network, with no training loss file,
so it has no curves to plot -- only ensembles trained here do.

Example:
    python plot_losses.py --ensembles flow/0301 --out figures/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import flyvis
from flyvis.network.ensemble_view import EnsembleView


def has_curves(ensemble):
    root = Path(flyvis.results_dir) / ensemble
    nets = sorted(d for d in root.iterdir() if d.name.isdigit())
    if not nets:
        return False, "no networks"
    if not (nets[0] / "loss.h5").exists():
        return False, "no training loss file (published models ship only the best checkpoint)"
    return True, f"{len(nets)} networks"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ensembles", nargs="+", default=["flow/0301"])
    p.add_argument("--out", default="figures")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for ensemble in args.ensembles:
        ok, note = has_curves(ensemble)
        print(f"{ensemble}: {note}")
        if not ok:
            continue
        # EnsembleView defaults to ranking by an `epe` validation file, which
        # neither these ensembles nor the published one record
        view = EnsembleView(
            ensemble,
            best_checkpoint_fn_kwargs={
                "validation_subdir": "validation",
                "loss_file_name": "loss",
            },
        )
        tag = ensemble.replace("/", "_")

        fig, ax = view.training_loss()
        ax.set_title(f"{ensemble} training loss")
        fig.savefig(out / f"{tag}_training_loss.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = view.validation_loss()
        ax.set_title(f"{ensemble} validation loss")
        fig.savefig(out / f"{tag}_validation_loss.pdf", bbox_inches="tight")
        plt.close(fig)

        # a smoothed training curve is easier to read than 250k raw points
        losses = np.array([nv.dir.loss[:] for nv in view.values()])
        window = max(1, losses.shape[1] // 500)
        n = (losses.shape[1] // window) * window
        smooth = losses[:, :n].reshape(len(losses), -1, window).mean(axis=2)
        x = np.arange(smooth.shape[1]) * window
        fig, ax = plt.subplots(figsize=(5, 3))
        for i, curve in enumerate(smooth):
            ax.plot(x, curve, lw=0.8, label=view.names[i].split("/")[-1])
        ax.set(xlabel="iterations", ylabel=f"training loss (mean of {window})")
        ax.set_title(f"{ensemble} training loss, smoothed")
        ax.legend(fontsize=6, ncol=2, frameon=False)
        fig.savefig(out / f"{tag}_training_loss_smoothed.pdf", bbox_inches="tight")
        plt.close(fig)

        val = view.validation_losses()
        print(f"  validation: first {val[:, 0].mean():.2f}, "
              f"best {val.min(axis=1).mean():.2f} +- {val.min(axis=1).std():.2f} "
              f"(n={len(val)})")
        print(f"  wrote 3 figures to {out}/{tag}_*")


if __name__ == "__main__":
    main()
