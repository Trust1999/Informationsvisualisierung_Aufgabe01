"""
IVDA 2 - Praktikum 1: Klassen fuer Aufgabe 1 (Training, Testauswertung, Kennzahlen).

Trainer        Training mit Early Stopping, Checkpointing, Lernkurven
Evaluation     einmalige Testauswertung: Accuracy, F1, Confusion Matrix, Fehlerpaare, Beispielbilder
MetricsStore   sammelt Kennzahlen aller Aufgaben und schreibt sie als JSON
"""

import os
import time

import matplotlib
matplotlib.use("Agg")   # alle Abbildungen werden als PNG gespeichert, es wird kein Fenster gebraucht
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score

from praktikum1_starter import CLASSES, SEED, device, extract, run_epoch, show_images

OUT_DIR = "output"                                    # Ausgabeordner fuer alle Abbildungen und Kennzahlen
STATE_PATH = os.path.join(OUT_DIR, "best_state.pt")   # bester Zustand waehrend des Trainings (Zwischendatei)
MODEL_PATH = "best_model.pt"                          # Abgabedatei: state_dict + Trainingshistorie
MAX_EPOCHS = 50
ES_PATIENCE = 6


"""
Training mit Early Stopping auf dem Val-Loss, Checkpointing und Lernkurven.
"""
class Trainer:

    def __init__(self, model, train_loader, val_loader, optimizer, scheduler,
                 max_epochs=MAX_EPOCHS, patience=ES_PATIENCE, state_path=STATE_PATH):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.patience = patience
        self.state_path = state_path
        self.history = None   # wird durch fit() oder load() gesetzt

    # -- Training ---------------------------------------------------------------------------------------------
    """
    Max. max_epochs Epochen; Abbruch, wenn der Val-Loss patience Epochen in Folge kein neues Minimum erreicht.
    Bei jedem neuen Minimum wird der Zustand nach state_path gespeichert (restore_best() laedt ihn zurueck).
    """
    def fit(self):
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "epoch_time": []}
        best_val_loss, best_epoch = float("inf"), 0
        epochs_without_improvement = 0
        stopped_early = False

        training_start = time.perf_counter()
        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.perf_counter()
            lr = self.optimizer.param_groups[0]["lr"]                       # Lernrate, mit der diese Epoche läuft
            train_loss, train_acc = run_epoch(self.model, self.train_loader, self.optimizer)
            val_loss, val_acc = run_epoch(self.model, self.val_loader)
            epoch_time = time.perf_counter() - epoch_start                  # Laufzeit Training + Validierung
            self.scheduler.step(val_loss)                                   # genau einmal pro Epoche

            for key, value in (("train_loss", train_loss), ("train_acc", train_acc), ("val_loss", val_loss),
                               ("val_acc", val_acc), ("lr", lr), ("epoch_time", epoch_time)):
                history[key].append(value)

            improved = val_loss < best_val_loss                             # neues Minimum des Val-Loss
            if improved:
                best_val_loss, best_epoch = val_loss, epoch
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.state_path)
            else:
                epochs_without_improvement += 1

            print(f"Epoche {epoch:2d}/{self.max_epochs} | "
                  f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | "
                  f"lr {lr:.2e} | {epoch_time:5.1f} s" + ("  <- bestes Modell" if improved else ""))

            if epochs_without_improvement >= self.patience:
                stopped_early = True
                print(f"Early Stopping: {self.patience} Epochen in Folge ohne neues Minimum des Val-Loss.")
                break

        history["total_time"] = time.perf_counter() - training_start        # Gesamtlaufzeit des Trainings in s
        history["best_epoch"] = best_epoch                                  # 1-basiert
        history["best_val_loss"] = best_val_loss
        history["epochs_run"] = len(history["train_loss"])
        history["stopped_early"] = stopped_early
        self.history = history
        return history


    """
    Besten Zustand zurückladen.
    """
    def restore_best(self):
        self.model.load_state_dict(torch.load(self.state_path, map_location=device))

    """
    Abgabedatei: Gewichte (state_dict) und Trainingshistorie. Erst nach restore_best() aufrufen.
    """
    def save(self, path=MODEL_PATH):
        torch.save({"state_dict": self.model.state_dict(), "history": self.history}, path)

    """
    Gespeichertes Modell und Historie laden 
    """
    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.history = ckpt["history"]

    """
    Loss und Accuracy für Training und Validierung über die Epochen, Epoche des gespeicherten Modells markiert.
    """
    def plot(self, path):
        h = self.history
        n_epochs = len(h["train_loss"])
        epochs = np.arange(1, n_epochs + 1)
        best = int(h["best_epoch"])
        lr_drops = [e for e in range(2, n_epochs + 1) if h["lr"][e - 1] < h["lr"][e - 2] * 0.999]

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
        for ax, key, ylabel in zip(axes, ("loss", "acc"), ("Loss (Cross-Entropy)", "Accuracy")):
            ax.plot(epochs, h[f"train_{key}"], marker="o", ms=3, label="Training (Train-Modus, Dropout aktiv)")
            ax.plot(epochs, h[f"val_{key}"], marker="o", ms=3, label="Validierung (eval-Modus)")
            ax.axvline(best, color="black", ls="--", lw=1, label=f"gespeichertes Modell (Epoche {best})")
            ax.scatter([best], [h[f"val_{key}"][best - 1]], color="black", zorder=5, s=35)
            for k, e in enumerate(lr_drops):
                ax.axvline(e - 0.5, color="grey", ls=":", lw=1, label="Lernrate × 0,3" if k == 0 else None)
            ax.set_xlabel("Epoche")
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.3)
            ax.legend()
        axes[0].set_title("Loss über die Epochen")
        axes[1].set_title("Accuracy über die Epochen")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    """
    Zusammenfassung in Konsole ausgeben.
    """
    def print_summary(self):
        h, best = self.history, self.history["best_epoch"]
        print(f"\nBestes Modell: Epoche {best} von {len(h['train_loss'])} "
              f"(Val Loss {h['val_loss'][best - 1]:.4f}, Val Acc {h['val_acc'][best - 1]:.4f}, "
              f"Train Loss {h['train_loss'][best - 1]:.4f}, Train Acc {h['train_acc'][best - 1]:.4f})")
        print(f"Laufzeit: gesamt {h['total_time']:.1f} s, "
              f"im Durchschitt {float(np.mean(h['epoch_time'])):.1f} s je Epoche")

    """
    Kennzahlen des Trainings als JSON-faehiges Dict.
    """
    def metrics(self):
        h, best = self.history, self.history["best_epoch"]
        return {
            "beste_epoche": int(best),
            "epochen_gelaufen": len(h["train_loss"]),
            "early_stopping_ausgeloest": bool(h["stopped_early"]),
            "bester_val_loss": float(h["val_loss"][best - 1]),
            "val_acc_beste_epoche": float(h["val_acc"][best - 1]),
            "train_loss_beste_epoche": float(h["train_loss"][best - 1]),
            "train_acc_beste_epoche": float(h["train_acc"][best - 1]),
            "gesamtlaufzeit_s": float(h["total_time"]),
            "laufzeit_je_epoche_s": [float(t) for t in h["epoch_time"]],
            "mittlere_laufzeit_je_epoche_s": float(np.mean(h["epoch_time"])),
            "lernrate_je_epoche": [float(v) for v in h["lr"]],
            "train_loss": [float(v) for v in h["train_loss"]],
            "train_acc": [float(v) for v in h["train_acc"]],
            "val_loss": [float(v) for v in h["val_loss"]],
            "val_acc": [float(v) for v in h["val_acc"]],
        }
