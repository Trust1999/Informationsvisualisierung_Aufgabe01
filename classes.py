"""
IVDA 2 - Praktikum 1: Klassen für die Aufgabe 1und Aufgabe 2

Trainer        Training mit Early Stopping, Checkpointing, Lernkurven
Evaluation     einmalige Testauswertung: Accuracy, F1, Confusion Matrix, Fehlerpaare, Beispielbilder
MetricsStore   sammelt Kennzahlen aller Aufgaben und schreibt sie als JSON
LayoutStudy    Aufgabe 2: t-SNE/UMAP-Layouts, Trustworthiness/Continuity, Darstellungen
"""

import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Subset

from praktikum1_starter import CLASSES, SEED, align, device, extract, run_epoch, show_images, test_subsample

OUT_DIR = "output"                                    # Ausgabeordner für alle Abbildungen und Kennzahlen
STATE_PATH = os.path.join(OUT_DIR, "best_state.pt")   # bester Zustand während des Trainings (Zwischendatei)
MODEL_PATH = "best_model.pt"                          # Abgabedatei: state_dict + Trainingshistorie
METRICS_PATH = os.path.join(OUT_DIR, "kennzahlen.json")
MAX_EPOCHS = 50
ES_PATIENCE = 6
TSNE_PERPLEXITIES = (5, 30, 100)      # für Aufgabe 2: Parameterstudie t-SNE
UMAP_N_NEIGHBORS = (5, 15, 100)       # für Aufgabe 2: Parameterstudie UMAP
K_VALUES = (5, 15, 30, 100, 500)      # für Aufgabe 2: K fuer Trustworthiness und Continuity
PURITY_K = 15                         # für Aufgabe 2: Nachbarzahl für das Hilfsmass neighbor_statistics()


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
        self.history = None   # wird von fit() oder load() gesetzt

    """
    Max. max_epochs Epochen; Abbruch, wenn der Val-Loss patience Epochen in Folge kein neues Minimum erreicht.
    Bei jedem neuen Minimum wird der Zustand nach state_path gespeichert 
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
                  f"lr {lr:.2e} | {epoch_time:5.1f} s" + ("  ist bestes Modell" if improved else ""))

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
    Besten Zustand zurückladen (sonst ist es as Gewichte der letzten Epoche).
    """
    def restore_best(self):
        self.model.load_state_dict(torch.load(self.state_path, map_location=device))


    """
    Abgabedatei: Gewichte (state_dict) und Trainingshistorie.
    """
    def save(self, path=MODEL_PATH):
        torch.save({"state_dict": self.model.state_dict(), "history": self.history}, path)


    """
    Gespeichertes Modell und Historie laden (Training wird übersprungen).
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
    Zusammenfassung in der Konsole ausgeben
    """
    def print_summary(self):
        h, best = self.history, self.history["best_epoch"]
        print(f"\nBestes Modell: Epoche {best} von {len(h['train_loss'])} "
              f"(Val Loss {h['val_loss'][best - 1]:.4f}, Val Acc {h['val_acc'][best - 1]:.4f}, "
              f"Train Loss {h['train_loss'][best - 1]:.4f}, Train Acc {h['train_acc'][best - 1]:.4f})")
        print(f"Laufzeit: gesamt {h['total_time']:.1f} s, "
              f"im Mittel {float(np.mean(h['epoch_time'])):.1f} s je Epoche")

    
    """
    Kennzahlen des Trainings als JSON.
    """
    def metrics(self):
        h, best = self.history, self.history["best_epoch"]
        return {
            "beste_epoche": int(best),
            "epochen_gelaufen": len(h["train_loss"]),
            "early_stopping_ausgelöst": bool(h["stopped_early"]),
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



"""
Einmalige Endbewertung des finalen Modells auf einem Datensatz (hier: Test-Split).
"""
class Evaluation:
    def __init__(self, model, dataset, out_dir=OUT_DIR, n_pairs=3, n_examples=3):
        self.dataset = dataset
        self.out_dir = out_dir
        self.n_examples = n_examples
        self.y, self.pred, _ = extract(model, dataset)                       # Zeile i zu dataset[i]
        self.accuracy = accuracy_score(self.y, self.pred)
        self.f1 = f1_score(self.y, self.pred, average=None)                  # je Klasse, Reihenfolge wie CLASSES
        self.counts = confusion_matrix(self.y, self.pred)                    # Rohzählungen: Zeile = wahr, Spalte = vorhergesagt
        self.normalized = confusion_matrix(self.y, self.pred, normalize="true")   # zeilennormiert
        self.recall = np.diag(self.normalized)
        self.weakest_f1 = int(np.argmin(self.f1))
        self.worst_class = self._find_worst_class()
        self.pairs = self._find_top_pairs(n_pairs)

    
    """
    Klasse mit der schlechtesten Diagonale (Recall) und ihre meisten Fehlerziele.
    """
    def _find_worst_class(self):
        worst = int(np.argmin(self.recall))
        row = self.normalized[worst].copy()
        row[worst] = 0.0
        return {
            "klasse": CLASSES[worst],
            "recall": float(self.recall[worst]),
            "häufigste_fehlerziele": [
                {"vorhergesagt_als": CLASSES[j], "anteil": float(row[j]), "anzahl": int(self.counts[worst, j])}
                for j in np.argsort(row)[::-1][:3] if row[j] > 0],
        }

    def _find_top_pairs(self, n_pairs):
        """Häuufigste Verwechslungspaare {i, j}, gerankt nach n(i->j) + n(j->i); dominante Richtung = mehr Fehler."""
        totals = sorted(((int(self.counts[i, j] + self.counts[j, i]), i, j)
                         for i in range(10) for j in range(i + 1, 10)), reverse=True)
        pairs = []
        for total, i, j in totals[:n_pairs]:
            n_ij, n_ji = int(self.counts[i, j]), int(self.counts[j, i])
            a, b = (i, j) if n_ij >= n_ji else (j, i)
            pairs.append({
                "klassen": [CLASSES[i], CLASSES[j]],
                "anzahl_i_zu_j": n_ij, "anzahl_j_zu_i": n_ji, "summe": total,
                "rate_i_zu_j": float(self.normalized[i, j]), "rate_j_zu_i": float(self.normalized[j, i]),
                "dominante_richtung": f"{CLASSES[a]} -> {CLASSES[b]}",
                "dominant_ids": [int(a), int(b)],
            })
        return pairs

    """
    Zeilennormierte Confusion Matrix: Zeile = wahre Klasse, Spalte = Vorhersage, Diagonale = Recall.
    """
    def plot_confusion_matrix(self, path):
        fig, ax = plt.subplots(figsize=(9.5, 8))
        display = ConfusionMatrixDisplay(confusion_matrix=self.normalized, display_labels=CLASSES)
        display.plot(ax=ax, cmap="Blues", values_format=".2f", xticks_rotation=45, colorbar=False)
        fig.colorbar(display.im_, ax=ax, label="Anteil der wahren Klasse (Zeilensumme = 1)")
        ax.set_xlabel("Vorhergesagte Klasse")
        ax.set_ylabel("Wahre Klasse")
        ax.set_title("Confusion Matrix auf dem Test-Split (zeilennormiert)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    
    """
    Je Paar zwei bis drei fehlklassifizierte Bilder der dominanten Richtung.
    """
    def save_misclassified_examples(self):
        rng = np.random.default_rng(SEED)
        for rank, pair in enumerate(self.pairs, start=1):
            a, b = pair["dominant_ids"]
            candidates = np.where((self.y == a) & (self.pred == b))[0]       # Positionen in dataset
            chosen = np.sort(rng.choice(candidates, size=min(self.n_examples, len(candidates)), replace=False))
            pair["beispiel_indizes"] = [int(i) for i in chosen]
            pair["beispielbilder"] = os.path.join(self.out_dir, f"a1_fehlbilder_paar{rank}_klasse{a}_zu_klasse{b}.png")
            show_images(self.dataset, pair["beispiel_indizes"], predictions=self.pred,
                        path=pair["beispielbilder"], ncols=3)

    """
    Ausgabe auf der Konsole.
    """
    def print_report(self):
        print(f"\nTest-Accuracy: {self.accuracy:.4f}")
        print("F1 und Recall je Klasse:")
        for name, f1_value, recall in zip(CLASSES, self.f1, self.recall):
            print(f"  {name:<12s} F1 {f1_value:.4f}   Recall {recall:.4f}")
        print(f"Schwächste Klasse nach F1: {CLASSES[self.weakest_f1]} ({self.f1[self.weakest_f1]:.4f})")
        print(f"Schlechteste Diagonale (Recall): {self.worst_class['klasse']} ({self.worst_class['recall']:.4f}); "
              f"Fehler gehen an:")
        for target in self.worst_class["häufigste_fehlerziele"]:
            print(f"  {target['vorhergesagt_als']:<12s} {target['anteil']:.3f} ({target['anzahl']} Bilder)")
        print("Häufigste Verwechslungspaare:")
        for pair in self.pairs:
            name_i, name_j = pair["klassen"]
            examples = pair.get("beispiel_indizes", [])
            print(f"  {name_i} <-> {name_j}: {name_i}->{name_j} {pair['anzahl_i_zu_j']}, "
                  f"{name_j}->{name_i} {pair['anzahl_j_zu_i']} (dominant: {pair['dominante_richtung']}); ")

    """
    Kennzahlen der Testauswertung als JSON
    """
    def metrics(self):
        return {
            "accuracy": float(self.accuracy),
            "f1_je_klasse": {name: float(v) for name, v in zip(CLASSES, self.f1)},
            "f1_macro": float(np.mean(self.f1)),
            "schwächste_klasse_nach_f1": CLASSES[self.weakest_f1],
            "recall_je_klasse": {name: float(v) for name, v in zip(CLASSES, self.recall)},
            "schlechteste_klasse_nach_recall": self.worst_class,
            "häufigste_paare": self.pairs,
            "confusion_matrix_zählungen": self.counts.tolist(),
            "confusion_matrix_zeilennormiert": self.normalized.tolist(),
        }


"""
Sammelt Kennzahlen aller Aufgaben und schreibt sie als eine JSON-Datei in den Ausgabeordner.
"""
class MetricsStore:

    def __init__(self, path=METRICS_PATH):
        self.path = path
        self.data = {}

    def add(self, section, values):
        self.data.setdefault(section, {}).update(values)

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)



"""
t-SNE- und UMAP-Layouts des 64D-Embeddings, Qualitaetsmasse und Darstellungen.
"""
class LayoutStudy:
    def __init__(self, model, test_set, out_dir=OUT_DIR):
        sub = Subset(test_set, test_subsample())          # feste Stichprobe von 5.000 Testbildern
        self.y, self.pred, self.E = extract(model, sub)   # gleiche Reihenfolge wie sub; E nicht standardisiert
        self.out_dir = out_dir
        self.pca_reference = PCA(n_components=2, random_state=SEED).fit_transform(self.E)
        self.layouts = {}     # ausgerichtetes 2D-Layout, Form (n, 2)
        self.runtimes = {}    # Laufzeit der Projektion in s
        self.scores = {}      # {"T": [...], "C": [...]}, je ein Wert pro K in K_VALUES
        self.chosen = None    
        self.k_values = list(K_VALUES)

    @staticmethod
    def name(method, param):
        return f"tsne_perplexity_{param}" if method == "tsne" else f"umap_n_neighbors_{param}"

    """
    t-SNE (perplexity 5/30/100) und UMAP (n_neighbors 5/15/100) direkt auf dem 64D-Embedding.
    """
    def compute_layouts(self):
        for method, params in (("tsne", TSNE_PERPLEXITIES), ("umap", UMAP_N_NEIGHBORS)):
            for param in params:
                name = self.name(method, param)
                start = time.perf_counter()
                if method == "tsne":
                    reducer = TSNE(perplexity=param, init="pca", random_state=SEED)
                else:
                    reducer = umap.UMAP(n_neighbors=param, init="spectral", random_state=SEED)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="n_jobs value", category=UserWarning) # Warnung nach PDF Ingnorieren
                    layout = reducer.fit_transform(self.E)
                self.layouts[name] = align(layout, self.pca_reference)
                self.runtimes[name] = time.perf_counter() - start
                print(f"Layout {name}: {self.runtimes[name]:.1f} s")

    
    """
    Trustworthiness T(K) (falsche Nachbarn im Layout) und 
    Continuity C(K) (verlorene 64D-Nachbarn).
    """
    def evaluate(self, k_values=K_VALUES):
        self.k_values = list(k_values)
        for name, layout in self.layouts.items():
            self.scores[name] = {
                "T": [float(trustworthiness(self.E, layout, n_neighbors=k)) for k in self.k_values],
                # Layout und self.E vertauscht
                "C": [float(trustworthiness(layout, self.E, n_neighbors=k)) for k in self.k_values],
            }
            print(f"Qualitaet {name}: bewertet")

    """
    Ausgabe des Ergebnis in der Konsole
    """
    def print_scores(self):
        for measure, label in (("T", "Trustworthiness"), ("C", "Continuity")):
            print(f"\n{label}:")
            print(f"  {'Layout':<22s}" + "".join(f"K={k:<7d}" for k in self.k_values))
            for name, values in self.scores.items():
                print(f"  {name:<22s}" + "".join(f"{v:<9.4f}" for v in values[measure]))

    """
    Gewählte Layouts festlegen.
    """
    def choose(self, tsne_perplexity, umap_n_neighbors):
        for method, param in (("tsne", tsne_perplexity), ("umap", umap_n_neighbors)):
            if self.name(method, param) not in self.layouts:
                raise ValueError(f"Kein Layout {self.name(method, param)}; verfügbar: {sorted(self.layouts)}")
        self.chosen = {"tsne": tsne_perplexity, "umap": umap_n_neighbors}

    """(t-SNE-Layout, UMAP-Layout) für Aufgabe 3, beide ausgerichtet, Reihenfolge wie y, pred, E."""
    def chosen_layouts(self):
        return (self.layouts[self.name("tsne", self.chosen["tsne"])],
                self.layouts[self.name("umap", self.chosen["umap"])])

    @staticmethod
    def _class_colors():
        return np.array(plt.get_cmap("tab10").colors)       # eine Farbe je Klasse, Form (10, 3)

    @staticmethod
    def _class_handles(colors):
        return [Line2D([0], [0], marker="o", linestyle="", color=colors[i], markersize=7, label=CLASSES[i])
                for i in range(len(CLASSES))]

    @staticmethod
    def _style_axes(ax):
        ax.set_xlabel("Layout-Achse 1 (an PCA ausgerichtet)", fontsize=8)
        ax.set_ylabel("Layout-Achse 2", fontsize=8)
        ax.set_xticks([])      # Achsenwerte eines Layouts haben keine Bedeutung, nur Lage und Abstaende
        ax.set_yticks([])

    """
    2x3-Raster: eine Zeile je Verfahren (t-SNE, UMAP), eine Spalte je Parameter, Farbe = wahre Klasse.
    """
    def plot_parameter_study(self, path):
        colors = self._class_colors()
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        rows = (("tsne", TSNE_PERPLEXITIES, "t-SNE", "perplexity"), ("umap", UMAP_N_NEIGHBORS, "UMAP", "n_neighbors"))
        for row, (method, params, label, param_name) in zip(axes, rows):
            for ax, param in zip(row, params):
                layout = self.layouts[self.name(method, param)]
                ax.scatter(layout[:, 0], layout[:, 1], c=colors[self.y], s=4, alpha=0.7, linewidths=0)
                ax.set_title(f"{label}, {param_name} = {param}")
                self._style_axes(ax)
        fig.legend(handles=self._class_handles(colors), loc="lower center", ncol=5, title="Wahre Klasse",
                   frameon=False)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(path, dpi=150)
        plt.close(fig)

    
    """
    T(K) und C(K) über K, sechs Kurven je Diagramm.
    """
    def plot_quality(self, path):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        shades = (0.45, 0.7, 0.95)
        styles = {"tsne": ("Blues", "-", "o", "t-SNE, perplexity", TSNE_PERPLEXITIES),
                  "umap": ("Oranges", "--", "s", "UMAP, n_neighbors", UMAP_N_NEIGHBORS)}
        for ax, measure, title in zip(axes, ("T", "C"), ("Trustworthiness T(K)", "Continuity C(K)")):
            for method, (cmap, linestyle, marker, label, params) in styles.items():
                for shade, param in zip(shades, params):
                    ax.plot(self.k_values, self.scores[self.name(method, param)][measure], linestyle=linestyle,
                            marker=marker, ms=4, color=plt.get_cmap(cmap)(shade), label=f"{label} = {param}")
            ax.set_xscale("log")
            ax.set_xticks(self.k_values)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.minorticks_off()
            ax.set_xlabel("K (Anzahl geprüfter Nachbarn, logarithmisch)")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


    """
    Gewählte Layouts: alle Punkte blass, Fehlklassifikationen deckend, größer, mit dünnem schwarzem Rand,
    jeweils in der Farbe ihrer wahren Klasse.
    """
    def plot_chosen(self, path):
        colors = self._class_colors()
        wrong = self.pred != self.y
        tsne_layout, umap_layout = self.chosen_layouts()
        panels = ((tsne_layout, f"t-SNE, perplexity = {self.chosen['tsne']}"),
                  (umap_layout, f"UMAP, n_neighbors = {self.chosen['umap']}"))
        fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
        for ax, (layout, title) in zip(axes, panels):
            ax.scatter(layout[:, 0], layout[:, 1], c=colors[self.y], s=6, alpha=0.15, linewidths=0)
            ax.scatter(layout[wrong, 0], layout[wrong, 1], c=colors[self.y[wrong]], s=24, alpha=1.0,
                       edgecolors="black", linewidths=0.4, zorder=3)
            ax.set_title(f"{title}\n{int(wrong.sum())} Fehlklassifikationen von {len(self.y)} Testbildern")
            self._style_axes(ax)
        handles = self._class_handles(colors) + [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="lightgrey", markeredgecolor="black",
                   markeredgewidth=0.6, markersize=8, label="Fehlklassifikation (Farbe = wahre Klasse)")]
        fig.legend(handles=handles, loc="lower center", ncol=6, title="Wahre Klasse", frameon=False)
        fig.tight_layout(rect=(0, 0.11, 1, 1))
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def neighbor_statistics(self, space, k=PURITY_K):
        neighbors = NearestNeighbors(n_neighbors=k + 1).fit(space).kneighbors(space, return_distance=False)[:, 1:]
        neighbor_classes = self.y[neighbors]                                    # (n, k)
        purity = (neighbor_classes == self.y[:, None]).mean(axis=1)
        majority = np.stack([(neighbor_classes == c).sum(axis=1) for c in range(len(CLASSES))], axis=1).argmax(axis=1)
        wrong = self.pred != self.y
        return {
            "k": k,
            "reinheit_fehlklassifiziert": float(purity[wrong].mean()),
            "reinheit_richtig_klassifiziert": float(purity[~wrong].mean()),
            "anteil_mehrheit_eigene_klasse": float((majority[wrong] == self.y[wrong]).mean()),
            "anteil_mehrheit_vorhergesagte_klasse": float((majority[wrong] == self.pred[wrong]).mean()),
            "anteil_mehrheit_andere_klasse": float(((majority[wrong] != self.y[wrong])
                                                    & (majority[wrong] != self.pred[wrong])).mean()),
        }

    """
    Kennzahlen als JSON
    """        
    def metrics(self):
        wrong = self.pred != self.y
        result = {
            "stichprobe": {"n": int(len(self.y)), "n_fehlklassifiziert": int(wrong.sum()),
                           "accuracy_stichprobe": float(1 - wrong.mean())},
            "parameter": {"tsne_perplexity": list(TSNE_PERPLEXITIES), "umap_n_neighbors": list(UMAP_N_NEIGHBORS),
                          "K": list(self.k_values)},
            "layouts": {name: {"laufzeit_s": float(self.runtimes[name]),
                               "trustworthiness": {str(k): v for k, v in zip(self.k_values, s["T"])},
                               "continuity": {str(k): v for k, v in zip(self.k_values, s["C"])}}
                        for name, s in self.scores.items()},
        }
        if self.chosen is not None:
            tsne_layout, umap_layout = self.chosen_layouts()
            result["auswahl"] = {"tsne_perplexity": self.chosen["tsne"], "umap_n_neighbors": self.chosen["umap"]}
            result["nachbarschaft_fehlklassifikationen"] = {
                "tsne_layout": self.neighbor_statistics(tsne_layout),
                "umap_layout": self.neighbor_statistics(umap_layout),
                "embedding_64d": self.neighbor_statistics(self.E),
            }
        return result