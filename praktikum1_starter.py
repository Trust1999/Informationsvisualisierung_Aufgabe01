"""
IVDA 2 - Praktikum 1: Ueberwachtes & unueberwachtes Lernen auf Fashion-MNIST
Starter-Skript.

Fest vorgegeben (bitte nicht aendern): Daten, Split, Seed, Modell, Test-Stichprobe.
Alles laeuft mit EINEM Seed (SEED = 0): torch, numpy, t-SNE, UMAP, k-Means.
make_run() setzt den Seed fuer jeden Trainingslauf neu (gleiche Initialisierung und Batch-Reihenfolge).
Auf GPU/MPS sind Ergebnisse nicht bitgenau reproduzierbar, aber vergleichbar.
"""

import math
import time   # fuer time.perf_counter() (Laufzeitmessung in Aufgabe 1)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = 0
CLASSES = ["T-Shirt/Top", "Hose", "Pullover", "Kleid", "Mantel",
           "Sandale", "Hemd", "Sneaker", "Tasche", "Stiefelette"]

np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Daten (vorgegeben)
# ---------------------------------------------------------------------------
def get_data(root="./data"):
    """Train 50k / Val 10k (1.000 je Klasse) aus dem offiziellen Train-Split, Test 10k."""
    # Standardisieren: x -> (x - 0.2860) / 0.3530 (Mittelwert und Std der Pixel im offiziellen Train-Split),
    # danach haben die Pixel etwa Mittelwert 0 und Std 1
    normalize = transforms.Normalize((0.2860,), (0.3530,))
    transform = transforms.Compose([transforms.ToTensor(), normalize])  # ToTensor: uint8 0-255 -> float [0, 1]

    official_train = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
    official_test = datasets.FashionMNIST(root, train=False, download=True, transform=transform)

    # Offiziellen Train-Split (6.000 je Klasse) je Klasse aufteilen: 1.000 in Val, 5.000 in Train
    generator = np.random.default_rng(SEED)
    class_labels = official_train.targets.numpy()
    train_indices, val_indices = [], []
    for class_id in range(10):
        class_indices = generator.permutation(np.where(class_labels == class_id)[0])
        val_indices += class_indices[:1000].tolist()
        train_indices += class_indices[1000:].tolist()
    return Subset(official_train, train_indices), Subset(official_train, val_indices), official_test


def test_subsample(test_size=10_000, n_samples=5000):
    """Feste Stichprobe des Test-Splits fuer Aufgabe 2 und 3 (fuer alle Gruppen gleich).
    Gibt n_samples Indizes in test_set zurueck, Verwendung: sub = Subset(test_set, test_subsample())."""
    return np.random.default_rng(SEED).choice(test_size, n_samples, replace=False)


# ---------------------------------------------------------------------------
# Modell (vorgegeben)
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """3 x (Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2) -> Flatten -> FC 64 (Embedding) -> Dropout -> FC 10 (Logits)."""

    def __init__(self, embedding_dim=64, dropout=0.3, num_classes=10):
        super().__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                 nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(
            conv_block(1, 16),     # 28x28x1  -> 14x14x16
            conv_block(16, 32),    # 14x14x16 -> 7x7x32
            conv_block(32, 64),    # 7x7x32   -> 3x3x64
            nn.Flatten())          # -> 576
        self.embed = nn.Sequential(nn.Linear(576, embedding_dim), nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, images, return_embedding=False):
        embedding = self.embed(self.features(images))   # vorletzte Schicht = Embedding (64D)
        logits = self.head(self.dropout(embedding))     # Dropout auf das Embedding, dann FC 64->10; Softmax steckt in F.cross_entropy
        return (logits, embedding) if return_embedding else logits


# ---------------------------------------------------------------------------
# Hilfsfunktionen (vorgegeben)
# ---------------------------------------------------------------------------
def make_run(train_set, weight_decay=0.01):
    """Neuer Trainingslauf mit festem Seed: gleiche Initialisierung und Batch-Reihenfolge fuer jeden Lauf.
    Weight Decay wirkt auf alle Parameter (inkl. Bias und BatchNorm)."""
    torch.manual_seed(SEED)
    model = SmallCNN().to(device)
    shuffle_generator = torch.Generator().manual_seed(SEED)   # feste Batch-Reihenfolge
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, generator=shuffle_generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    # patience=2: LR-Reduktion nach der 3. Epoche in Folge ohne Verbesserung; scheduler.step(val_loss) je Epoche.
    # Als Verbesserung zaehlt hier ein relativer Rueckgang um mehr als 1e-4 (Default threshold);
    # ein Early Stopping mit einfachem "<" kann im Grenzfall anders zaehlen.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=2)
    return model, train_loader, optimizer, scheduler


def run_epoch(model, loader, optimizer=None):
    """Eine Epoche Training (optimizer gesetzt) oder Evaluation (optimizer=None). Gibt (Loss, Accuracy) zurueck.
    Beide Werte sind Mittel ueber alle Bilder der Epoche (Batch-Loss mit Batchgroesse gewichtet).
    Training: Train-Modus (Dropout aktiv), Werte entstehen waehrend sich die Gewichte aendern. Evaluation: eval-Modus."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss, n_correct, n_samples = 0.0, 0, 0
    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)   # Mittel ueber den Batch
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += len(labels)
    return total_loss / n_samples, n_correct / n_samples


def align(layout, reference):
    """Procrustes: 2D-Layout zentrieren und so drehen/spiegeln, dass es moeglichst gut auf reference passt
    (z. B. reference = PCA(n_components=2).fit_transform(E)). Keine Skalierung, Abstaende im Layout bleiben gleich."""
    from scipy.linalg import orthogonal_procrustes
    layout_centered = layout - layout.mean(0)
    reference_centered = reference - reference.mean(0)
    rotation, _ = orthogonal_procrustes(layout_centered, reference_centered)   # orthogonale 2x2-Matrix
    return layout_centered @ rotation   # @ = Matrixmultiplikation (wie np.matmul): jeder Punkt wird gedreht


@torch.no_grad()
def extract(model, dataset):
    """Labels, Vorhersagen und 64D-Embeddings fuer einen Datensatz (eval-Modus).
    Reihenfolge wie in dataset (DataLoader ohne shuffle): Zeile i gehoert zu dataset[i]."""
    model.eval()
    all_labels, all_predictions, all_embeddings = [], [], []
    for images, labels in DataLoader(dataset, batch_size=512):
        logits, embedding = model(images.to(device), return_embedding=True)
        all_labels.append(labels)
        all_predictions.append(logits.argmax(1).cpu())
        all_embeddings.append(embedding.cpu())
    return (torch.cat(all_labels).numpy(), torch.cat(all_predictions).numpy(),
            torch.cat(all_embeddings).numpy())


def show_images(dataset, indices, predictions=None, path=None, ncols=6):
    """Zeigt die Bilder dataset[i] fuer alle i in indices (dataset auch als Subset) mit wahrer Klasse im Titel.
    indices sind Positionen in dataset (bei sub also 0-4999), keine Indizes im Test-Split; '#i' im Titel ebenso.
    predictions (optional): Vorhersagen in derselben Reihenfolge wie dataset (z. B. aus extract).
    path gesetzt: Abbildung als Datei speichern, sonst anzeigen."""
    import matplotlib.pyplot as plt
    indices = list(indices)
    nrows = max(1, math.ceil(len(indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, indices):
        # Bei einem Subset auf den zugrunde liegenden Datensatz und den Originalindex zurueckgehen,
        # um an das Rohbild (ohne Normalisierung) und das Label zu kommen
        if isinstance(dataset, Subset):
            base_dataset, base_index = dataset.dataset, dataset.indices[i]
        else:
            base_dataset, base_index = dataset, i
        ax.imshow(base_dataset.data[base_index].numpy(), cmap="gray")   # Rohbild (uint8), nicht normalisiert
        title = f"#{i} wahr: {CLASSES[int(base_dataset.targets[base_index])]}"
        if predictions is not None:
            title += f"\nvorh.: {CLASSES[int(predictions[i])]}"
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

# ---------------------------------------------------------------------------
# Aufgabe 1: CNN trainieren, Test-Auswertung, Confusion Matrix
# ---------------------------------------------------------------------------
import json
import os
 
import matplotlib
matplotlib.use("Agg")   # alle Abbildungen werden als PNG gespeichert, es wird kein Fenster gebraucht
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
 
OUT_DIR = "output"                                    # Ausgabeordner fuer alle Abbildungen und Kennzahlen
STATE_PATH = os.path.join(OUT_DIR, "best_state.pt")   # bester Zustand waehrend des Trainings (Zwischendatei)
MODEL_PATH = "best_model.pt"                          # Abgabedatei: state_dict + Trainingshistorie
METRICS_PATH = os.path.join(OUT_DIR, "kennzahlen.json")
MAX_EPOCHS = 50
ES_PATIENCE = 6
 
 
def train(model, train_loader, val_loader, optimizer, scheduler,
          max_epochs=MAX_EPOCHS, patience=ES_PATIENCE, state_path=STATE_PATH):

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "lr": [], "epoch_time": []}
    
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
 
    training_start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]                     # Lernrate, mit der diese Epoche laeuft
        train_loss, train_acc = run_epoch(model, train_loader, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader)
        epoch_time = time.perf_counter() - epoch_start
        scheduler.step(val_loss)                             

        # History beschreiben
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)
        history["epoch_time"].append(epoch_time)
 
        improved = val_loss < best_val_loss                      # neues Minimum des Val-Loss

        if improved:
            best_val_loss, best_epoch = val_loss, epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), state_path)
        else:
            epochs_without_improvement += 1

        # Ausgabe aktuellen History
        print(f"Epoche {epoch:2d}/{max_epochs} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | "
              f"Learning Rate {lr:.2e} | {epoch_time:5.1f} s" + ("  ist bestes Modell" if improved else ""))
 
        if epochs_without_improvement >= patience:
            stopped_early = True
            print(f"Early Stopping: {patience} Epochen in Folge ohne neues Minimum des Val-Loss.")
            break
 
    history["total_time"] = time.perf_counter() - training_start   # Gesamtlaufzeit des Trainings in s
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    history["epochs_run"] = len(history["train_loss"])
    history["stopped_early"] = stopped_early
    return history
 

# Loss (Verlust) und Accuracy (Genauigkeit) für Training und Validierung über die Epochen, Epoche des gespeicherten Modells markiert.
def plot_learning_curves(history, path):
    n_epochs = len(history["train_loss"])
    epochs = np.arange(1, n_epochs + 1)
    best_epoch = int(history["best_epoch"])
    Learning_rate_drops = [e for e in range(2, n_epochs + 1) if history["lr"][e - 1] < history["lr"][e - 2] * 0.999]
 
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, key, ylabel in zip(axes, ("loss", "acc"), ("Loss", "Accuracy")):
        ax.plot(epochs, history[f"train_{key}"], marker="o", ms=3, label="Training (Train-Modus, Dropout aktiv)")
        ax.plot(epochs, history[f"val_{key}"], marker="o", ms=3, label="Validierung (eval-Modus)")
        ax.axvline(best_epoch, color="black", ls="--", lw=1, label=f"gespeichertes Modell (Epoche {best_epoch})")
        ax.scatter([best_epoch], [history[f"val_{key}"][best_epoch - 1]], color="black", zorder=5, s=35)
        for k, e in enumerate(Learning_rate_drops):
            ax.axvline(e - 0.5, color="grey", ls=":", lw=1, label="Lernrate × 0,3" if k == 0 else None)
        ax.set_xlabel("Epoche")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_title("Loss (Verlust) über die Epochen")
    axes[1].set_title("Accuracy (Genauigkeit) über die Epochen")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
 

# Zeilennormierte Confusion Matrix: Zeile = wahre Klasse, Spalte = Vorhersage, Diagonale = Recall.
def plot_confusion_matrix(normalized, path):
    fig, ax = plt.subplots(figsize=(9.5, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=normalized, display_labels=CLASSES)
    display.plot(ax=ax, cmap="Blues", values_format=".2f", xticks_rotation=45, colorbar=False)
    fig.colorbar(display.im_, ax=ax, label="Anteil der wahren Klasse (Zeilensumme = 1)")
    ax.set_xlabel("Vorhergesagte Klasse")
    ax.set_ylabel("Wahre Klasse")
    ax.set_title("Confusion Matrix auf dem Test-Split (zeilennormiert)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
 

# Schlechteste Klasse (Diagonale), ihre häufigsten Fehlerziele und die häufigsten Verwechslungspaare.
def analyse_confusion(y, pred, n_pairs=3):
    counts = confusion_matrix(y, pred)                          # Rohzaehlungen: Zeile = wahr, Spalte = vorhergesagt
    normalized = confusion_matrix(y, pred, normalize="true")    # zeilennormiert
    recall = np.diag(normalized)
 
    worst = int(np.argmin(recall))
    row = normalized[worst].copy()
    row[worst] = 0.0
    worst_class = {
        "klasse": CLASSES[worst],
        "recall": float(recall[worst]),
        "haeufigste_fehlerziele": [
            {"vorhergesagt_als": CLASSES[j], "anteil": float(row[j]), "anzahl": int(counts[worst, j])}
            for j in np.argsort(row)[::-1][:3] if row[j] > 0],
    }
 
    pair_totals = sorted(((int(counts[i, j] + counts[j, i]), i, j)
                          for i in range(10) for j in range(i + 1, 10)), reverse=True)
    pairs = []
    for total, i, j in pair_totals[:n_pairs]:
        n_ij, n_ji = int(counts[i, j]), int(counts[j, i])
        a, b = (i, j) if n_ij >= n_ji else (j, i)               # dominante Richtung a -> b
        pairs.append({
            "klassen": [CLASSES[i], CLASSES[j]],
            "anzahl_i_zu_j": n_ij, "anzahl_j_zu_i": n_ji, "summe": total,
            "rate_i_zu_j": float(normalized[i, j]), "rate_j_zu_i": float(normalized[j, i]),
            "dominante_richtung": f"{CLASSES[a]} -> {CLASSES[b]}",
            "dominant_ids": [int(a), int(b)],
        })
    return counts, normalized, worst_class, pairs
 

# Je Paar zwei bis drei fehlklassifizierte Testbilder (dominante Richtung), reproduzierbar per Seed gewaehlt.
def save_misclassified_examples(test_set, y, pred, pairs, n_examples=3):
    rng = np.random.default_rng(SEED)
    for rank, pair in enumerate(pairs, start=1):
        a, b = pair["dominant_ids"]
        candidates = np.where((y == a) & (pred == b))[0]        # Positionen in test_set (= Indizes im Test-Split)
        chosen = np.sort(rng.choice(candidates, size=min(n_examples, len(candidates)), replace=False))
        pair["beispiel_indizes"] = [int(i) for i in chosen]
        pair["beispielbilder"] = os.path.join(OUT_DIR, f"a1_fehlbilder_paar{rank}_klasse{a}_zu_klasse{b}.png")
        show_images(test_set, pair["beispiel_indizes"], predictions=pred, path=pair["beispielbilder"], ncols=3)
 
 
def save_metrics(metrics, path=METRICS_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
 
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", metavar="PFAD",
                            help="gespeichertes Modell laden (z. B. best_model.pt) und Training ueberspringen")
    args = parser.parse_args()
    
    train_set, val_set, test_set = get_data()
    val_loader = DataLoader(val_set, batch_size=512)
    
    model, train_loader, optimizer, scheduler = make_run(train_set, weight_decay=0.01)
 
    # ---- Aufgabe 1.1: Training (oder gespeichertes Modell laden) -----------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {device}")

    # Model Laden
    if args.load:
        ckpt = torch.load(args.load, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        history = ckpt["history"]
        print(f"Modell und Historie aus {args.load} geladen, Training uebersprungen.")
    # Model trainieren
    else:
        history = train(model, train_loader, val_loader, optimizer, scheduler)
        model.load_state_dict(torch.load(STATE_PATH, map_location=device))
        torch.save({"state_dict": model.state_dict(), "history": history}, MODEL_PATH)
        print(f"Gespeichert: {MODEL_PATH}")
 
    best_epoch = history["best_epoch"]
    plot_learning_curves(history, os.path.join(OUT_DIR, "a1_lernkurven.png"))
    print(f"\nBestes Modell: Epoche {best_epoch} von {len(history['train_loss'])} "
          f"(Val Loss {history['val_loss'][best_epoch - 1]:.4f}, Val Acc {history['val_acc'][best_epoch - 1]:.4f}, "
          f"Train Loss {history['train_loss'][best_epoch - 1]:.4f}, Train Acc {history['train_acc'][best_epoch - 1]:.4f})")
    print(f"Laufzeit: gesamt {history['total_time']:.1f} s, "
          f"im Mittel {float(np.mean(history['epoch_time'])):.1f} s je Epoche")
 
    metrics = {"aufgabe1": {"training": {
        "beste_epoche": int(best_epoch),
        "epochen_gelaufen": len(history["train_loss"]),
        "early_stopping_ausgeloest": bool(history["stopped_early"]),
        "bester_val_loss": float(history["val_loss"][best_epoch - 1]),
        "val_acc_beste_epoche": float(history["val_acc"][best_epoch - 1]),
        "train_loss_beste_epoche": float(history["train_loss"][best_epoch - 1]),
        "train_acc_beste_epoche": float(history["train_acc"][best_epoch - 1]),
        "gesamtlaufzeit_s": float(history["total_time"]),
        "laufzeit_je_epoche_s": [float(t) for t in history["epoch_time"]],
        "mittlere_laufzeit_je_epoche_s": float(np.mean(history["epoch_time"])),
        "lernrate_je_epoche": [float(v) for v in history["lr"]],
        "train_loss": [float(v) for v in history["train_loss"]],
        "train_acc": [float(v) for v in history["train_acc"]],
        "val_loss": [float(v) for v in history["val_loss"]],
        "val_acc": [float(v) for v in history["val_acc"]],
    }}}
 
    # ---- Aufgabe 1.2: einmalige Testauswertung --------------------
    y_test, pred_test, _ = extract(model, test_set)
    accuracy = accuracy_score(y_test, pred_test)
    f1_per_class = f1_score(y_test, pred_test, average=None) 
    weakest = int(np.argmin(f1_per_class))

    counts, normalized, worst_class, pairs = analyse_confusion(y_test, pred_test)
     
    print(f"\nTest-Accuracy: {accuracy:.4f}")
    print("F1 und Recall je Klasse:")
    for name, f1_value, rec in zip(CLASSES, f1_per_class, np.diag(normalized)):
        print(f"  {name:<12s} F1 {f1_value:.4f}   Recall {rec:.4f}")

    print(f"Schwaechste Klasse nach F1: {CLASSES[weakest]} ({f1_per_class[weakest]:.4f})")
 
    # ---- Aufgabe 1.3: Confusion Matrix, Fehlerziele, Paare, Beispielbilder -------------------------------------
    plot_confusion_matrix(normalized, os.path.join(OUT_DIR, "a1_confusion_matrix.png"))
    save_misclassified_examples(test_set, y_test, pred_test, pairs)
    
    print(f"Schlechteste Diagonale (Recall): {worst_class['klasse']} ({worst_class['recall']:.4f}); Fehler gehen an:")
    for target in worst_class["haeufigste_fehlerziele"]:
        print(f"  {target['vorhergesagt_als']:<12s} {target['anteil']:.3f} ({target['anzahl']} Bilder)")
    print("Haeufigste Verwechslungspaare:")
    for pair in pairs:
        name_i, name_j = pair["klassen"]
        print(f"  {name_i} <-> {name_j}: {name_i}->{name_j} {pair['anzahl_i_zu_j']}, "
              f"{name_j}->{name_i} {pair['anzahl_j_zu_i']} (dominant: {pair['dominante_richtung']}); ")
 
    metrics["aufgabe1"]["test"] = {
        "accuracy": float(accuracy),
        "f1_je_klasse": {name: float(v) for name, v in zip(CLASSES, f1_per_class)},
        "f1_macro": float(np.mean(f1_per_class)),
        "schwaechste_klasse_nach_f1": CLASSES[weakest],
        "recall_je_klasse": {name: float(v) for name, v in zip(CLASSES, np.diag(normalized))},
        "schlechteste_klasse_nach_recall": worst_class,
        "haeufigste_paare": pairs,
        "confusion_matrix_zaehlungen": counts.tolist(),
        "confusion_matrix_zeilennormiert": normalized.tolist(),
    }
    save_metrics(metrics)
