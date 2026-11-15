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

# Aufgabe 1

def plot_learning_curves(history, output_dir):
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = history["best_epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, train_key, val_key, ylabel, title in [
        (axes[0], "train_loss", "val_loss", "Cross-Entropy-Loss", "Loss"),
        (axes[1], "train_accuracy", "val_accuracy", "Accuracy", "Accuracy"),
    ]:
        ax.plot(epochs, history[train_key], label="Training")
        ax.plot(epochs, history[val_key], label="Validierung")
        ax.axvline(best_epoch, color="black", linestyle="--",
                   label=f"Bestes Modell (Epoche {best_epoch})")
        ax.set(xlabel="Epoche", ylabel=ylabel, title=title)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "1_lernkurven.png", dpi=160)
    plt.close(fig)


def evaluate_test(model, test_set, output_dir):
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

    # Einmalige finale Testauswertung; danach keine Gewichte mehr veraendern.
    y_true, y_pred, _ = extract(model, test_set)
    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = f1_score(y_true, y_pred, labels=list(range(10)), average=None)
    cm_counts = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    class_accuracy = np.divide(np.diag(cm_counts), cm_counts.sum(axis=1),
                               out=np.zeros(10, dtype=float),
                               where=cm_counts.sum(axis=1) != 0)
    print(f"\nTest-Accuracy: {accuracy:.4f}")
    print("Accuracy und F1 je Klasse:")
    for name, acc, score in zip(CLASSES, class_accuracy, f1):
        print(f"  {name:15s}: Accuracy={acc:.4f}, F1={score:.4f}")
    weakest = int(np.argmin(f1))
    print(f"Schwaechste Klasse nach F1: {CLASSES[weakest]}")

    cm_normalized = confusion_matrix(y_true, y_pred, labels=list(range(10)), normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    display = ConfusionMatrixDisplay(cm_normalized, display_labels=CLASSES)
    display.plot(ax=ax, cmap="Blues", values_format=".2f", xticks_rotation=45,
                 colorbar=True)
    ax.set(title="Confusion Matrix (zeilennormiert)",
           xlabel="Vorhergesagte Klasse", ylabel="Wahre Klasse")
    fig.tight_layout()
    fig.savefig(output_dir / "1_confusion_matrix.png", dpi=160)
    plt.close(fig)

    worst_recall = int(np.argmin(np.diag(cm_normalized)))
    print(f"Niedrigste Erkennungsrate (Diagonale): {CLASSES[worst_recall]} "
          f"({cm_normalized[worst_recall, worst_recall]:.3f})")

    # Paare i,j nach Fehlern i->j PLUS j->i sortieren.
    pairs = sorted(
        ((int(cm_counts[i, j] + cm_counts[j, i]), i, j)
         for i in range(10) for j in range(i + 1, 10)),
        reverse=True,
    )[:3]
    print("\nDrei haeufigste Verwechslungspaare (ungerichtet):")
    for rank, (total, i, j) in enumerate(pairs, start=1):
        print(f"  {rank}. {CLASSES[i]} <-> {CLASSES[j]}: "
              f"{total} insgesamt ({i}->{j}: {cm_counts[i,j]}, "
              f"{j}->{i}: {cm_counts[j,i]})")
        
        forward = np.flatnonzero((y_true == i) & (y_pred == j)).tolist()
        backward = np.flatnonzero((y_true == j) & (y_pred == i)).tolist()
        indices = []
        if forward:
            indices.append(forward.pop(0))
        if backward:
            indices.append(backward.pop(0))
        indices.extend((forward + backward)[:3 - len(indices)])
        if indices:
            show_images(test_set, indices, predictions=y_pred,
                        path=str(output_dir / f"1_verwechslung_paar_{rank}.png"),
                        ncols=3)

    results = {
        "accuracy": accuracy,
        "accuracy_per_class": {name: float(score) for name, score in zip(CLASSES, class_accuracy)},
        "f1_per_class": {name: float(score) for name, score in zip(CLASSES, f1)},
        "weakest_f1_class": CLASSES[weakest],
        "worst_recall_class": CLASSES[worst_recall],
        "three_most_common_pairs": [
            {"classes": [CLASSES[i], CLASSES[j]], "total": total,
             "i_to_j": int(cm_counts[i, j]), "j_to_i": int(cm_counts[j, i])}
            for total, i, j in pairs
        ],
    }
    return results


# Aufgabe 2

def class_scatter(ax, Y, y, title, pred=None):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    for i in range(10):
        mask = (y == i)
        ax.scatter(Y[mask, 0], Y[mask, 1], s=6, color=cmap(i),
                   alpha=0.15 if pred is not None else 0.7,
                   linewidths=0, rasterized=True)
    if pred is not None:
        wrong = y != pred
        for i in range(10):
            mask = wrong & (y == i)
            ax.scatter(Y[mask, 0], Y[mask, 1], s=25, color=cmap(i),
                       edgecolors="black", linewidths=0.4,
                       alpha=1, rasterized=True)
    ax.set(title=title, xlabel="Dimension 1", ylabel="Dimension 2")


def class_legend(fig, bottom=0.09):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    cmap = plt.get_cmap("tab10")
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=7,
                      color=cmap(i), label=label)
               for i, label in enumerate(CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, bottom, 1, 1))


def run_dimensions(model, test_set, output_dir, selected_tsne, selected_umap):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE, trustworthiness
    import umap

    sub = Subset(test_set, test_subsample())
    y, pred, E = extract(model, sub)
    assert E.shape == (5000, 64), f"Unerwartete Embedding-Form: {E.shape}"
    reference = PCA(n_components=2).fit_transform(E)
    layouts = {}

    for perplexity in (5, 30, 100):
        name = f"tSNE_{perplexity}"
        print(f"Aufgabe 2: Berechne {name} ...", flush=True)
        raw = TSNE(n_components=2, perplexity=perplexity,
                   init="pca", random_state=SEED).fit_transform(E)
        layouts[name] = align(raw, reference)

    for neighbors in (5, 15, 100):
        name = f"UMAP_{neighbors}"
        print(f"Aufgabe 2: Berechne {name} ...", flush=True)
        raw = umap.UMAP(n_components=2, n_neighbors=neighbors,
                        init="spectral", random_state=SEED).fit_transform(E)
        layouts[name] = align(raw, reference)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, (name, Y) in zip(axes.flat, layouts.items()):
        class_scatter(ax, Y, y, name)
    class_legend(fig)
    fig.savefig(output_dir / "2_layouts.png", dpi=160)
    plt.close(fig)

    ks = (5, 15, 30, 100, 500)
    quality = {}
    print("Aufgabe 2: Trustworthiness / Continuity berechnen ...", flush=True)
    for name, Y in layouts.items():
        quality[name] = {}
        for k in ks:
            t = float(trustworthiness(E, Y, n_neighbors=k))
            c = float(trustworthiness(Y, E, n_neighbors=k))
            quality[name][str(k)] = {"trustworthiness": t, "continuity": c}
            print(f"  {name} K={k}: T={t:.4f}; C={c:.4f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)
    tsne_colors = {5: "#9ecae1", 30: "#3182bd", 100: "#08519c"}
    umap_colors = {5: "#fdd0a2", 15: "#f16913", 100: "#a63603"}
    for ax, metric_key, title in (
        (axes[0], "trustworthiness", "Trustworthiness"),
        (axes[1], "continuity", "Continuity"),
    ):
        for perplexity, color in tsne_colors.items():
            name = f"tSNE_{perplexity}"
            values = [quality[name][str(k)][metric_key] for k in ks]
            ax.plot(ks, values, color=color, linestyle="-", marker="o",
                    label=f"t-SNE (perplexity={perplexity})")
        for neighbors, color in umap_colors.items():
            name = f"UMAP_{neighbors}"
            values = [quality[name][str(k)][metric_key] for k in ks]
            ax.plot(ks, values, color=color, linestyle="--", marker="s",
                    label=f"UMAP (n_neighbors={neighbors})")
        ax.set_xscale("log")
        ax.set_xticks(ks, labels=[str(k) for k in ks])
        ax.set(xlabel="K (Anzahl Nachbarn)", ylabel=title, title=title)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.24, 1, 1))
    fig.savefig(output_dir / "2_trustworthiness_continuity.png", dpi=160)
    plt.close(fig)

    if selected_tsne not in (5, 30, 100):
        raise ValueError(f"Ungueltige t-SNE-Perplexity: {selected_tsne}")
    if selected_umap not in (5, 15, 100):
        raise ValueError(f"Ungueltiges UMAP-n_neighbors: {selected_umap}")
    selected_tsne_name = f"tSNE_{selected_tsne}"
    selected_umap_name = f"UMAP_{selected_umap}"
    selected = {selected_tsne_name: layouts[selected_tsne_name],
                selected_umap_name: layouts[selected_umap_name]}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (name, Y) in zip(axes, selected.items()):
        class_scatter(ax, Y, y, f"{name} (Fehler hervorgehoben)", pred=pred)
    class_legend(fig, bottom=0.15)
    fig.savefig(output_dir / "2_fehlklassifikationen.png", dpi=160)
    plt.close(fig)

    return {
        "sample_size": int(len(sub)),
        "embedding_dimension": int(E.shape[1]),
        "quality_by_layout_and_k": quality,
        "selected_layouts": {"tsne": selected_tsne_name, "umap": selected_umap_name},
    }

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", metavar="PFAD",
                        help="gespeichertes Modell laden (z. B. best_model.pt) und Training ueberspringen")
    args = parser.parse_args()

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    train_set, val_set, test_set = get_data()
    val_loader = DataLoader(val_set, batch_size=512)

    model, train_loader, optimizer, scheduler = make_run(train_set, weight_decay=0.01)

    if args.load:
        checkpoint = torch.load(args.load, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        history = checkpoint["history"]
        print(f"Modell und Trainingshistorie geladen: {args.load}", flush=True)
    else:
        history = {"train_loss": [], "train_accuracy": [], "val_loss": [],
                   "val_accuracy": [], "epoch_seconds": []}
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        best_state_path = output_dir / "best_state.pt"
        total_start = time.perf_counter()
        for epoch in range(1, 51):
            epoch_start = time.perf_counter()
            train_loss, train_acc = run_epoch(model, train_loader, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader)
            scheduler.step(val_loss)
            epoch_seconds = time.perf_counter() - epoch_start
            for key, value in (("train_loss", train_loss), ("train_accuracy", train_acc),
                               ("val_loss", val_loss), ("val_accuracy", val_acc),
                               ("epoch_seconds", epoch_seconds)):
                history[key].append(float(value))
            if val_loss < best_val_loss:
                best_val_loss, best_epoch = val_loss, epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_state_path)
            else:
                epochs_without_improvement += 1
            print(f"Epoche {epoch:02d} | Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | "
                  f"Val Loss {val_loss:.4f}, Acc {val_acc:.4f} | "
                  f"{epoch_seconds:.1f}s | ohne Verbesserung: {epochs_without_improvement}/6",
                  flush=True)
            if epochs_without_improvement >= 6:
                print("Early Stopping", flush=True)
                break
        history["total_seconds"] = float(time.perf_counter() - total_start)
        history["best_epoch"] = int(best_epoch)
        history["best_val_loss"] = float(best_val_loss)
        model.load_state_dict(torch.load(best_state_path, map_location=device, weights_only=True))
        torch.save({"state_dict": model.state_dict(), "history": history}, "best_model.pt")
        print(f"Bestes Modell: Epoche {best_epoch}; best_model.pt gespeichert", flush=True)

    # Auswahl der beiden Layouts fuer die Fehlerdarstellung (Aufgabe 2.4).
    # t-SNE: Perplexity (erlaubt: 5, 30 oder 100).
    selected_tsne = 100
    # UMAP: n_neighbors (erlaubt: 5, 15 oder 100).
    selected_umap = 15

    plot_learning_curves(history, output_dir)
    test_metrics = evaluate_test(model, test_set, output_dir)  # Test einmalig auswerten
    dimension_metrics = run_dimensions(model, test_set, output_dir, selected_tsne, selected_umap)

    # Kennzahlen speichern
    all_metrics = {
        "aufgabe_1": {"training": history, "test": test_metrics},
        "aufgabe_2": dimension_metrics,
        "aufgabe_3": {},
    }
    (output_dir / "kennzahlen.json").write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAlle Ergebnisse in {output_dir.resolve()}", flush=True)

    # TODO: max. 50 Epochen; je Epoche run_epoch (Train/Val), scheduler.step(val_loss);
    #       Early Stopping mit Patience 6; besten Zustand bei jeder Verbesserung per
    #       torch.save(model.state_dict(), pfad) in eine eigene Datei speichern (z. B. pfad = "best_state.pt")
    #       und nach dem Training zurueckladen: model.load_state_dict(torch.load(pfad, map_location=device)).
    # TODO: ohne --load: nach dem Zurueckladen (sonst landen die Gewichte der letzten Epoche in der Datei) speichern, z. B.
    #           torch.save({"state_dict": model.state_dict(), "history": history}, "best_model.pt")
    #       mit --load: Training ueberspringen und laden:
    #           ckpt = torch.load(args.load, map_location=device)
    #           model.load_state_dict(ckpt["state_dict"]); history = ckpt["history"]
    #       history nur mit Python-Listen/-Zahlen fuellen: run_epoch und time.perf_counter liefern schon
    #       Python-floats, NumPy-Werte umwandeln (z. B. beste Epoche: int(np.argmin(...)));
    #       torch.load laedt standardmaessig mit weights_only=True und lehnt NumPy-Objekte ab.
    # Danach: einmalige Testauswertung, dann Aufgabe 2 und 3 mit diesem Modell.
