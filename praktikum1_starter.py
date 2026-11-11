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



"""
IVDA 2 - Praktikum 1: Einstiegspunkt fuer Aufgabe 1.
 
    python praktikum1.py                        trainieren, best_model.pt und output/ schreiben
    python praktikum1.py --load best_model.pt   Training ueberspringen, alles aus dem gespeicherten Modell erzeugen

"""
if __name__ == "__main__":

    import argparse
    import os

    from torch.utils.data import DataLoader

    from classes import MODEL_PATH, OUT_DIR, Evaluation, LayoutStudy, MetricsStore, Trainer

    CHOSEN_TSNE_PERPLEXITY = 30     # möglich: 5, 30, 100
    CHOSEN_UMAP_N_NEIGHBORS = 15    # möglich: 5, 15, 100


    parser = argparse.ArgumentParser()
    parser.add_argument("--load", metavar="PFAD",
                        help="gespeichertes Modell laden (z. B. best_model.pt) und Training ueberspringen")
    args = parser.parse_args()

    train_set, val_set, test_set = get_data()
    val_loader = DataLoader(val_set, batch_size=512)

    model, train_loader, optimizer, scheduler = make_run(train_set, weight_decay=0.01)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {device}")

    # Sammelt Kennzahlen und speichert diese
    metrics = MetricsStore()
 
    # ---- Aufgabe 1.1: Training (oder gespeichertes Modell laden) ------------------------------------------------
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler)

    if args.load:
        trainer.load(args.load)
        print(f"Modell und Historie aus {args.load} geladen, Training uebersprungen.")
    else:
        trainer.fit()
        trainer.restore_best()
        trainer.save(MODEL_PATH)
        print(f"Gespeichert: {MODEL_PATH}")
    trainer.plot(os.path.join(OUT_DIR, "a1_lernkurven.png"))
    trainer.print_summary()
 
    # ---- Aufgabe 1.2 und 1.3: einmalige Testauswertung, Confusion Matrix, Fehlerpaare ---------------------------
    evaluation = Evaluation(model, test_set)
    evaluation.plot_confusion_matrix(os.path.join(OUT_DIR, "a1_confusion_matrix.png"))
    evaluation.save_misclassified_examples()
    evaluation.print_report()
 
    metrics.add("aufgabe1", {"training": trainer.metrics(), "test": evaluation.metrics()})
    metrics.save()

    
    # ---- Aufgabe 2: Dimensionsreduktion (5.000 Test-Embeddings, Layouts, Qualitaet, Auswahl) --------------------
    study = LayoutStudy(model, test_set)
    # 3 x t-SNE, 3 x UMAP
    study.compute_layouts()    

    # 2.1
    study.plot_parameter_study(os.path.join(OUT_DIR, "a2_parameterstudie.png"))

    # 2.2: T(K), C(K)
    study.evaluate()                                                                  
    study.plot_quality(os.path.join(OUT_DIR, "a2_trust_continuity.png"))
    study.print_scores()

    # 2.3
    study.choose(CHOSEN_TSNE_PERPLEXITY, CHOSEN_UMAP_N_NEIGHBORS)  

    # 2.4
    study.plot_chosen(os.path.join(OUT_DIR, "a2_gewaehlte_layouts.png"))              
    metrics.add("aufgabe2", study.metrics())
    metrics.save()


