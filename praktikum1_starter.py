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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import contingency_matrix
from hdbscan.validity import validity_index

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from clustering import Clustering
from visualization import Visualization
from dimensionality_reduction import DimensionalityReduction
from cnn_evaluation import CNNEvaluation
from result_logger import ResultLogger

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

def run_training(epochs, patience):
    counter = 0
    runtime = 0
    losses_train = []
    accs_train = []
    losses_val = []
    accs_val = []
    loss_i = 0
    acc_i =0
    best_val_loss_epoch = 0
    best_val_loss = float("inf")
    
    for i in range(epochs):
        print(f"==== Epoch {i+1} ==== \n")
        start = time.perf_counter()
        loss_i, acc_i = run_epoch(model, train_loader, optimizer)
        end = time.perf_counter()
        
        train_runtime = end - start
        
        losses_train.append(loss_i)
        accs_train.append(acc_i)
        print(f"## Training ({train_runtime:.2f} s) ## \n Loss: {loss_i:.4f} \n Accuracy: {acc_i:.4f}")
        runtime += end - start
        
        start = time.perf_counter()
        loss_i, acc_i = run_epoch(model, val_loader)
        end = time.perf_counter()
        
        val_runtime = end - start
        
        scheduler.step(loss_i)
        
        losses_val.append(loss_i)
        accs_val.append(acc_i)
        print(f"## Validation ({val_runtime:.2f} s) ## \n Loss: {loss_i:.4f} \n Accuracy: {acc_i:.4f}\n")
        
        runtime += end - start
        
        if loss_i < best_val_loss:
            best_val_loss = loss_i
            best_val_loss_epoch = i
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")

        else:
            counter += 1
            
        logger.add_epoch("ResNet18", i + 1, {
            "train_loss": losses_train[-1],
            "train_accuracy": accs_train[-1],
            "val_loss": losses_val[-1],
            "val_accuracy": accs_val[-1],
            "train_runtime": train_runtime,
            "val_runtime": val_runtime
        })
    
        if counter >= patience:
            print("Early Stopping!")
            epochs = i+1
            break
        
    
    print(f"Runtime: {runtime:.2f} s\n")
    
    logger.add("CNN", "ResNet18", {
        "runtime": runtime,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch + 1,
        "epochs_completed": len(losses_train)
    })
    
    result_dict = {
                    "epochs" : epochs,
                    "losses_val": losses_val,
                    "losses_train": losses_train,
                    "best_val_loss_epoch": best_val_loss_epoch,
                    "accs_val" : accs_val,
                    "accs_train" : accs_train
    }

    
    return result_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", metavar="PFAD",
                        help="gespeichertes Modell laden (z. B. best_model.pt) und Training ueberspringen")
    args = parser.parse_args()

    train_set, val_set, test_set = get_data()
    val_loader = DataLoader(val_set, batch_size=512)

    model, train_loader, optimizer, scheduler = make_run(train_set, weight_decay=0.01)
    
    vis = Visualization()
    logger = ResultLogger()
    cnn_eval = CNNEvaluation(logger)
    
    
    train = False
    ###################### Aufgabe 1 (CNN trainieren) ############################
    
    if train:
        result = run_training(epochs = 50, patience = 6)
        vis.plot_loss_accuracy(result["epochs"], result["losses_val"], result["losses_train"], result["best_val_loss_epoch"], result["accs_val"], result["accs_train"], save = True)
        
    test_loader = DataLoader(test_set, batch_size=512)
    model.load_state_dict(torch.load(r"./best_model.pth", map_location=device))
    all_labels, all_predictions, all_embeddings = extract(model, test_set)
    
    error_classes = cnn_eval.get_error_classes(all_labels, all_predictions, CLASSES)
    
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(conf_matrix)
    disp.plot()
    
    examples_num = 3
    example_ctr = 0
    for i in error_classes:
        for j in range(len(all_labels)):
            if all_labels[j] == i[0] and all_predictions[j] == i[1]:
                show_images(test_set, [j], path = f"results/error_example_{str(i)}_{str(example_ctr)}.png")
                example_ctr +=1
            if example_ctr >= examples_num:
                example_ctr = 0
                break
    
    #############################################################
    
    ########### Aufgabe 2 (Dimensionsreduktion) #################
    K = [5, 15, 30, 100, 500]
    perplexity = [5,30,100]
    neighbours = [5,15,100]
    
    sub = Subset(test_set, test_subsample())
    y, pred, E = extract(model, sub)
    dr = DimensionalityReduction(E)
    reference = PCA(n_components=2).fit_transform(E)
    
    trustworthiness_tsne = []
    continuity_tsne = []
    trustworthiness_umap = []
    continuity_umap = []
    tsne_layout_dict = dict()
    umap_layout_dict = dict()
    
    #tsne
    for p in range(len(perplexity)):
        layout_centered = dr.get_tsne_layout(reference, init='pca', random_state=0, perplexity=perplexity[p])
        tsne_layout_dict[str(perplexity[p])] = layout_centered
       
        trustworthiness_tsne.append(dr.get_trustworthiness(layout_centered, K))
        continuity_tsne.append(dr.get_continuity(layout_centered, K))
    
    #umap
    for n in range(len(neighbours)):
        layout_centered = dr.get_umap_layout(reference, init="spectral", random_state=0, n_neighbors = neighbours[n])
        umap_layout_dict[str(neighbours[n])] = layout_centered
        
        trustworthiness_umap.append(dr.get_trustworthiness(layout_centered, K))
        continuity_umap.append(dr.get_continuity(layout_centered, K))
    
    vis.plot_reduction_comparison(tsne_layout_dict, umap_layout_dict, y, CLASSES, save = True)
    vis.plot_reduction_evaluation(trustworthiness_tsne, continuity_tsne, trustworthiness_umap, continuity_umap, K, save = True)
    
    #====Final Layouts
    final_perplexity = 100
    final_n = 100
    
    tsne_layout_final = tsne_layout_dict[str(final_perplexity)]
    umap_layout_final = umap_layout_dict[str(final_n)]
    
    vis.plot_final_layouts(y, pred, tsne_layout_final, umap_layout_final, save = True)
    #############################################################
    
    ########### Aufgabe 3 (Clustering) ##########################
    
    clust = Clustering(E)
    
    
    epsilon = clust.get_epsilon_from_k_distances(n_neighbors=50, save_plot = True)
    
    labels_dict = { "K-Means" : clust.get_kmeans_labels(n_clusters=10, n_init=10, random_state=0),
                    "DBSCAN" : clust.get_dbscan_labels(eps=epsilon, min_samples=50),
                    "DBSCAN (0.5 x epsilon)" : clust.get_dbscan_labels(eps=0.5*epsilon, min_samples=50),
                    "HDBSCAN (eom)" : clust.get_hdbscan_labels(min_cluster_size=50, min_samples=10, cluster_selection_method="eom"),
                    "HDBSCAN (leaf)" : clust.get_hdbscan_labels(min_cluster_size=50, min_samples=10, cluster_selection_method="leaf")
    }
    
    vis.plot_clusters(tsne_layout_final, umap_layout_final, labels_dict, save=True)
    
    print("HDBSCAN (eom)")
    print(contingency_matrix(y, labels_dict["HDBSCAN (eom)"]))
    print("\n")
    print("HDBSCAN (leaf)")
    print(contingency_matrix(y, labels_dict["HDBSCAN (leaf)"]))
    print("\n")
    
    logger.add("Clustering", "HDBSCAN (eom)", {
        "contingency_matrix": contingency_matrix(y, labels_dict["HDBSCAN (eom)"])
    })

    logger.add("Clustering", "HDBSCAN (leaf)", {
        "contingency_matrix": contingency_matrix(y, labels_dict["HDBSCAN (leaf)"])
    })
    
    for key in labels_dict.keys():
        if key != "K-Means":

            print(key)
            print(validity_index(E.astype(np.float64), labels_dict[key]))
            print("\n")
            
            logger.add("Clustering", key, {
                "validity_index": validity_index(E.astype(np.float64), labels_dict[key])
            })
            
    logger.save()
    
    #############################################################

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
