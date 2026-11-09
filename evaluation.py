import os
import matplotlib
matplotlib.use("Agg")   # alle Abbildungen werden als PNG gespeichert, es wird kein Fenster gebraucht
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score

from praktikum1_starter import CLASSES, SEED, extract, show_images

OUT_DIR = "output"                                    # Ausgabeordner fuer alle Abbildungen und Kennzahlen
STATE_PATH = os.path.join(OUT_DIR, "best_state.pt")   # bester Zustand waehrend des Trainings (Zwischendatei)
MODEL_PATH = "best_model.pt"                          # Abgabedatei: state_dict + Trainingshistorie
MAX_EPOCHS = 50
ES_PATIENCE = 6


"""
Einmalige Endbewertung des finalen Modells auf einem Datensatz.

Der Konstruktor ruft extract() genau einmal auf; alles Weitere (Accuracy, F1, Confusion Matrix, Fehlerpaare)
wird aus diesen Vorhersagen berechnet. Das Modell wird danach nicht mehr angefasst.
"""
class Evaluation:

    def __init__(self, model, dataset, out_dir=OUT_DIR, n_pairs=3, n_examples=3):
        self.dataset = dataset
        self.out_dir = out_dir
        self.n_examples = n_examples
        self.y, self.pred, _ = extract(model, dataset)                       # Zeile i gehoert zu dataset[i]
        self.accuracy = accuracy_score(self.y, self.pred)
        self.f1 = f1_score(self.y, self.pred, average=None)                  # je Klasse, Reihenfolge wie CLASSES
        self.counts = confusion_matrix(self.y, self.pred)                    # Rohzaehlungen: Zeile = wahr, Spalte = vorhergesagt
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

    """
    Häufigste Verwechslungspaare {i, j}, gerankt nach n(i->j) + n(j->i); dominante Richtung = mehr Fehler.
    """       
    def _find_top_pairs(self, n_pairs):
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
    Trägt Indizes und Dateipfade in self.pairs ein, daher vor metrics() aufrufen.
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
                  f"{name_j}->{name_i} {pair['anzahl_j_zu_i']} (dominant: {pair['dominante_richtung']}); "
                  f"Beispiele #{examples}")

    def metrics(self):
        """Kennzahlen der Testauswertung als JSON-faehiges Dict."""
        return {
            "accuracy": float(self.accuracy),
            "f1_je_klasse": {name: float(v) for name, v in zip(CLASSES, self.f1)},
            "f1_macro": float(np.mean(self.f1)),
            "schwächste_klasse_nach_f1": CLASSES[self.weakest_f1],
            "recall_je_klasse": {name: float(v) for name, v in zip(CLASSES, self.recall)},
            "schlechteste_klasse_nach_recall": self.worst_class,
            "haeufigste_paare": self.pairs,
            "confusion_matrix_zählungen": self.counts.tolist(),
            "confusion_matrix_zeilennormiert": self.normalized.tolist(),
        }