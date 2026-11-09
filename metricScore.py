import json
import os

OUT_DIR = "output"                                    # Ausgabeordner fuer alle Abbildungen und Kennzahlen
METRICS_PATH = os.path.join(OUT_DIR, "kennzahlen.json")

"""Sammelt Kennzahlen aller Aufgaben und schreibt sie als eine JSON-Datei in den Ausgabeordner."""
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