import json
import numpy as np


class ResultLogger:

    def __init__(self, filename="results/results.json"):

        self.filename = filename

        self.data = {"CNN": {}, "DimensionalityReduction": {}, "Clustering": {}}

    def add(self, category, name, metrics):

        if name not in self.data[category]:
            self.data[category][name] = {}
    
        self.data[category][name].update(metrics)


    def add_epoch(self, model, epoch, metrics):
    
        if model not in self.data["CNN"]:
            self.data["CNN"][model] = {}
    
        if "epochs" not in self.data["CNN"][model]:
            self.data["CNN"][model]["epochs"] = []
    
        self.data["CNN"][model]["epochs"].append({"epoch": epoch, **metrics})

    def save(self):

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False, default=convert)
