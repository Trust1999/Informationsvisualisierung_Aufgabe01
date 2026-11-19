from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

class CNNEvaluation:
    def __init__(self, logger):
        self.logger = logger

    def get_error_classes(self, all_labels, all_predictions, classes):
        f1_score_list = f1_score(all_labels, all_predictions, average = None)
        
        accuracies = []
        
        for i in range(10):
            indices = np.where(all_labels == i)[0]
            true = all_labels[indices]
            pred = all_predictions[indices]
            acc = accuracy_score(true, pred)
            accuracies.append(acc)
            
            print(f"---> {classes[i]}:\n")
            print(f"Accuracy: {acc}")
            print(f"F1-Score: {f1_score_list[i]}\n")
            
        conf_matrix = confusion_matrix(all_labels, all_predictions)    
        
        error_classes = [(0,0),(0,0),(0,0)]
        
        np.fill_diagonal(conf_matrix, 0)
        
        print("=== Häufigste Verwechslungen ===")
        print("=== Wahr <---> Vorhergesagt  ===")
        
        for i in range(len(error_classes)):
            flat_idx = np.argmax(conf_matrix)
            row, col = np.unravel_index(flat_idx, conf_matrix.shape)
            error_classes[i] = (row, col)
            conf_matrix[row, col] = 0
            print(f"{i+1}. {classes[row]} <---> {classes[col]}")
        
        self.logger.add("CNN", "ResNet18", {
        "class_metrics": {
            classes[i]: {
                "accuracy": accuracies[i],
                "f1_score": f1_score_list[i]
            }
            for i in range(len(classes))
        },
        "most_common_confusions": [
            {
                "true": classes[row],
                "predicted": classes[col]
            }
            for row, col in error_classes
        ]
        })

        
        return error_classes