import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


class Visualization:

    def __init__(self):
        pass

    def plot_clusters(self, tsne_layout, umap_layout, labels_dict, save = True):
        x_coords_tsne = tsne_layout[:, 0]
        y_coords_tsne = tsne_layout[:, 1]
        x_coords_umap = umap_layout[:, 0]
        y_coords_umap = umap_layout[:, 1]
        
        for cluster_method, labels in labels_dict.items():
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            noise_mask = labels == -1
            
            scatter = ax[0].scatter(x_coords_tsne[np.invert(noise_mask)], y_coords_tsne[np.invert(noise_mask)], c=labels[np.invert(noise_mask)], cmap="tab20b", alpha=0.8)
            scatter = ax[0].scatter(x_coords_tsne[noise_mask], y_coords_tsne[noise_mask], color='gray', alpha=0.8)
            
            ax[0].set_title(f"{cluster_method} im t-SNE Layout")
            ax[0].set_xlabel("t-SNE 1")
            ax[0].set_ylabel("t-SNE 2")
            ax[0].grid(True, linestyle="--", alpha=0.5)
            
            
            scatter = ax[1].scatter(x_coords_umap[np.invert(noise_mask)], y_coords_umap[np.invert(noise_mask)], c=labels[np.invert(noise_mask)], cmap="tab20b", alpha=0.8)
            scatter = ax[1].scatter(x_coords_umap[noise_mask], y_coords_umap[noise_mask], color='gray', alpha=0.8)
            
            ax[1].set_title(f"{cluster_method} im UMAP Layout")
            ax[1].set_xlabel("t-SNE 1")
            ax[1].set_ylabel("t-SNE 2")
            ax[1].grid(True, linestyle="--", alpha=0.5)
            
            plt.tight_layout()
            
            if save:     
                plt.savefig(f"results/{cluster_method.replace(" ", "_")}.png", dpi=300)
            
            plt.show()
        
    def plot_k_distances(self, x, sorted_dist, save=True):
        min_max_line = x * (max(sorted_dist)-min(sorted_dist))/5000 + min(sorted_dist)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.title("k-Distanz-Plot")
        plt.ylabel("Distanz")
        plt.xlabel("Index")
        plt.yticks(np.arange(0, max(sorted_dist) + 1, 1))
        plt.plot(sorted_dist)
        plt.plot(min_max_line)
        
        if save:     
            plt.savefig(f"results/k_distances.png", dpi=300)
        
        plt.show()
        
        
    def plot_reduction_comparison(self, tsne_layout_dict, umap_layout_dict, y, classes, save = True):
        fig, ax = plt.subplots(2, 3, figsize=(10, 4), layout="constrained")
        ax[0,0].set_ylabel("t-SNE")
        ax[1,0].set_ylabel("UMAP")
        
        for i, (perplexity, layout) in enumerate(tsne_layout_dict.items()):
            sc = ax[0, i].scatter(layout[:, 0], layout[:, 1], s=5, alpha=0.6, c=y, cmap='tab10')
            ax[0, i].set_title(f"Perplexity={perplexity}")

        for i, (neighbours, layout) in enumerate(umap_layout_dict.items()):
            ax[1, i].scatter(layout[:, 0], layout[:, 1], s=5, alpha=0.6, c=y, cmap='tab10')
            ax[1, i].set_title(f"n_neighbors={neighbours}")

    
        plt.tight_layout()
        fig.subplots_adjust(right=0.82) 
        handles, _ = sc.legend_elements(prop="colors")
        fig.legend(handles=handles,labels=classes,loc="center right",title="Klassen",borderaxespad=0.5, bbox_to_anchor=(0.97, 0.5))
        
        if save:     
            plt.savefig(f"results/reduction_comparison.png", dpi=300)
        
        plt.show()
        
    def plot_reduction_evaluation(self, trustworthiness_tsne, continuity_tsne, trustworthiness_umap, continuity_umap, k_list, save = True):
        X=np.array(k_list)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        ax[0].set_title("Trustworthiness")
        ax[0].set_ylabel("T(K)")
        ax[0].set_xlabel("K")
        ax[0].plot(X, trustworthiness_tsne[0], 'o-', label="t-SNE (p=5)")
        ax[0].plot(X, trustworthiness_tsne[1], 'o-', label="t-SNE (p=30)")
        ax[0].plot(X, trustworthiness_tsne[2], 'o-', label="t-SNE (p=100)")
        ax[0].plot(X, trustworthiness_umap[0], 'o-', label="UMAP (n=5)")
        ax[0].plot(X, trustworthiness_umap[1], 'o-', label="UMAP (n=15)")
        ax[0].plot(X, trustworthiness_umap[2], 'o-', label="UMAP (n=100)")
        ax[0].set_xscale('log')
        ax[0].legend(loc='lower left')
        ax[0].set_xticks(X)
        ax[0].xaxis.set_major_formatter(ticker.ScalarFormatter())
        
        
        ax[1].set_ylabel("C(K)")
        ax[1].set_xlabel("K")
        ax[1].set_title("Continuity")
        ax[1].plot(X, continuity_tsne[0], 'o-', label="t-SNE (p=5)")
        ax[1].plot(X, continuity_tsne[1], 'o-', label="t-SNE (p=30)")
        ax[1].plot(X, continuity_tsne[2], 'o-', label="t-SNE (p=100)")
        ax[1].plot(X, continuity_umap[0], 'o-', label="UMAP (n=5)")
        ax[1].plot(X, continuity_umap[1], 'o-', label="UMAP (n=15)")
        ax[1].plot(X, continuity_umap[2], 'o-', label="UMAP (n=100)")
        ax[1].set_xscale('log')
        ax[1].legend(loc='lower left')
        ax[1].set_xticks(X)
        ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
        
        if save:     
            plt.savefig(f"results/reduction_evaluation.png", dpi=300)
        
        plt.show()
    
    def plot_final_layouts(self, y, pred, tsne_layout, umap_layout, save = True):
        mask = y != pred
        
        y_masked = y[mask]
        tsne_layout_final_centered_masked = tsne_layout[mask]
        umap_layout_final_centered_masked = umap_layout[mask]
        

        plt.figure(1) 
        #plt.figure(figsize=(8, 12), dpi=300)
        plt.scatter(tsne_layout[:, 0], tsne_layout[:, 1], s=5, alpha=0.15, c=y, cmap='tab10')
        plt.scatter(tsne_layout_final_centered_masked[:, 0], tsne_layout_final_centered_masked[:, 1], s=7, alpha=0.6, c=y_masked, cmap='tab10', edgecolors='black', linewidths=1)
        plt.title('t-SNE')
        
        #plt.figure(2, figsize=(8, 12), dpi=300)
        
        plt.figure(2)
        #plt.figure(2, figsize=(8, 12), dpi=300)
        plt.scatter(umap_layout[:, 0], umap_layout[:, 1], s=5, alpha=0.15, c=y, cmap='tab10')
        plt.scatter(umap_layout_final_centered_masked[:, 0], umap_layout_final_centered_masked[:, 1], s=7, alpha=0.6, c=y_masked, cmap='tab10', edgecolors='black', linewidths=1)
        plt.title('UMAP')
        
        
        if save:     
            plt.figure(1).savefig(f"results/tsne_final.png", dpi=300)
            plt.figure(2).savefig(f"results/umap_final.png", dpi=300)
        
        plt.show()
    
    def plot_loss_accuracy(self, epochs, losses_val, losses_train, best_val_loss_epoch, accs_val, accs_train, save = True):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        ax[0].plot(np.arange(1,epochs+1), losses_val, 'o-', label="Validation")
        ax[0].plot(np.arange(1,epochs+1), losses_train,'o-', label = "Training")
        ax[0].set_title("Loss")
        ax[0].set_xticks(range(1, epochs+1))
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("")
        ax[0].grid(True)
        
        ax[0].get_xticklabels()[best_val_loss_epoch].set_color('green')
        ax[0].get_xticklabels()[best_val_loss_epoch].set_fontweight('bold')
        
        ax[1].plot(np.arange(1,epochs+1), accs_val, 'o-', label="Validation")
        ax[1].plot(np.arange(1,epochs+1), accs_train, 'o-', label = "Training")
        ax[1].set_title("Accuracy")
        ax[1].set_xticks(range(1, epochs+1))
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("")
        ax[1].grid(True)
        
        ax[1].get_xticklabels()[best_val_loss_epoch].set_color('green')
        ax[1].get_xticklabels()[best_val_loss_epoch].set_fontweight('bold')
        
        for a in ax:
            y_min, y_max = a.get_ylim()
            y_min = round(y_min, 2)
            y_max = round(y_max, 2)
            if list(ax).index(a) == 0:
                a.set_yticks(np.arange(y_min, y_max + 0.01, 0.05))
            else:
                a.set_yticks(np.arange(y_min, y_max + 0.01, 0.02))
        
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)

        plt.tight_layout()
        
        if save:     
            plt.savefig(f"results/loss_accuracy.png", dpi=300)
        
        plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    