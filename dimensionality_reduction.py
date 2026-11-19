from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness

class DimensionalityReduction:

    def __init__(self, E):
        self.E = E
        
    def align(self, layout, reference):
        """Procrustes: 2D-Layout zentrieren und so drehen/spiegeln, dass es moeglichst gut auf reference passt
        (z. B. reference = PCA(n_components=2).fit_transform(E)). Keine Skalierung, Abstaende im Layout bleiben gleich."""
        from scipy.linalg import orthogonal_procrustes
        layout_centered = layout - layout.mean(0)
        reference_centered = reference - reference.mean(0)
        rotation, _ = orthogonal_procrustes(layout_centered, reference_centered)   # orthogonale 2x2-Matrix
        return layout_centered @ rotation   # @ = Matrixmultiplikation (wie np.matmul): jeder Punkt wird gedreht

    def get_tsne_layout(self, reference, init='pca', random_state=0, perplexity=5):
        layout = TSNE(init=init, random_state=random_state, perplexity=perplexity).fit_transform(self.E)
        return self.align(layout, reference)
        
    def get_umap_layout(self, reference, init="spectral", random_state=0, n_neighbors = 5):
        layout = umap.UMAP(init=init, random_state=random_state, n_neighbors = n_neighbors).fit_transform(self.E)
        return self.align(layout, reference)
    
    def get_trustworthiness(self, layout, k_list):
        t_list = []
        for k in k_list:
            t_list.append(trustworthiness(self.E, layout, n_neighbors=k))
        return t_list
        
    def get_continuity(self, layout, k_list):
        c_list = []
        for k in k_list:
            c_list.append(trustworthiness(layout, self.E, n_neighbors=k))
        return c_list