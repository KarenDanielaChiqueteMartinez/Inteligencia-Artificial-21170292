# Información de versiones
# ==============================================================================
from sinfo import sinfo
print("=" * 50)
print("INFORMACIÓN DE VERSIONES")
print("=" * 50)
sinfo()

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import LinearSVC

# Sistema de archivos
# ==============================================================================
from pathlib import Path

# Configuración de carpeta de salida
# ==============================================================================
output_folder = Path('imagenes_linearsvc')
output_folder.mkdir(exist_ok=True)

X, y = make_blobs(n_samples=40, centers=2, random_state=0)

plt.figure(figsize=(10, 5))
for i, C in enumerate([1, 100]):
    # "hinge" is the standard SVM loss
    clf = LinearSVC(C=C, loss="hinge", random_state=42).fit(X, y)
    # obtain the support vectors through the decision function
    decision_function = clf.decision_function(X)
    # we can also calculate the decision function manually
    # decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]
    # The support vectors are the samples that lie within the margin
    # boundaries, whose size is conventionally constrained to 1
    support_vector_indices = (np.abs(decision_function) <= 1 + 1e-15).nonzero()[0]
    support_vectors = X[support_vector_indices]

    plt.subplot(1, 2, i + 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        ax=ax,
        grid_resolution=50,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )
    plt.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    plt.title("C=" + str(C))
plt.tight_layout()
plt.savefig(output_folder / 'linearsvc_comparacion_C.png', dpi=100, bbox_inches='tight')
print("✓ Gráfica guardada: linearsvc_comparacion_C.png")
plt.show()
plt.close()

print("")
print("=" * 50)
print(f"✓ Todas las gráficas han sido guardadas en: {output_folder}")
print("=" * 50)