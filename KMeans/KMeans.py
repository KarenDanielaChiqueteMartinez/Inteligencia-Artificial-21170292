# Implementacion con problema de ejemplo
# Referencias
# K-Means Clustering - Python Machine Learning Tutorial Link:https://youtu.be/EItlUEPCIzM?si=U5f95CgS5tO9_4Kg
# Implementación de K-Means en Python - Tutorial Link:https://youtu.be/V5h7E-zK5Ko?si=U5f95CgS5tO9_4Kg  
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

from sklearn.cluster import KMeans
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Generar datos de ejemplo más interesantes
np.random.seed(42)
# Crear tres grupos de datos bien separados
cluster1 = np.random.randn(50, 2) + [2, 2]
cluster2 = np.random.randn(50, 2) + [8, 8]
cluster3 = np.random.randn(50, 2) + [2, 8]

# Combinar todos los datos
X = np.vstack([cluster1, cluster2, cluster3])

# Aplicar K-Means con 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

# Obtener las etiquetas y centroides
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Crear la visualización
plt.figure(figsize=(10, 8))

# Colores para los clusters
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', '#AA96DA']
cmap = ListedColormap(colors[:3])

# Graficar los puntos de cada cluster con colores diferentes
for i in range(3):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                c=colors[i], label=f'Cluster {i+1}', 
                s=100, alpha=0.6, edgecolors='black', linewidth=1.5)

# Graficar los centroides
plt.scatter(centers[:, 0], centers[:, 1], 
            c='red', marker='X', s=300, 
            label='Centroides', edgecolors='black', 
            linewidth=2, zorder=10)

# Configurar el gráfico
plt.title('Clustering K-Means con 3 Clusters', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Característica 1', fontsize=12)
plt.ylabel('Característica 2', fontsize=12)
plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Guardar la imagen
plt.savefig('kmeans_visualization.png', dpi=300, bbox_inches='tight')
print("Imagen guardada como 'kmeans_visualization.png'")

# Cerrar la figura para liberar memoria
plt.close()

# Información adicional
print(f"\nNúmero de puntos por cluster:")
for i in range(3):
    print(f"Cluster {i+1}: {np.sum(labels == i)} puntos")
    
print(f"\nCoordenadas de los centroides:")
for i, center in enumerate(centers):
    print(f"Cluster {i+1}: ({center[0]:.2f}, {center[1]:.2f})")
