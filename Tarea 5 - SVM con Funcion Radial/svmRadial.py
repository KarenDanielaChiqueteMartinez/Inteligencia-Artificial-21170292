# SVM con Función Radial (RBF - Radial Basis Function)
# ==============================================================================
# Este ejemplo demuestra el uso de SVM con kernel radial para clasificación
# de datos no linealmente separables

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

# Forzar salida sin buffer
sys.stdout.reconfigure(encoding='utf-8')

# Configuración de carpeta de salida
# ==============================================================================
import os
script_dir = Path(__file__).parent
output_folder = script_dir / 'imagenes_svm_radial'
output_folder.mkdir(exist_ok=True)

# Configuración matplotlib
# ==============================================================================
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['savefig.bbox'] = 'tight'

print("=" * 60)
print("EJEMPLO: SVM CON FUNCIÓN RADIAL (RBF)")
print("=" * 60)
print("")
sys.stdout.flush()

# Ejemplo 1: Datos en círculos concéntricos
# ==============================================================================
print("Ejemplo 1: Datos en círculos concéntricos")
print("-" * 60)

# Generar datos no linealmente separables
X1, y1 = make_circles(n_samples=300, factor=0.5, noise=0.1, random_state=42)

# Visualizar datos originales
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico de datos originales
axes[0].scatter(X1[:, 0], X1[:, 1], c=y1, cmap=plt.cm.RdYlBu, s=50, edgecolors='black', linewidth=1)
axes[0].set_title('Datos Originales - Círculos Concéntricos', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Característica 1')
axes[0].set_ylabel('Característica 2')
axes[0].grid(True, alpha=0.3)

# División train/test
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42, shuffle=True
)

# Entrenar SVM con kernel radial
svm_radial_1 = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_radial_1.fit(X1_train, y1_train)

# Predicciones
y1_pred = svm_radial_1.predict(X1_test)
accuracy_1 = accuracy_score(y1_test, y1_pred)

print(f"Accuracy en test: {accuracy_1*100:.2f}%")
print(f"Número de vectores soporte: {len(svm_radial_1.support_vectors_)}")
print("")

# Visualizar resultados con frontera de decisión
# Crear una malla para visualizar la frontera
h = 0.02
x_min, x_max = X1[:, 0].min() - 0.5, X1[:, 0].max() + 0.5
y_min, y_max = X1[:, 1].min() - 0.5, X1[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predecir para cada punto de la malla
Z = svm_radial_1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Gráfico con frontera de decisión
axes[1].contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
axes[1].scatter(X1_train[:, 0], X1_train[:, 1], c=y1_train, cmap=plt.cm.RdYlBu, 
                s=50, edgecolors='black', linewidth=1, label='Datos entrenamiento')
axes[1].scatter(svm_radial_1.support_vectors_[:, 0], 
                svm_radial_1.support_vectors_[:, 1],
                s=200, linewidth=2, facecolors='none', 
                edgecolors='black', label='Vectores soporte')
axes[1].set_title(f'SVM Radial - Accuracy: {accuracy_1*100:.2f}%', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Característica 1')
axes[1].set_ylabel('Característica 2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_folder / '01_svm_radial_circulos.png', dpi=300, bbox_inches='tight')
print(f"✓ Gráfica guardada: 01_svm_radial_circulos.png")
plt.close()

# Ejemplo 2: Comparación de diferentes valores de C y gamma
# ==============================================================================
print("\nEjemplo 2: Efecto de los parámetros C y gamma")
print("-" * 60)

X2, y2 = make_moons(n_samples=200, noise=0.2, random_state=42)

# Crear malla para visualización
h = 0.02
x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Diferentes combinaciones de parámetros
parametros = [
    {'C': 0.1, 'gamma': 0.1, 'title': 'C=0.1, gamma=0.1 (Bajo ajuste)'},
    {'C': 1.0, 'gamma': 1.0, 'title': 'C=1.0, gamma=1.0 (Balanceado)'},
    {'C': 100, 'gamma': 10, 'title': 'C=100, gamma=10 (Alto ajuste)'}
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, params in enumerate(parametros):
    svm = SVC(kernel='rbf', C=params['C'], gamma=params['gamma'], random_state=42)
    svm.fit(X2, y2)
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    axes[idx].scatter(X2[:, 0], X2[:, 1], c=y2, cmap=plt.cm.RdYlBu, 
                     s=50, edgecolors='black', linewidth=1)
    axes[idx].scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                     s=200, linewidth=2, facecolors='none', edgecolors='black')
    axes[idx].set_title(params['title'], fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Característica 1')
    axes[idx].set_ylabel('Característica 2')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_folder / '02_svm_radial_parametros.png', dpi=300, bbox_inches='tight')
print(f"✓ Gráfica guardada: 02_svm_radial_parametros.png")
plt.close()

# Ejemplo 3: Comparación con datos más complejos
# ==============================================================================
print("\nEjemplo 3: Clasificación con datos más complejos")
print("-" * 60)

# Generar datos más complejos
X3, y3 = make_circles(n_samples=400, factor=0.3, noise=0.15, random_state=42)

X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3, y3, test_size=0.3, random_state=42, shuffle=True
)

# Entrenar modelo optimizado
svm_radial_3 = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
svm_radial_3.fit(X3_train, y3_train)

y3_pred = svm_radial_3.predict(X3_test)
accuracy_3 = accuracy_score(y3_test, y3_pred)

print(f"Accuracy en test: {accuracy_3*100:.2f}%")
print(f"Número de vectores soporte: {len(svm_radial_3.support_vectors_)}")
print("\nReporte de clasificación:")
print(classification_report(y3_test, y3_pred, target_names=['Clase 0', 'Clase 1']))

# Visualización final
fig, ax = plt.subplots(figsize=(10, 8))

h = 0.02
x_min, x_max = X3[:, 0].min() - 0.5, X3[:, 0].max() + 0.5
y_min, y_max = X3[:, 1].min() - 0.5, X3[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = svm_radial_3.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap=plt.cm.RdYlBu)
ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', linestyles='--')
ax.scatter(X3_train[:, 0], X3_train[:, 1], c=y3_train, cmap=plt.cm.RdYlBu,
          s=60, edgecolors='black', linewidth=1.5, alpha=0.8, label='Entrenamiento')
ax.scatter(X3_test[:, 0], X3_test[:, 1], c=y3_test, cmap=plt.cm.RdYlBu,
          s=100, marker='^', edgecolors='black', linewidth=2, label='Test')
ax.scatter(svm_radial_3.support_vectors_[:, 0], 
           svm_radial_3.support_vectors_[:, 1],
           s=300, linewidth=3, facecolors='none', 
           edgecolors='yellow', label='Vectores soporte', zorder=10)

ax.set_title(f'SVM Radial - Clasificación Compleja\nAccuracy: {accuracy_3*100:.2f}%', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Característica 1', fontsize=12)
ax.set_ylabel('Característica 2', fontsize=12)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_folder / '03_svm_radial_complejo.png', dpi=300, bbox_inches='tight')
print(f"✓ Gráfica guardada: 03_svm_radial_complejo.png")
plt.close()

print("")
print("=" * 60)
print(f"✓ Todas las gráficas han sido guardadas en: {output_folder}")
print("=" * 60)
print("\nResumen:")
print("- El kernel radial (RBF) permite clasificar datos no linealmente separables")
print("- El parámetro C controla el trade-off entre margen y error de clasificación")
print("- El parámetro gamma controla la influencia de cada vector soporte")
print("- Valores altos de gamma pueden llevar a sobreajuste (overfitting)")

