"""Script reorganizado: carga dos datasets (breast cancer y fashion_mnist), entrena modelos
simples, genera gráficas y guarda las imágenes en la carpeta `imagenes_cnn`.

Ejecutar:
  python cnn.py

Requisitos (instalar si hace falta):
  pip install sinfo numpy pandas matplotlib seaborn scikit-learn tensorflow
"""

from sinfo import sinfo
from pathlib import Path
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report


def ensure_output_folder(folder_name: str) -> Path:
  out = Path(folder_name)
  out.mkdir(parents=True, exist_ok=True)
  return out


def plot_and_save_history(history, out_path: Path, title: str = "History"):
  df = pd.DataFrame(history.history)
  ax = df.plot(figsize=(10, 5))
  ax.set_title(title)
  fig = ax.get_figure()
  fig.savefig(out_path, dpi=100, bbox_inches='tight')
  plt.close(fig)


def plot_and_save_confusion(cm, labels, out_path: Path, title: str = "Confusion Matrix"):
  fig, ax = plt.subplots(figsize=(6, 6))
  sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=True,
        xticklabels=labels, yticklabels=labels, ax=ax)
  ax.set_ylabel('Actual label')
  ax.set_xlabel('Predicted label')
  ax.set_title(title)
  fig.savefig(out_path, dpi=100, bbox_inches='tight')
  plt.close(fig)


def breast_cancer_workflow(output_folder: Path):
  print('\n' + '=' * 40)
  print('BREAST CANCER - Data and model')
  print('=' * 40)

  data = load_breast_cancer()
  X = data.data
  y = data.target

  df = pd.DataFrame(X, columns=data.feature_names)
  print('Dataset shape:', df.shape)

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
  )
  print('Train samples:', X_train.shape[0], 'Test samples:', X_test.shape[0])

  scaler = MinMaxScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  input_shape = X_train_scaled.shape[1:]

  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=30, input_shape=input_shape, activation='relu'),
    tf.keras.layers.Dense(units=15, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

  history = model.fit(X_train_scaled, y_train, epochs=20, validation_split=0.15, verbose=1)

  plot_and_save_history(history, output_folder / '01_breast_history.png', title='Breast - Training History')
  print('✓ Guardada: 01_breast_history.png')

  eval_test = model.evaluate(X_test_scaled, y_test, verbose=0)
  print(f'Test loss: {eval_test[0]:.4f}  Test accuracy: {eval_test[1]:.4f}')

  preds = model.predict(X_test_scaled)
  preds = tf.round(preds).numpy().astype(int).ravel()

  cm = confusion_matrix(y_test, preds)
  labels = ['Malignant (0)', 'Benign (1)']
  plot_and_save_confusion(cm, labels, output_folder / '02_breast_confusion.png', title='Breast - Confusion Matrix')
  print('✓ Guardada: 02_breast_confusion.png')

  print(classification_report(y_test, preds))


def fashion_mnist_workflow(output_folder: Path):
  print('\n' + '=' * 40)
  print('FASHION MNIST - Data and model')
  print('=' * 40)

  fashion_mnist = tf.keras.datasets.fashion_mnist
  (fashion_train, fashion_train_label), (fashion_test, fashion_test_label) = fashion_mnist.load_data()

  class_names = ['T-shirt/top', 'Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

  # Save one sample image
  sample_idx = 10
  fig = plt.figure()
  plt.imshow(fashion_train[sample_idx], cmap='gray')
  plt.title(f'Sample: {class_names[fashion_train_label[sample_idx]]} ({fashion_train_label[sample_idx]})')
  fig.savefig(output_folder / '03_fashion_sample.png', dpi=100, bbox_inches='tight')
  plt.close(fig)
  print('✓ Guardada: 03_fashion_sample.png')

  # Save a grid of random samples
  import random
  fig, axes = plt.subplots(2, 3, figsize=(8, 6))
  for i, ax in enumerate(axes.flatten()):
    idx = random.randrange(len(fashion_train))
    ax.imshow(fashion_train[idx], cmap='gray')
    ax.set_title(class_names[fashion_train_label[idx]])
    ax.axis('off')
  fig.savefig(output_folder / '04_fashion_samples.png', dpi=100, bbox_inches='tight')
  plt.close(fig)
  print('✓ Guardada: 04_fashion_samples.png')

  # Scale
  fashion_train_scaled = fashion_train / 255.0
  fashion_test_scaled = fashion_test / 255.0

  # Simple dense classifier
  def classifier():
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(units=64, activation='relu'),
      tf.keras.layers.Dense(units=32, activation='relu'),
      tf.keras.layers.Dense(units=10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

  model = classifier()
  history = model.fit(fashion_train_scaled, fashion_train_label, epochs=10, validation_split=0.15, verbose=1)

  plot_and_save_history(history, output_folder / '05_fashion_history.png', title='Fashion - Training History')
  print('✓ Guardada: 05_fashion_history.png')

  eval_test = model.evaluate(fashion_test_scaled, fashion_test_label, verbose=0)
  print(f'Fashion Test loss: {eval_test[0]:.4f}  Test accuracy: {eval_test[1]:.4f}')

  preds = model.predict(fashion_test_scaled)
  preds = np.argmax(preds, axis=1)

  cm = confusion_matrix(fashion_test_label, preds)
  plot_and_save_confusion(cm, class_names, output_folder / '06_fashion_confusion.png', title='Fashion - Confusion Matrix')
  print('✓ Guardada: 06_fashion_confusion.png')


def main():
  print('=' * 60)
  print('INFORMACIÓN DEL ENTORNO')
  print('=' * 60)
  sinfo()

  out = ensure_output_folder('imagenes_cnn')
  print(f'Las gráficas se guardarán en: {out.resolve()}')

  breast_cancer_workflow(out)
  fashion_mnist_workflow(out)

  print('\nTodas las imágenes han sido guardadas en:', out)


if __name__ == '__main__':
  main()