"""
Utilidades para el proyecto Ising Model
Contiene todas las clases y funciones reutilizables
"""

import numpy as np
import re
import matplotlib.pyplot as plt
import os
import json
import h5py
import torch
from torch.utils.data import Dataset


class read_ising_file:
    """Clase para leer y procesar archivos de datos del modelo Ising"""

    def __init__(self, file):
        self.file = file
        # Leer archivo
        with open(file, 'r') as f:
            self.metadata = json.loads(f.readlines()[0][:-1])

    def reshape_spin(self, index):
        """Convertir un vector de espines plano a una matriz de espines tamaño LxL"""
        # Leer linea
        with open(self.file, 'r') as f:
            spin = f.readlines()[index][:-1]

        # Convertir a np.array (leer como float primero, luego convertir a int)
        arr = np.fromstring(spin, dtype=float, sep=',')
        arr = arr.astype(int)  # Convertir a int después de leer
        arr = arr.reshape(self.metadata['L'], self.metadata['L'])
        return arr

    def plot_spin(self, spin_index=1, spin_arrows=True, slice=None):
        """Graficar red de espines en B/N, con opción de flechas ↑ ↓ en cada celda"""
        spin = self.reshape_spin(spin_index)
        if slice != None:
            spin = spin[slice[0]:slice[1], slice[2]:slice[3]]
        L = self.metadata["L"]
        # Usar las dimensiones reales del array (puede ser recortado)
        rows, cols = spin.shape

        plt.figure(figsize=(6, 6))
        plt.imshow(spin, cmap='gray', vmin=-1, vmax=1)

        plt.title(f'Red de espines T = {self.metadata["T"]:.3f}')
        plt.xlabel(f'Tamaño de la red: {L} x {L}')
        plt.ylabel('')

        if spin_arrows:
            # Dibuja las flechas usando las dimensiones reales del array
            for i in range(rows):
                for j in range(cols):
                    arrow = '↑' if spin[i, j] == 1 else '↓'
                    color = 'black' if spin[i, j] == 1 else 'white'
                    plt.text(
                        j, i, arrow,
                        ha='center', va='center',
                        color=color, fontsize=14, fontweight='bold'
                    )

        plt.xticks([])
        plt.yticks([])
        plt.show()

    def magnetization(self, index):
        """Calcular magnetización de la red de espines"""
        spin = self.reshape_spin(index)
        return np.sum(spin)

    def load_all_spins(self, invert=True):
        """Cargar todos los espines del archivo de una vez (optimización I/O)"""
        all_spins = []
        expected_size = self.metadata['L'] * self.metadata['L']

        with open(self.file, 'r') as f:
            lines = f.readlines()[1:]  # Saltar el header (metadata)
            for line in lines:
                if line.strip():  # Ignorar líneas vacías
                    # Leer como float primero (los archivos contienen 1.0, -1.0)
                    arr = np.fromstring(line[:-1], dtype=float, sep=',')
                    # Validar que el tamaño sea correcto antes de reshape
                    if arr.size != expected_size:
                        continue  # Saltar líneas con tamaño incorrecto
                    arr = arr.astype(int)  # Convertir a int después de leer
                    arr = arr.reshape(self.metadata['L'], self.metadata['L'])
                    all_spins.append(arr)

                    # Agregar la red invertida
                    if invert:
                        all_spins.append(-arr)

        return all_spins

    def energy(self, index):
        """Calcular energía de la red de espines, con interacción entre primeros vecinos y campo magnético externo"""
        spin = self.reshape_spin(index)

        # Término de interacción entre primeros vecinos
        interaction_energy = 0.0

        # Interacciones horizontales (hacia la derecha)
        interaction_energy -= self.metadata['J'] * np.sum(spin[:, :-1] * spin[:, 1:])
        interaction_energy -= self.metadata['J'] * np.sum(spin[:, -1] * spin[:, 0])  # Condición periódica

        # Interacciones verticales (hacia abajo)
        interaction_energy -= self.metadata['J'] * np.sum(spin[:-1, :] * spin[1:, :])
        interaction_energy -= self.metadata['J'] * np.sum(spin[-1, :] * spin[0, :])  # Condición periódica

        # Término del campo magnético externo
        field_energy = -self.metadata['H'] * np.sum(spin)

        return interaction_energy + field_energy

    @staticmethod
    def calculate_energy(spin, metadata):
        """Calcular energía de un espín ya cargado (sin I/O)"""
        # Término de interacción entre primeros vecinos
        interaction_energy = 0.0

        # Interacciones horizontales (hacia la derecha)
        interaction_energy -= metadata['J'] * np.sum(spin[:, :-1] * spin[:, 1:])
        interaction_energy -= metadata['J'] * np.sum(spin[:, -1] * spin[:, 0])  # Condición periódica

        # Interacciones verticales (hacia abajo)
        interaction_energy -= metadata['J'] * np.sum(spin[:-1, :] * spin[1:, :])
        interaction_energy -= metadata['J'] * np.sum(spin[-1, :] * spin[0, :])  # Condición periódica

        # Término del campo magnético externo
        field_energy = -metadata['H'] * np.sum(spin)

        return interaction_energy + field_energy

    @staticmethod
    def calculate_magnetization(spin):
        """Calcular magnetización de un espín ya cargado (sin I/O)"""
        return np.sum(spin)

    @staticmethod
    def invert_spins(spin):
        """Invierte toda la red"""
        return -spin

class ising_data_builder(read_ising_file):
    """Construye el dataset para entrenar los modelos, a partir de los datos generados en Ising_generator"""

    def __init__(self, folder, invert=True, kind='classification', alpha=0.05):
        self.folder = folder
        self.invert = invert
        self.kind = kind
        self.alpha = alpha

        self._scan_folder()
        if self.kind == 'classification':
            self._build_dataset()

        elif self.kind == 'inpainting':
            self._build_inpainting_dataset()

    def _scan_folder(self):
        """Escanear el folder con los archivos validos"""
        patron = re.compile(r"ising_")
        self.files = []

        for file in os.listdir(self.folder):
            match = patron.search(file)

            if match:
                self.files.append(file)

    def _load_file(self, file):
        super().__init__(file)

    def _build_dataset(self, h5_filename='dataset.h5'):
        """Crear un file de tipo HDF5 para evitar uso excesivo de recurso usando recursión de lista"""
        # Guardar el h5
        h5_path = os.path.join(self.folder, h5_filename)
        if os.path.exists(h5_path):
            os.remove(h5_path)

        total = 0
        f = h5py.File(h5_path, 'w')
        dX = dy = None  # Inicializar datasets en None

        # Iterar sobre cada archivo de Ising encontrado en el folder
        for ising_file in self.files:
            self._load_file(self.folder + ising_file)
            X = np.asarray(self.load_all_spins())
            y = self.metadata['class'] * np.ones(len(X))

            # Si es la primera vez, crear los datasets dentro del archivo h5
            if dX is None:
                dX = f.create_dataset('X', data=X,
                                      maxshape=(None,) + X.shape[1:],
                                      chunks=True)
                dy = f.create_dataset('y', data=y,
                                      maxshape=(None,),
                                      chunks=True)
            else:
                # Si ya existen, hacer resize para agregar las nuevas muestras al final
                m = len(X)
                dX.resize((total + m), axis=0)
                dy.resize((total + m), axis=0)
                dX[total: total + m] = X  # Escribir los nuevos datos de X
                dy[total: total + m] = y  # Escribir las nuevas etiquetas

            total += len(X)

        f.close()
        self.h5_path = h5_path
        self.n_samples = total

        return self.h5_path

    def _build_inpainting_dataset(self, mask_ratio=0.05, h5_filename='dataset_inpainting.h5'):
        """
        Construye un dataset para inpainting:
        X  -> microestados
        y  -> probabilidad de spin up = (spin+1)/2
        """

        # Crear/limpiar archivo final
        h5_path = os.path.join(self.folder, h5_filename)
        if os.path.exists(h5_path):
            os.remove(h5_path)

        total = 0
        f = h5py.File(h5_path, 'w')
        dX = dy = None

        # Recorrer todos los archivos de Ising del folder
        for ising_file in self.files:

            # Cargar spins
            self._load_file(self.folder + ising_file)
            X_full = np.asarray(self.load_all_spins())   # (N, 10, 10)
            y_prob = (X_full + 1) / 2                     # +1->1  , -1->0

            # Guardar los microestados 
            if dX is None:
                # primera creación
                dX = f.create_dataset(
                    'X', data=X_full,
                    maxshape=(None,) + X_full.shape[1:],
                    chunks=True
                )
                dy = f.create_dataset(
                    'y', data=y_prob,
                    maxshape=(None,) + y_prob.shape[1:],
                    chunks=True
                )

            else:
                # append normal
                m = len(X_full)

                dX.resize((total + m), axis=0)
                dy.resize((total + m), axis=0)

                dX[total:total+m] = X_full
                dy[total:total+m] = y_prob

            total += len(X_full)

        f.close()

        self.h5_path = h5_path
        self.n_samples = total

        return self.h5_path

class H5IsingDataset(Dataset):
    """Dataset que sabe leer específicamente el dataset de Ising en formato HDF5"""

    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.file = h5py.File(self.h5_path, 'r')

        self.X = self.file['X']
        self.y = self.file['y']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

    def close(self):
        """Cerrar el archivo HDF5"""
        if self.file:
            self.file.close()
            self.file = None

    def __del__(self):
        """Cerrar el archivo automáticamente cuando se destruye el objeto"""
        if hasattr(self, 'file') and self.file:
            self.file.close()


