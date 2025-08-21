import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import flwr as fl
import torch
import matplotlib.pyplot as plt

class IoTPCAClient(fl.client.NumPyClient):
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        self.pca = PCA(n_components=2)  # Réduire à 2 composantes principales
        self.scaler = StandardScaler()

    def fit(self, parameters, config):
        # Normalisation des données
        X_scaled = self.scaler.fit_transform(self.X_train)
        
        # Entraînement PCA
        self.pca.fit(X_scaled)
        
        # Retourner les paramètres du modèle
        return self.pca.components_.flatten(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        # Reconstruction des composantes principales
        components = parameters.reshape(-1, 2)
        self.pca.components_ = components
        
        # Normalisation et transformation des données de test
        X_test_scaled = self.scaler.transform(self.X_test)
        X_test_transformed = self.pca.transform(X_test_scaled)
        X_test_reconstructed = self.pca.inverse_transform(X_test_transformed)
        
        # Calcul de l'erreur de reconstruction
        reconstruction_error = np.mean(np.square(X_test_scaled - X_test_reconstructed))
        
        return reconstruction_error, len(self.X_test), {"reconstruction_error": reconstruction_error}

def load_data():
    # TODO: Charger votre dataset IoT ici
    # Exemple avec un dataset synthétique
    X = np.random.randn(1000, 10)  # 1000 échantillons avec 10 caractéristiques
    return X

def main():
    # Chargement des données
    X = load_data()
    
    # Division des données pour simulation fédérée
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # Création du client
    client = IoTPCAClient(X_train, X_test)
    
    # Démarrage du client Flower
    fl.client.start_numpy_client(server_address="[::]:8080", client=client)

if __name__ == "__main__":
    main()