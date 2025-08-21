import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import flwr as fl
import torch
import matplotlib.pyplot as plt
import os

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
        
        # Calculer l'erreur de reconstruction pour les données d'entraînement
        X_transformed = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        reconstruction_error = np.mean(np.square(X_scaled - X_reconstructed))
        
        # Retourner les paramètres du modèle dans le format attendu
        parameters = [self.pca.components_.flatten()]
        
        return parameters, len(self.X_train), {"reconstruction_error": float(reconstruction_error)}

    def evaluate(self, parameters, config):
        # Reconstruction des composantes principales
        components = parameters[0].reshape(-1, 2)
        self.pca.components_ = components
        
        # Normalisation et transformation des données de test
        X_test_scaled = self.scaler.transform(self.X_test)
        X_test_transformed = self.pca.transform(X_test_scaled)
        X_test_reconstructed = self.pca.inverse_transform(X_test_transformed)
        
        # Calcul de l'erreur de reconstruction
        reconstruction_error = np.mean(np.square(X_test_scaled - X_test_reconstructed))
        
        print(f"Erreur de reconstruction sur les données de test: {reconstruction_error}")
        return float(reconstruction_error), len(self.X_test), {"reconstruction_error": float(reconstruction_error)}

def load_data(client_id=1):
    # Charger les données du client spécifié
    train_file = f"abnormal_detection_data/train/client{client_id}_preprocessed.csv"
    test_file = "abnormal_detection_data/test/abnormal_test.csv"
    
    # Vérifier si les fichiers existent
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"Les fichiers de données {train_file} ou {test_file} n'existent pas.")
    
    # Charger les données
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    # Supprimer la première colonne si c'est un index
    if train_data.columns[0].isdigit() or train_data.columns[0] == 'Unnamed: 0':
        train_data = train_data.drop(train_data.columns[0], axis=1)
    if test_data.columns[0].isdigit() or test_data.columns[0] == 'Unnamed: 0':
        test_data = test_data.drop(test_data.columns[0], axis=1)
    
    # Convertir en numpy array et supprimer les colonnes non numériques
    X_train = train_data.select_dtypes(include=[np.number]).values
    X_test = test_data.select_dtypes(include=[np.number]).values
    
    print(f"Données chargées - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test

def main():
    # Chargement des données pour le client 1
    X_train, X_test = load_data(client_id=1)
    
    # Création du client
    client = IoTPCAClient(X_train, X_test)
    
    # Démarrage du client Flower
    fl.client.start_numpy_client(server_address="127.0.0.1:8888", client=client)

if __name__ == "__main__":
    main()