import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

def evaluate_centralized_pca(X_train, X_test, n_components=2):
    # Mesure du temps de début
    start_time = time.time()
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA centralisé
    pca = PCA(n_components=n_components)
    pca.fit(X_train_scaled)
    
    # Transformation et reconstruction
    X_test_transformed = pca.transform(X_test_scaled)
    X_test_reconstructed = pca.inverse_transform(X_test_transformed)
    
    # Calcul de l'erreur de reconstruction
    reconstruction_error = np.mean(np.square(X_test_scaled - X_test_reconstructed))
    
    # Temps d'exécution
    execution_time = time.time() - start_time
    
    return {
        'reconstruction_error': reconstruction_error,
        'execution_time': execution_time,
        'explained_variance_ratio': pca.explained_variance_ratio_
    }

def plot_comparison(centralized_metrics, federated_metrics):
    # Création d'un graphique comparatif
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Comparaison des erreurs de reconstruction
    errors = [centralized_metrics['reconstruction_error'], federated_metrics['reconstruction_error']]
    ax1.bar(['Centralisé', 'Fédéré'], errors)
    ax1.set_title('Erreur de reconstruction')
    ax1.set_ylabel('Erreur moyenne quadratique')
    
    # Comparaison des temps d'exécution
    times = [centralized_metrics['execution_time'], federated_metrics['execution_time']]
    ax2.bar(['Centralisé', 'Fédéré'], times)
    ax2.set_title('Temps d\'exécution')
    ax2.set_ylabel('Temps (secondes)')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.close()

def main():
    # TODO: Charger votre dataset IoT
    # Exemple avec données synthétiques
    X = np.random.randn(1000, 10)
    
    # Division des données
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # Évaluation de l'approche centralisée
    centralized_metrics = evaluate_centralized_pca(X_train, X_test)
    
    # Les métriques de l'approche fédérée seront collectées pendant l'exécution
    # du serveur et des clients. Pour cet exemple, nous utilisons des valeurs fictives
    federated_metrics = {
        'reconstruction_error': 0.0,  # À remplacer par la vraie valeur
        'execution_time': 0.0,        # À remplacer par la vraie valeur
    }
    
    # Affichage des résultats
    print("\nRésultats de l'analyse comparative :")
    print("\nApproche centralisée :")
    print(f"Erreur de reconstruction : {centralized_metrics['reconstruction_error']:.4f}")
    print(f"Temps d'exécution : {centralized_metrics['execution_time']:.2f} secondes")
    print(f"Variance expliquée : {centralized_metrics['explained_variance_ratio']}")
    
    print("\nApproche fédérée :")
    print(f"Erreur de reconstruction : {federated_metrics['reconstruction_error']:.4f}")
    print(f"Temps d'exécution : {federated_metrics['execution_time']:.2f} secondes")
    
    # Création du graphique comparatif
    plot_comparison(centralized_metrics, federated_metrics)

if __name__ == "__main__":
    main()