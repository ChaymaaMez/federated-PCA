import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
from flwr.server.client_proxy import ClientProxy

class PCAStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, fl.common.FitRes]], failures: List[BaseException]) -> Optional[fl.common.Parameters]:
        # Agréger les composantes principales de tous les clients
        if not results:
            return None

        # Extraire les composantes principales et les poids
        weights_results = [
            (fit_res.parameters, fit_res.num_examples)
            for _, fit_res in results
        ]

        # Calculer la moyenne pondérée des composantes principales
        parameters_aggregated = self.aggregate_parameters(weights_results)
        
        if parameters_aggregated is None:
            return None
            
        return parameters_aggregated

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, fl.common.EvaluateRes]], failures: List[BaseException]) -> Optional[Tuple[float, Dict[str, float]]]:
        if not results:
            return None

        # Calculer l'erreur de reconstruction moyenne pondérée
        reconstruction_errors = []
        examples = []
        
        for _, eval_res in results:
            reconstruction_errors.append(eval_res.metrics["reconstruction_error"] * eval_res.num_examples)
            examples.append(eval_res.num_examples)

        # Afficher les métriques pour chaque client
        for idx, (_, eval_res) in enumerate(results):
            print(f"Client {idx+1} - Erreur de reconstruction: {eval_res.metrics['reconstruction_error']:.6f} (sur {eval_res.num_examples} exemples)")

        avg_reconstruction_error = sum(reconstruction_errors) / sum(examples)
        print(f"\nErreur de reconstruction moyenne: {avg_reconstruction_error:.6f}")
        
        return avg_reconstruction_error, {"reconstruction_error": float(avg_reconstruction_error)}

def main():
    # Stratégie d'agrégation personnalisée
    strategy = PCAStrategy(
        min_fit_clients=1,  # Nombre minimum de clients pour l'entraînement
        min_available_clients=1,  # Nombre minimum de clients disponibles
    )

    # Démarrer le serveur Flower
    fl.server.start_server(
        server_address="127.0.0.1:8888",
        config=fl.server.ServerConfig(num_rounds=3),  # Nombre de rounds d'apprentissage fédéré
        strategy=strategy
    )

if __name__ == "__main__":
    main()