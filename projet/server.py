import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
from flwr.server.client_proxy import ClientProxy

class PCAStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, Dict]], failures: List[BaseException]) -> Optional[np.ndarray]:
        # Agréger les composantes principales de tous les clients
        if not results:
            return None

        # Extraire les composantes principales et les poids
        weights_results = [
            (parameters, num_examples)
            for client_proxy, (parameters, num_examples, _) in results
        ]

        # Calculer la moyenne pondérée des composantes principales
        return self.aggregate_parameters(weights_results)

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, Dict]], failures: List[BaseException]) -> Optional[float]:
        if not results:
            return None

        # Calculer l'erreur de reconstruction moyenne pondérée
        reconstruction_errors = [r[1]["reconstruction_error"] * r[2] for _, r in results]
        examples = [r[2] for _, r in results]

        return sum(reconstruction_errors) / sum(examples)

def main():
    # Stratégie d'agrégation personnalisée
    strategy = PCAStrategy(
        min_fit_clients=2,  # Nombre minimum de clients pour l'entraînement
        min_available_clients=2,  # Nombre minimum de clients disponibles
    )

    # Démarrer le serveur Flower
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=3),  # Nombre de rounds d'apprentissage fédéré
        strategy=strategy
    )

if __name__ == "__main__":
    main()