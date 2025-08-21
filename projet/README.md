# Détection d'intrusions IoT avec Apprentissage Fédéré PCA

Ce projet implémente un système de détection d'intrusions en temps réel pour les environnements IoT en utilisant l'analyse en composantes principales (PCA) dans un contexte d'apprentissage fédéré.

## Structure du Projet

```
projet/
├── requirements.txt    # Dépendances du projet
├── main.py            # Script principal du client PCA fédéré
├── server.py          # Serveur d'apprentissage fédéré
├── compare_performance.py  # Script de comparaison des performances
└── README.md          # Documentation du projet
```

## Installation

1. Créez un environnement virtuel Python :
```bash
python -m venv venv
```

2. Activez l'environnement virtuel :
- Windows :
```bash
venv\Scripts\activate
```
- Linux/Mac :
```bash
source venv/bin/activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Démarrez le serveur d'apprentissage fédéré :
```bash
python server.py
```

2. Dans des terminaux séparés, lancez plusieurs clients (simulant différents appareils IoT) :
```bash
python main.py
```

3. Pour comparer les performances entre l'approche centralisée et fédérée :
```bash
python compare_performance.py
```

## Fonctionnalités

- **Apprentissage Fédéré** : Utilisation du framework Flower pour l'apprentissage distribué
- **PCA** : Réduction de dimensionnalité pour la détection d'anomalies
- **Comparaison des Performances** : Analyse comparative entre approches centralisée et fédérée
- **Visualisation** : Génération de graphiques comparatifs

## Personnalisation

1. **Dataset** : Remplacez la fonction `load_data()` dans `main.py` avec votre propre dataset IoT
2. **Paramètres PCA** : Ajustez le nombre de composantes principales dans la classe `IoTPCAClient`
3. **Configuration Fédérée** : Modifiez les paramètres du serveur dans `server.py`

## Métriques d'Évaluation

- Erreur de reconstruction
- Temps d'exécution
- Ratio de variance expliquée

## Notes

- Le système est actuellement configuré pour 2 clients minimum
- Les données sont normalisées avant l'application de PCA
- L'apprentissage fédéré se fait sur 3 rounds par défaut