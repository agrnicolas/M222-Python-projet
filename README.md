# M222-Python-projet
Projet dans le cadre du cours de python du Master 222. Code de trading quantitatif sur Binance. Enchaîne screening, tests de robustesse et algo génétique pour identifier des paires cointégrées stables et rentables sans biais d'anticipation.

# Crypto Pair Trading 

Ce projet implémente un pipeline complet de Pair Trading sur le marché des crypto-monnaies (Binance). Il utilise des méthodes statistiques (Cointégration, VECM, Validation Walk-Forward) pour identifier des paires d'actifs corrélées et exploiter leurs écarts de prix (spread) via un retour à la moyenne.

## Fonctionnalités Clés

* **Log-Price Analysis :** Traitement mathématique sur les logarithmes des prix pour linéariser les ratios.
* **Filtre ECM (Error Correction Model) :** Validation de la force de rappel du spread (Double Réversion).
* **Robustesse Glissante :** Vérification de la stabilité de la cointégration sur des fenêtres temporelles (Rolling Window).
* **Optimisation par IA :** Algorithme Génétique (via `DEAP`) pour trouver les seuils d'entrée/sortie optimaux.
* **Simulation Walk-Forward :** Test final sur une période "future" inconnue du modèle d'entraînement.

## Architecture du Projet

Le code est modulaire pour faciliter la maintenance et l'évolution :

| Fichier | Description |
| `main.py` | Point d'entrée. Tableau de bord CLI pour orchestrer les étapes. |
| `config.py` | Configuration globale (Clés API, Périodes, Paramètres mathématiques). |
| `data_processor.py` | Téléchargement des données historiques Binance et nettoyage. |
| `pair_selector.py` | Scan des paires cointégrées (Engle-Granger) et filtres ECM. |
| `robustness.py` | Test de robustesse via fenêtres glissantes (Multiprocessing inclus). |
| `backtester.py` | Optimisation des paramètres (Entry/Exit/Stop) sur le jeu d'entraînement. |
| `live_simulator.py` | Simulation "Paper Trading" sur les données hors-échantillon. |
| **`research/`** | **Dossier contenant les notebooks, tests préliminaires etc** |

## Installation

1.  **Télécharger le codde :**


2.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
    *(Assurez-vous d'avoir : pandas, numpy, statsmodels, binance-connector, deap, matplotlib)*

3.  **Configuration :**
    Modifiez le fichier `config.py` avec vos clés API Binance (optionnel pour le backtest sur données déjà téléchargées) et ajustez la `TRAIN_END_DATE`.

## Utilisation (Workflow)

Lancez le programme principal :
python main.py
