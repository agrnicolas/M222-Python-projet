# config.py

# API BINANCE
# Cles API
BINANCE_API_KEY = ''
BINANCE_API_SECRET = ''

# DONNÉES
RAW_FILE = "crypto_log_prices.csv"     # Fichier contenant les données propres
CANDIDATES_FILE = "candidates.csv"     # Fichier intermédiaire (Pairs sélectionnées)
FINAL_FILE = "validated_pairs.csv"     # Fichier final (Pairs robustes)

START_DATE = "2024-09-01"
END_DATE = "2025-01-01"
TRAIN_END_DATE = "2024-12-01"
TIMEFRAME = "1m"

# Liste des cryptos
SYMBOLS = [
    "SFPUSDT", "PONDUSDT",
    "DATAUSDT", "SCUSDT",
    "DEXEUSDT", "ERNUSDT",
    "BTCUSDT", "ETHUSDT",
    "BNBUSDT", "SOLUSDT",
    "XRPUSDT", "ADAUSDT"
]

# PARAMÈTRES STATISTIQUES 
P_VALUE_THRESHOLD = 0.05    # Seuil critique cointégration
ECM_JUMPS_THRESHOLD = 20    # Max sauts violents autorisés par an

# PARAMÈTRES ROLLING BACKTEST (Robustesse de la cointegration)
ROLLING_WINDOW = 43200    # Fenêtre glissante pour a la minute
ROLLING_STEP = 10080           # Calcul /semaine
MIN_ROBUSTNESS = 75         # La paire doit être cointégrée X% du temps