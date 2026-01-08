# config.py
import os

# =========================
# API BINANCE
# =========================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "ta_cle_api")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "ton_code_secret_api")

# =========================
# FICHIERS
# =========================
FICHIER_LOG_PRIX = "crypto_log_prices.csv"
FICHIER_CANDIDATS = "candidates.csv"
FICHIER_VALIDES = "validated_pairs.csv"

# =========================
# DONNEES
# =========================
DATE_DEBUT = "2024-11-01"
DATE_FIN = "2025-01-01"
TIMEFRAME = "15m"

# =========================
# UNIVERS
# =========================
SYMBOLES = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT",
    "TRXUSDT","DOTUSDT","AVAXUSDT","LINKUSDT","MATICUSDT","LTCUSDT","BCHUSDT",
    "ETCUSDT","UNIUSDT","ATOMUSDT","XLMUSDT","NEARUSDT","ICPUSDT","APEUSDT",
    "FTMUSDT","SUIUSDT","SEIUSDT","OPUSDT","ARBUSDT","GMXUSDT","RNDRUSDT",
    "INJUSDT","GRTUSDT","AAVEUSDT","SNXUSDT","CRVUSDT","COMPUSDT","IMXUSDT",
    "APTUSDT","RLCUSDT","AGIXUSDT","OCEANUSDT"
]

# =========================
# SELECTION DES PAIRES
# =========================
SEUIL_PVALUE_CANDIDATS = 0.25
SEUIL_JUMPS = 120

# =========================
# ROBUSTESSE ROLLING (15m)
# =========================
FENETRE_ROLLING = 2880
PAS_ROLLING = 96
ROBUSTESSE_MIN = 70

# =========================
# NETTOYAGE
# =========================
PROPORTION_MIN_COLONNES = 0.80
LIMITE_FFILL = 8

# =========================
# TRADING (OU)
# =========================
TF_TRADING = "1h"
FRAIS_BPS = 10.0
FRAIS_DEUX_JAMBES = True

FENETRE_BETA = 48
FENETRE_Z = 72
FENETRE_OU = 96

Z_ENTREE = 2.5
Z_SORTIE = 0.5
Z_STOP = 6.0
MAX_HOLD = 96

DEMI_VIE_MAX = 40.0
Z_MAX_ABS = 10.0

# Nouveau: filtre AR(1)
AR1_B_MAX = 0.995
