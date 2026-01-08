# data_processor.py
import pandas as pd
import numpy as np
import time
from binance.client import Client
import config as cfg


def telecharger_et_nettoyer() -> bool:
    print("[DATA] Téléchargement et nettoyage des données...")

    try:
        client = Client(cfg.BINANCE_API_KEY, cfg.BINANCE_API_SECRET)
    except Exception as e:
        print(f"[DATA] Erreur connexion API: {e}")
        return False

    liste_dfs = []

    for symbole in cfg.SYMBOLES:
        try:
            print(f"[DATA] Récupération: {symbole}")
            klines = client.get_historical_klines(symbole, cfg.TIMEFRAME, cfg.DATE_DEBUT, cfg.DATE_FIN)

            if not klines:
                print(f"[DATA] Aucune donnée pour {symbole}")
                continue

            data = pd.DataFrame(
                klines,
                columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "qav", "trades", "tbb", "tbq", "ignore"
                ],
            )
            data["open_time"] = pd.to_datetime(data["open_time"], unit="ms")
            data.set_index("open_time", inplace=True)
            data["close"] = pd.to_numeric(data["close"], errors="coerce")

            data = data[["close"]].rename(columns={"close": symbole})
            data = data.sort_index()
            data = data[~data.index.duplicated(keep="last")]

            liste_dfs.append(data)
            time.sleep(0.15)

        except Exception as e:
            print(f"[DATA] Erreur sur {symbole}: {e}")

    if not liste_dfs:
        print("[DATA] Aucune donnée téléchargée.")
        return False

    df_prix = pd.concat(liste_dfs, axis=1).sort_index()
    df_prix = df_prix.replace([np.inf, -np.inf, 0], np.nan)

    nb_min = int(cfg.PROPORTION_MIN_COLONNES * df_prix.shape[1])
    df_prix = df_prix.dropna(thresh=nb_min)
    df_prix = df_prix.ffill(limit=cfg.LIMITE_FFILL).dropna()

    df_log = np.log(df_prix)
    df_log.to_csv(cfg.FICHIER_LOG_PRIX)

    print(f"[DATA] Données sauvegardées dans: {cfg.FICHIER_LOG_PRIX}")
    print(f"[DATA] Shape: {df_log.shape}")
    print(f"[DATA] Range: {df_log.index.min()} -> {df_log.index.max()}")
    return True
