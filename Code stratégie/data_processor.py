# data_processor.py
import pandas as pd
import numpy as np
import time
from binance.client import Client
from binance.enums import *
import config as cfg

def download_and_clean():
    print("⬇[1/3] Démarrage du téléchargement et nettoyage...")
    
    try:
        client = Client(cfg.BINANCE_API_KEY, cfg.BINANCE_API_SECRET)
    except Exception as e:
        print(f"Erreur connexion API : {e}")
        return False

    all_dfs = []
    
    for symbol in cfg.SYMBOLS:
        try:
            print(f"Récupération {symbol}")
            klines = client.get_historical_klines(symbol, cfg.TIMEFRAME, cfg.START_DATE, cfg.END_DATE)
            
            # Création DF
            data = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'QAV', 'Trades', 'TBB', 'TBQ', 'Ignore'])
            data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
            data.set_index('Open time', inplace=True)
            data['Close'] = data['Close'].astype(float)
            
            # Renommage et sélection
            data = data[['Close']].rename(columns={'Close': symbol})
            all_dfs.append(data)
            
            time.sleep(0.2) # Anti-ban api binance
            
        except Exception as e:
            print(f"Erreur sur {symbol}: {e}")

    if not all_dfs:
        print("Aucune donnée téléchargée.")
        return False

    # Fusion
    df = pd.concat(all_dfs, axis=1)
    
    # NETTOYAGE
    print(" Nettoyage + log transfo")
    df = df.replace([np.inf, -np.inf, 0], np.nan)
    df.dropna(inplace=True)
    df_log = np.log(df)
    
    # Sauvegarde
    df_log.to_csv(cfg.RAW_FILE)
    print(f"Données sauvegardées dans '{cfg.RAW_FILE}'")
    return True