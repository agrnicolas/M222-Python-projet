# live_simulator.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
import warnings
import os

warnings.filterwarnings("ignore")

# CONFIGURATION LIVE 
TRANSACTION_FEE = 0.001   
INITIAL_CAPITAL = 1000     

def run_simulation():
    print(f"\n" + "="*60)
    print(f"SIMULATION LIVE (OUT-OF-SAMPLE)")
    print(f"Date Début Live : {cfg.TRAIN_END_DATE}")
    print(f"Données utilisées : Strictement postérieures au Split")
    print(f"Frais par ordre   : {TRANSACTION_FEE*100}%")
    print("="*60)

    if not os.path.exists("live_strategies.csv"):
        print("'live_strategies.csv' introuvable. Lancez le Backtest d'abord.")
        return

    try:
        strategies = pd.read_csv("live_strategies.csv")
        df_all = pd.read_csv(cfg.RAW_FILE, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Erreur lecture fichiers : {e}")
        return

    if strategies.empty:
        print("Aucune stratégie à tester.")
        return

    # Date de démarrage du live sur données jamais utilisées (derniers mois)
    start_live_date = pd.to_datetime(cfg.TRAIN_END_DATE)

    for idx, row in strategies.iterrows():
        asset1, asset2 = row['Asset1'], row['Asset2']
        hedge_ratio = row['HedgeRatio']
        win = int(row['Window'])
        entry = row['Entry']
        stop_thresh = row['Stop']
        exit_thresh = row['Exit']

        print(f"\nSimulation Live : {asset1} / {asset2}")

        if asset1 not in df_all.columns or asset2 not in df_all.columns: continue

        # Préparation données
        buffer_start = start_live_date - pd.Timedelta(minutes=win + 2000)
        raw_data = df_all.loc[buffer_start:][[asset1, asset2]].dropna()
        
        # Vérif si on a des données futures
        if raw_data.index.max() <= start_live_date:
            print("Pas de données futures (après le split). Simulation impossible.")
            continue

        # Calcul des indicateurs
        spread = raw_data[asset1] - (hedge_ratio * raw_data[asset2])
        roll_mean = spread.rolling(win).mean()
        roll_std = spread.rolling(win).std()
        z_score = (spread - roll_mean) / roll_std
        # On ne garde que les données strictement supérieures à la date de fin d'entraînement
        z_score_live = z_score.loc[start_live_date:]
        spread_live = spread.loc[start_live_date:]
        
        if len(z_score_live) == 0:
            print("Erreur découpage : Aucune donnée en zone Live.")
            continue
            
        print(f"Début du trading : {z_score_live.index[0]}")
        print(f"Fin du trading   : {z_score_live.index[-1]}")

        # Strat
        position = 0 
        capital = INITIAL_CAPITAL
        equity_curve = [capital]
        trades_count = 0
        fees_paid = 0
        
        for t in range(len(z_score_live)):
            z_val = z_score_live.iloc[t]
            current_price = spread_live.iloc[t]
            
            # PnL Latent
            if t > 0 and position != 0:
                prev_price = spread_live.iloc[t-1]
                # PnL approx sur Log-Spread
                step_pnl = (current_price - prev_price) * position * capital
                equity_curve.append(equity_curve[-1] + step_pnl)
            else:
                equity_curve.append(equity_curve[-1])
            
            # Signaux
            new_position = position
            
            if position == 0:
                if z_val > entry: new_position = -1
                elif z_val < -entry: new_position = 1
            else:
                if abs(z_val) < exit_thresh: new_position = 0
                elif abs(z_val) > stop_thresh: new_position = 0

            # Exécution
            if new_position != position:
                fee_cost = capital * TRANSACTION_FEE
                equity_curve[-1] -= fee_cost
                fees_paid += fee_cost
                
                if position == 0: trades_count += 1
                position = new_position

        # résultat
        final_equity = equity_curve[-1]
        net_pnl = final_equity - INITIAL_CAPITAL
        
        print(f"Résultat : {net_pnl:.2f} $ ({(net_pnl/INITIAL_CAPITAL)*100:.2f}%) | Trades: {trades_count}")

        # Graphique
        plt.figure(figsize=(10, 4))
        plt.plot(z_score_live.index, equity_curve[:-1], label='Equity ($)', color='#2ecc71')
        plt.title(f"Live Simulation (Out-of-Sample): {asset1}/{asset2}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    run_simulation()