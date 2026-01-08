# backtester.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from deap import base, creator, tools, algorithms
from sklearn.model_selection import TimeSeriesSplit
import random
import config as cfg
import warnings
import os

warnings.filterwarnings("ignore")

# Paramètres
# On optimise sur les 60 derniers jours de la période d'entraînement
OPTIMIZATION_DAYS = 60 
N_SPLITS = 3
GEN_ALGO = 10
POP_SIZE = 50

# SETUP DEAP
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

def get_toolbox(spread_series):
    toolbox = base.Toolbox()
    # Intervalle des paramètres
    toolbox.register("attr_window", random.randint, 60, 4320)
    toolbox.register("attr_entry", random.uniform, 1.5, 5.0)
    toolbox.register("attr_stop", random.uniform, 3.0, 10.0)
    toolbox.register("attr_exit", random.uniform, 0.0, 1.5)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_window, toolbox.attr_entry, toolbox.attr_stop, toolbox.attr_exit), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        window, entry, stop, exit_val = individual
        window = int(window)
        # Contraintes 
        if window < 50 or window > len(spread_series) // 2: return (-999,)
        if stop <= entry or exit_val >= entry: return (-999,)
        
        roll_mean = spread_series.rolling(window).mean()
        roll_std = spread_series.rolling(window).std()
        if roll_std.iloc[-1] == 0: return (-999,)
        
        z = (spread_series - roll_mean) / roll_std
        sigs = pd.Series(0, index=spread_series.index)
        sigs[z > entry] = -1
        sigs[z < -entry] = 1
        sigs[abs(z) < exit_val] = 0
        sigs[abs(z) > stop] = 0
        
        pos = sigs.replace(0, np.nan).ffill().fillna(0).shift(1)
        pnl = spread_series.diff() * pos
        
        if pnl.std() == 0: return (-999,)
        # Sharpe Ratio Annualisé (minutes)
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(24 * 60 * 365)
        return (sharpe,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

def analyze_trades(position_series, pnl_series):
    trade_ids = (position_series != position_series.shift(1)).cumsum()
    active_trades = pnl_series.groupby(trade_ids).sum()
    active_pos = position_series.groupby(trade_ids).first()
    real_trades_mask = active_pos != 0
    trade_pnls = active_trades[real_trades_mask]
    
    if len(trade_pnls) == 0: return 0, 0, 0
    
    nb_trades = len(trade_pnls)
    gross_wins = trade_pnls[trade_pnls > 0].sum()
    gross_losses = abs(trade_pnls[trade_pnls < 0].sum())
    
    win_rate = (len(trade_pnls[trade_pnls > 0]) / nb_trades) * 100
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else 999
    return nb_trades, win_rate, profit_factor

def run_final_validation():
    print(f"\n" + "="*60)
    print(f"BACKTEST & OPTIMISATION (TRAINING SET)")
    print(f"Date limite apprentissage : {cfg.TRAIN_END_DATE}")
    print(f"Optimisation sur les derniers {OPTIMIZATION_DAYS} jours avant le split")
    print("="*60)

    if not os.path.exists(cfg.FINAL_FILE):
        print(f"Fichier {cfg.FINAL_FILE} manquant. Lancez l'étape 2 (Robustesse).")
        return

    candidates = pd.read_csv(cfg.FINAL_FILE)
    try:
        df_raw = pd.read_csv(cfg.RAW_FILE, index_col=0, parse_dates=True)
    except:
        print("Erreur chargement données.")
        return
    
    if cfg.TRAIN_END_DATE not in df_raw.index:
        df_train = df_raw.loc[:cfg.TRAIN_END_DATE]
    else:
        df_train = df_raw.loc[:cfg.TRAIN_END_DATE]

    print(f"Données chargées jusqu'au {df_train.index.max()}")
    
    live_strategies = [] 
    
    for idx, row in candidates.iterrows():
        asset1, asset2 = row['Asset1'], row['Asset2']
        if asset1 not in df_train.columns or asset2 not in df_train.columns: continue
            
        print(f"\nOptimisation {asset1}/{asset2}...")
        
        # On prépare les données pour l'optimisation
        pair_df = df_train[[asset1, asset2]].dropna()
        
        # On prend la fenêtre d'optimisation
        start_opti = pair_df.index[-1] - pd.Timedelta(days=OPTIMIZATION_DAYS)
        pair_df = pair_df.loc[start_opti:]
        
        if len(pair_df) < 1000:
            print("Pas assez de données pour optimiser.")
            continue

        # Hedge Ratio sur cette période
        X_global = sm.add_constant(pair_df[asset2])
        model = sm.OLS(pair_df[asset1], X_global).fit()
        hedge_ratio = model.params[asset2]
        spread_full = pair_df[asset1] - (hedge_ratio * pair_df[asset2])
        
        # Walk-Forward Optimization
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        hist_pnl, hist_pos = [], []
        last_best_params = None 

        for train_idx, test_idx in tscv.split(spread_full):
            spread_train = spread_full.iloc[train_idx]
            spread_test = spread_full.iloc[test_idx]
            
            # Algo Génétique sur le train set du split
            toolbox = get_toolbox(spread_train)
            pop = toolbox.population(n=POP_SIZE)
            algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=GEN_ALGO, verbose=False)
            
            best_ind = tools.selBest(pop, 1)[0]
            last_best_params = best_ind
            
            # Test sur test
            win, entry, stop, exit_val = best_ind
            win = int(win)
            
            # On reconstitue le spread test
            start_roll = spread_train.iloc[-win:] if len(spread_train) > win else spread_train
            spread_fold = pd.concat([start_roll, spread_test])
            
            roll_mean = spread_fold.rolling(win).mean()
            roll_std = spread_fold.rolling(win).std()
            z_score = (spread_fold - roll_mean) / roll_std
            z_score = z_score.iloc[len(start_roll):] 
            
            sigs = pd.Series(0, index=z_score.index)
            sigs[z_score > entry] = -1
            sigs[z_score < -entry] = 1
            sigs[abs(z_score) < exit_val] = 0
            sigs[abs(z_score) > stop] = 0
            
            pos = sigs.replace(0, np.nan).ffill().fillna(0).shift(1)
            pnl = spread_test.loc[pos.index].diff() * pos
            
            hist_pnl.append(pnl); hist_pos.append(pos)

        # Analyse globale du backtest In-Sample
        full_pnl = pd.concat(hist_pnl)
        full_pos = pd.concat(hist_pos)
        nb, win_rate, pf = analyze_trades(full_pos, full_pnl)
        
        print(f"-> In-Sample Result : PnL={full_pnl.sum():.4f} | Trades={nb} | PF={pf:.2f}")

        # SAUVEGARDE
        # Si les seuils ont été rentable en entraînement, on garde
        if pf > 1.0 and nb >= 3: 
            win, entry, stop, exit_val = last_best_params
            live_strategies.append({
                "Asset1": asset1,
                "Asset2": asset2,
                "HedgeRatio": hedge_ratio, 
                "Window": int(win),
                "Entry": entry,
                "Stop": stop,
                "Exit": exit_val
            })
            print("Stratégie validée et prête pour le Futur.")

    # Export
    if live_strategies:
        pd.DataFrame(live_strategies).to_csv("live_strategies.csv", index=False)
        print(f"\n💾 {len(live_strategies)} stratégies exportées vers 'live_strategies.csv'.")
    else:
        print("\Aucune stratégie n'a survécu à l'entraînement.")

if __name__ == "__main__":
    run_final_validation()