import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from itertools import combinations
import config as cfg
import warnings

warnings.filterwarnings("ignore")

def find_best_pairs():
    print(f"\n[2/3] Recherche des paires cointégrées (sur le derniers mois)")
    
    try:
        # Chargement complet
        df_full = pd.read_csv(cfg.RAW_FILE, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Erreur: Lancez d'abord le téléchargement.")
        return
    
    # On s'assure que l'index est bien en DateTime
    if not isinstance(df_full.index, pd.DatetimeIndex):
        df_full.index = pd.to_datetime(df_full.index)

    df_history = df_full.loc[:cfg.TRAIN_END_DATE].copy()
    
    # période de scan
    end_scan = df_history.index.max() 
    start_scan = end_scan - pd.Timedelta(days=30) 
    
    df = df_history.loc[start_scan:end_scan].copy()
    
    print(f" Date Max autorisée (Split) : {cfg.TRAIN_END_DATE}")
    print(f" Période de scan (Sélection) : du {start_scan} au {end_scan}")
    
    if len(df) < 1000:
        print("Attention : Moins de 1000 prix trouvés sur le dernier mois. Vérifiez vos données.")

    assets = df.columns.tolist()
    pairs = list(combinations(assets, 2))
    results = []
    
    print(f"Analyse de {len(pairs)} combinaisons")

    for asset1, asset2 in pairs:
        s1 = df[asset1]
        s2 = df[asset2]
        
        # Nettoyage des NA potentiels
        valid_idx = s1.dropna().index.intersection(s2.dropna().index)
        if len(valid_idx) < 100: continue
        s1, s2 = s1.loc[valid_idx], s2.loc[valid_idx]

        # Test Cointégration (Engle-Granger)
        try:
            # autolag=AIC pour trouver lag opti
            score, pvalue, _ = coint(s1, s2, autolag='AIC')
        except: continue
            
        if pvalue < cfg.P_VALUE_THRESHOLD:
            
            # Calcul du Spread et Beta (regression linéaire)
            X = sm.add_constant(s2)
            model = sm.OLS(s1, X).fit()
            beta = model.params[asset2]
            residuals = model.resid
            
            # ECM (Error Correction Model)
            ect = residuals.shift(1).dropna()
            d_y1 = s1.diff().dropna()
            d_y2 = s2.diff().dropna()
            
            # Alignement des index
            common_idx = ect.index.intersection(d_y1.index).intersection(d_y2.index)
            if len(common_idx) < 50: continue
            
            ect, d_y1, d_y2 = ect.loc[common_idx], d_y1.loc[common_idx], d_y2.loc[common_idx]
            
            try:
                X_ecm = sm.add_constant(ect)
                
                # Equation 1
                mod1 = sm.OLS(d_y1, X_ecm).fit()
                l1 = mod1.params[0] # Lambda 1
                pv1 = mod1.pvalues[0]
                
                # Equation 2
                mod2 = sm.OLS(d_y2, X_ecm).fit()
                l2 = mod2.params[0] # Lambda 2
                pv2 = mod2.pvalues[0]
                
                # Classification
                # Les deux actifs corrigent l'erreur ou pas
                is_double = (pv1 < 0.05 and l1 < 0) and (pv2 < 0.05 and l2 > 0)
                
                # Stabilité (Jumps)
                std_spread = residuals.std()
                if std_spread == 0: continue 
                
                jumps = (np.abs(residuals) > (3 * std_spread)).sum()
                is_stable = jumps < cfg.ECM_JUMPS_THRESHOLD
                
                rank = 99
                status = "Rejeté"
                
                if is_double and is_stable:
                    rank = 1
                    status = "Level 1 (Parfait)"
                elif is_double and not is_stable:
                    rank = 2
                    status = "Level 2 (Volatile)"
                
                if rank <= 2:
                    results.append({
                        "Pair": f"{asset1}/{asset2}",
                        "Asset1": asset1,
                        "Asset2": asset2,
                        "Rank": rank,
                        "Status": status,
                        "P_Value": round(pvalue, 5),
                        "Beta": round(beta, 4),
                        "Jumps": jumps
                    })
                    
            except: continue

    # Export
    if results:
        df_res = pd.DataFrame(results).sort_values(by=["Rank", "P_Value"])
        df_res.to_csv(cfg.CANDIDATES_FILE, index=False)
        print(f"{len(df_res)} Paires candidates trouvées (sauvegardées dans {cfg.CANDIDATES_FILE})")
        print(df_res[['Pair', 'Status', 'P_Value']].head())
        return True
    else:
        print("Aucune paire Level 1 ou 2 trouvée sur le dernier mois.")
        return False