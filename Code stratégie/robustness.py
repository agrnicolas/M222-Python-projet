# robustness.py
import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import config as cfg
import warnings

warnings.filterwarnings("ignore")

def check_rolling_cointegration(series1, series2, window, step):
    """
    Vérifie la cointégration sur une fenêtre glissante.
    Retourne : 
    - Le score de robustesse (% de fenêtres cointégrées)
    - La p-value de la DERNIÈRE fenêtre (État actuel)
    """
    df = pd.concat([series1, series2], axis=1).dropna()
    s1 = df.iloc[:, 0]
    s2 = df.iloc[:, 1]
    
    total_windows = 0
    coint_windows = 0
    last_p_value = 1.0 
    
    # Boucle Rolling
    for i in range(0, len(df) - window, step):
        w1 = s1.iloc[i : i + window]
        w2 = s2.iloc[i : i + window]
        
        # Test de Cointégration (Engle-Granger)
        # return: t-stat, p-value, crit_values
        try:
            result = ts.coint(w1, w2)
            p_value = result[1]
            
            if p_value < cfg.P_VALUE_THRESHOLD:
                coint_windows += 1
            
            # et on stocke la p-value de la dernière itération (pour vérifier/avoir au moins des pairs pour le reste du code)
            last_p_value = p_value
            
        except:
            pass 
            
        total_windows += 1
        
    if total_windows == 0:
        return 0, 1.0
    
    robustness_score = (coint_windows / total_windows) * 100
    return robustness_score, last_p_value

def run_robustness_test():
    print(f"\n" + "="*50)
    print(f"TEST DE ROBUSTESSE")
    print(f"Fenêtre : {cfg.ROLLING_WINDOW} minutes")
    print(f"Pas    : {cfg.ROLLING_STEP} minutes")
    print("="*50)

    # Chargement
    try:
        candidates = pd.read_csv(cfg.CANDIDATES_FILE)
        df_raw = pd.read_csv(cfg.RAW_FILE, index_col=0, parse_dates=True)
        df_all = df_raw.loc[:cfg.TRAIN_END_DATE]
        print(f"Robustesse testée sur la période : {df_all.index.min()} -> {df_all.index.max()}")
    except FileNotFoundError:
        print("Fichiers manquants. Lancez l'étape 1 et 2.")
        return

    validated_results = []
    
    print(f"🧪 Analyse de {len(candidates)} paires candidates...")

    # Boucle d'analyse
    for idx, row in candidates.iterrows():
        asset1, asset2 = row['Asset1'], row['Asset2']
        
        if asset1 not in df_all.columns or asset2 not in df_all.columns:
            continue
            
        score, last_pval = check_rolling_cointegration(
            df_all[asset1], 
            df_all[asset2], 
            window=cfg.ROLLING_WINDOW, 
            step=cfg.ROLLING_STEP
        )
        
        status = "Good" if score > cfg.MIN_ROBUSTNESS else "Probleme"
        print(f"   • {asset1}/{asset2} : Score={score:.1f}% | Last P-Val={last_pval:.4f} {status}")
        
        validated_results.append({
            "Asset1": asset1,
            "Asset2": asset2,
            "Robustness": score,
            "Last_P_Value": last_pval
        })

    # Filtrage suivant ce qu'on trouve pour la suite  du code
    df_res = pd.DataFrame(validated_results)
    
    if df_res.empty:
        print("Aucune paire analysée.")
        return

    # Méthode opti : crtières stricts  (ceux normalement voulus)
    strong_pairs = df_res[df_res['Robustness'] >= cfg.MIN_ROBUSTNESS]
    
    if not strong_pairs.empty:
        print(f"\n✅ {len(strong_pairs)} Paires robustes trouvées (> {cfg.MIN_ROBUSTNESS}%).")
        final_selection = strong_pairs
        
    else:
        # Methode pas opti : Selection avec la dernière P-value
        print(f"\nAUCUNE PAIRE ROBUSTE (> {cfg.MIN_ROBUSTNESS}%).")
        print("Selection des dernières P-value (Last P-Value < 0.05)")
        
        # On prend celles qui sont cointégrées sur la dernière fenêtre
        current_opps = df_res[df_res['Last_P_Value'] < 0.05]
        
        if not current_opps.empty:
            print(f"{len(current_opps)} Paires récupérées (Cointégrées actuellement).")
            final_selection = current_opps
        else:
            print("Aucune paire n'est cointégrée, même récemment.")
            return

    # 4. Sauvegarde
    final_selection.to_csv(cfg.FINAL_FILE, index=False)
    print(f"Sélection sauvegardée dans '{cfg.FINAL_FILE}'")

if __name__ == "__main__":
    run_robustness_test()