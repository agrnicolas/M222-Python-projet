# pair_selector.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from itertools import combinations
import config as cfg
import warnings

warnings.filterwarnings("ignore")


def trouver_paires() -> bool:
    print("[PAIRS] Recherche des paires cointegrées...")

    try:
        df = pd.read_csv(cfg.FICHIER_LOG_PRIX, index_col=0, parse_dates=True).sort_index()
    except FileNotFoundError:
        print("[PAIRS] Fichier log-prix manquant. Lance DATA d'abord.")
        return False

    print(f"[PAIRS] Donnees: shape={df.shape} | range={df.index.min()} -> {df.index.max()}")

    actifs = df.columns.tolist()
    paires = list(combinations(actifs, 2))

    candidats = []
    meilleurs = []

    for a1, a2 in paires:
        s1 = df[a1].dropna()
        s2 = df[a2].dropna()
        idx = s1.index.intersection(s2.index)
        if len(idx) < 500:
            continue
        s1 = s1.loc[idx]
        s2 = s2.loc[idx]

        if float(s1.std()) == 0.0 or float(s2.std()) == 0.0:
            continue

        try:
            _, pval, _ = coint(s1, s2)
        except Exception:
            continue

        if not np.isfinite(pval):
            continue

        meilleurs.append((float(pval), a1, a2))
        if pval >= cfg.SEUIL_PVALUE_CANDIDATS:
            continue

        try:
            X = sm.add_constant(s2)
            m = sm.OLS(s1, X).fit()
            beta = float(m.params[a2])
            residus = m.resid
        except Exception:
            continue

        ecart_type = float(residus.std())
        if ecart_type <= 0 or (not np.isfinite(ecart_type)):
            continue

        jumps = int((np.abs(residus) > (3 * ecart_type)).sum())
        if jumps >= cfg.SEUIL_JUMPS:
            continue

        rang = 1 if pval < 0.05 else 2

        candidats.append({
            "Pair": f"{a1}/{a2}",
            "Asset1": a1,
            "Asset2": a2,
            "Rank": rang,
            "P_Value": round(float(pval), 6),
            "Beta": round(beta, 6),
            "Jumps": jumps
        })

    if meilleurs:
        meilleurs = sorted(meilleurs, key=lambda x: x[0])
        print("[PAIRS] Top 10 p-values:")
        for p, a1, a2 in meilleurs[:10]:
            print(f"   {a1}/{a2}  p-value={p:.4f}")

    if not candidats:
        print("[PAIRS] Aucune paire candidate.")
        pd.DataFrame().to_csv(cfg.FICHIER_CANDIDATS, index=False)
        return False

    df_cand = pd.DataFrame(candidats).sort_values(["Rank", "P_Value"], ascending=[True, True])
    df_cand.to_csv(cfg.FICHIER_CANDIDATS, index=False)

    print(f"[PAIRS] {len(df_cand)} paires candidates sauvegardees: {cfg.FICHIER_CANDIDATS}")
    print(df_cand[["Pair", "Rank", "P_Value", "Jumps"]].head(15))
    return True
