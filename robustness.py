# robustness.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import config as cfg
import warnings

warnings.filterwarnings("ignore")


def tester_robustesse():
    print("[ROBUST] Test de robustesse rolling...")

    try:
        df = pd.read_csv(cfg.FICHIER_LOG_PRIX, index_col=0, parse_dates=True).sort_index()
        candidats = pd.read_csv(cfg.FICHIER_CANDIDATS)
    except FileNotFoundError:
        print("[ROBUST] Fichiers manquants. Lance DATA puis PAIRS.")
        return

    if candidats.empty:
        print("[ROBUST] candidates.csv est vide.")
        pd.DataFrame().to_csv(cfg.FICHIER_VALIDES, index=False)
        return

    valides = []

    for _, ligne in candidats.iterrows():
        a1, a2 = ligne["Asset1"], ligne["Asset2"]
        nom = ligne["Pair"]

        s1 = df[a1].dropna()
        s2 = df[a2].dropna()
        idx = s1.index.intersection(s2.index)
        if len(idx) < cfg.FENETRE_ROLLING + cfg.PAS_ROLLING:
            continue

        s1 = s1.loc[idx]
        s2 = s2.loc[idx]

        dates, pvals, betas = [], [], []

        for t in range(cfg.FENETRE_ROLLING, len(idx), cfg.PAS_ROLLING):
            sous_s1 = s1.iloc[t - cfg.FENETRE_ROLLING:t]
            sous_s2 = s2.iloc[t - cfg.FENETRE_ROLLING:t]
            date = idx[t]

            try:
                _, pv, _ = coint(sous_s1, sous_s2)
                X = sm.add_constant(sous_s2)
                m = sm.OLS(sous_s1, X).fit()
                beta = float(m.params[a2])
            except Exception:
                pv, beta = 1.0, 0.0

            dates.append(date)
            pvals.append(float(pv))
            betas.append(float(beta))

        pvals_np = np.array(pvals)
        score = 100.0 * float((pvals_np < 0.05).sum()) / float(len(pvals_np))

        if score >= cfg.ROBUSTESSE_MIN:
            r = ligne.to_dict()
            r["RobustnessScore"] = round(score, 2)
            valides.append(r)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax1.plot(dates, pvals, label="P-value")
            ax1.axhline(0.05, color="red", linestyle="--", label="Seuil 0.05")
            ax1.set_title(f"Robustesse: {nom} (Score: {score:.1f}%)")
            ax1.grid(alpha=0.3)
            ax1.legend()

            ax2.plot(dates, betas, label="Beta")
            ax2.grid(alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            plt.show()

    if not valides:
        print("[ROBUST] Aucune paire validée.")
        pd.DataFrame().to_csv(cfg.FICHIER_VALIDES, index=False)
        return

    df_final = pd.DataFrame(valides).sort_values(["Rank", "RobustnessScore"], ascending=[True, False])
    df_final.to_csv(cfg.FICHIER_VALIDES, index=False)
    print(f"[ROBUST] Paires validées sauvegardées dans: {cfg.FICHIER_VALIDES}")
    print(df_final[["Pair", "Rank", "RobustnessScore"]].head(10))


if __name__ == "__main__":
    tester_robustesse()
