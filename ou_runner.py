# ou_runner.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config as cfg
from ou_strategy import construire_features


def resampler_log_prix(df_log: pd.DataFrame, tf: str) -> pd.DataFrame:
    return df_log.sort_index().resample(tf.upper()).last().ffill(limit=2).dropna()


def backtest(feat: pd.DataFrame, frais_bps: float) -> pd.DataFrame:
    bt = feat.copy()
    bt["dy"] = bt["y"].diff()
    bt["dx"] = bt["x"].diff()
    bt["pos_lag"] = bt["position"].shift(1).fillna(0.0)

    bt["pnl_brut"] = bt["pos_lag"] * bt["scale"] * (bt["dy"] - bt["beta"] * bt["dx"])

    taux = frais_bps / 10000.0
    turnover = (bt["position"].diff().abs().fillna(0.0) > 0).astype(int)

    mult = 2.0 if cfg.FRAIS_DEUX_JAMBES else 1.0
    bt["frais"] = turnover * taux * mult * bt["gross"]

    bt["pnl_net"] = bt["pnl_brut"] - bt["frais"]
    bt["equity"] = np.exp(bt["pnl_net"].fillna(0.0).cumsum())

    return bt.dropna(subset=["dy", "dx"])


def rapport(bt: pd.DataFrame, periodes_par_an: int = 365 * 24):
    pnl = bt["pnl_net"].fillna(0.0)
    total = float(bt["equity"].iloc[-1] - 1.0)
    vol = float(pnl.std())
    moy = float(pnl.mean())
    sharpe = float(moy / vol * np.sqrt(periodes_par_an)) if vol > 0 else np.nan

    dd = (bt["equity"] / bt["equity"].cummax()) - 1.0
    maxdd = float(dd.min())

    trades = float(bt["position"].diff().abs().sum() / 2.0)
    hit = float((pnl[pnl != 0] > 0).mean()) if (pnl != 0).any() else np.nan

    return total, sharpe, maxdd, trades, hit


def plot_z(feat: pd.DataFrame, pair_name: str):
    t = feat.index
    z = feat["z"]

    plt.figure(figsize=(14, 5))
    plt.plot(t, z, label="Z-score")
    plt.axhline(0, linewidth=1)

    plt.axhline(cfg.Z_ENTREE, linestyle="--", label="Entree")
    plt.axhline(-cfg.Z_ENTREE, linestyle="--")

    plt.axhline(cfg.Z_SORTIE, linestyle=":", label="Sortie")
    plt.axhline(-cfg.Z_SORTIE, linestyle=":")

    plt.axhline(cfg.Z_STOP, linestyle="-.", label="Stop")
    plt.axhline(-cfg.Z_STOP, linestyle="-.")

    plt.title(f"{pair_name} | Z-score ({cfg.TF_TRADING})")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def tester_plusieurs_paires(df_log: pd.DataFrame, df_valides: pd.DataFrame, nb_paires: int = 10) -> pd.DataFrame:
    resultats = []

    df_tri = df_valides.copy()
    if "RobustnessScore" in df_tri.columns:
        df_tri = df_tri.sort_values(["Rank", "RobustnessScore"], ascending=[True, False])
    else:
        df_tri = df_tri.sort_values(["Rank"], ascending=True)

    df_tri = df_tri.head(nb_paires)

    for _, ligne in df_tri.iterrows():
        a1, a2 = ligne["Asset1"], ligne["Asset2"]
        pair_name = ligne["Pair"]

        sub = df_log[[a1, a2]].dropna()
        df_tf = resampler_log_prix(sub, cfg.TF_TRADING)

        min_obs = max(cfg.FENETRE_BETA, cfg.FENETRE_Z, cfg.FENETRE_OU) + 50
        if len(df_tf) < min_obs:
            continue

        feat = construire_features(
            df_tf,
            a1,
            a2,
            fenetre_beta=cfg.FENETRE_BETA,
            fenetre_z=cfg.FENETRE_Z,
            fenetre_ou=cfg.FENETRE_OU,
            demi_vie_max=cfg.DEMI_VIE_MAX,
            z_max_abs=cfg.Z_MAX_ABS,
            z_entree=cfg.Z_ENTREE,
            z_sortie=cfg.Z_SORTIE,
            z_stop=cfg.Z_STOP,
            max_hold=cfg.MAX_HOLD,
        )

        if feat.empty:
            continue

        bt = backtest(feat, frais_bps=cfg.FRAIS_BPS)
        total, sharpe, maxdd, trades, hit = rapport(bt, periodes_par_an=365 * 24)

        resultats.append({
            "Pair": pair_name,
            "TotalReturn": total,
            "Sharpe": sharpe,
            "MaxDD": abs(maxdd),
            "Trades": trades,
            "HitRate": hit,
        })

    if not resultats:
        return pd.DataFrame()

    return pd.DataFrame(resultats).sort_values(
        by=["TotalReturn", "Sharpe"],
        ascending=[False, False]
    )


def main():
    df_log = pd.read_csv(cfg.FICHIER_LOG_PRIX, index_col=0, parse_dates=True).sort_index()
    valides = pd.read_csv(cfg.FICHIER_VALIDES)

    if valides.empty:
        print("[OU] validated_pairs.csv vide.")
        return

    tableau = tester_plusieurs_paires(df_log, valides, nb_paires=10)

    if tableau.empty:
        print("[OU] Aucun backtest exploitable sur les paires testées.")
        return

    print("\n=== CLASSEMENT (top) ===")
    print(tableau.head(10).to_string(index=False))

    meilleure = tableau.iloc[0]
    pair_name = str(meilleure["Pair"])

    ligne = valides[valides["Pair"] == pair_name].iloc[0]
    a1, a2 = ligne["Asset1"], ligne["Asset2"]

    print(f"\n[OU] Paire retenue (meilleur backtest): {pair_name}")

    sub = df_log[[a1, a2]].dropna()
    df_tf = resampler_log_prix(sub, cfg.TF_TRADING)

    feat = construire_features(
        df_tf,
        a1,
        a2,
        fenetre_beta=cfg.FENETRE_BETA,
        fenetre_z=cfg.FENETRE_Z,
        fenetre_ou=cfg.FENETRE_OU,
        demi_vie_max=cfg.DEMI_VIE_MAX,
        z_max_abs=cfg.Z_MAX_ABS,
        z_entree=cfg.Z_ENTREE,
        z_sortie=cfg.Z_SORTIE,
        z_stop=cfg.Z_STOP,
        max_hold=cfg.MAX_HOLD,
    )

    bt = backtest(feat, frais_bps=cfg.FRAIS_BPS)
    total, sharpe, maxdd, trades, hit = rapport(bt, periodes_par_an=365 * 24)

    print("\n=== REPORT (meilleure paire) ===")
    print("TotalReturn:", total)
    print("SharpeAnn:", sharpe)
    print("MaxDD:", abs(maxdd))
    print("Trades:", trades)
    print("HitRate:", hit)

    plot_z(feat, pair_name)

    plt.figure(figsize=(12, 4))
    plt.plot(bt.index, bt["equity"])
    plt.title(f"{pair_name} | Equity ({cfg.TF_TRADING})")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.plot(feat.index, feat["spread_std"])
    plt.title(f"{pair_name} | Std rolling spread")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.plot(feat.index, feat["demi_vie"])
    plt.title(f"{pair_name} | Demi-vie (AR1 sur spread)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
