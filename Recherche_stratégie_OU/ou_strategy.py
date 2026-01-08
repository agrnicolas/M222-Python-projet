# ou_strategy.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
import config as cfg


def estimer_demi_vie_ar1(serie: pd.Series) -> float:
    s1 = serie.shift(1).dropna()
    s0 = serie.loc[s1.index]

    if len(s0) < 30:
        return np.nan

    X = sm.add_constant(s1.values)
    m = sm.OLS(s0.values, X).fit()
    b = float(m.params[1])

    if b <= 0 or b >= 1:
        return np.nan

    # Si b est trop proche de 1, la demi-vie explose et ce n'est pas tradable.
    # On rejette ces cas pour éviter des valeurs absurdes.
    if b >= cfg.AR1_B_MAX:
        return np.nan

    demi_vie = -np.log(2) / np.log(b)
    return float(demi_vie)


def construire_features(
    df_log: pd.DataFrame,
    actif1: str,
    actif2: str,
    fenetre_beta: int,
    fenetre_z: int,
    fenetre_ou: int,
    demi_vie_max: float,
    z_max_abs: float,
    z_entree: float,
    z_sortie: float,
    z_stop: float,
    max_hold: int,
) -> pd.DataFrame:

    df = df_log[[actif1, actif2]].dropna().copy()
    df = df.rename(columns={actif1: "y", actif2: "x"})

    y = df["y"]
    x = df["x"]

    # Beta rolling simple (OLS)
    betas = []
    for i in range(len(df)):
        if i < fenetre_beta:
            betas.append(np.nan)
            continue

        sous_y = y.iloc[i - fenetre_beta:i]
        sous_x = x.iloc[i - fenetre_beta:i]

        X = sm.add_constant(sous_x.values)
        m = sm.OLS(sous_y.values, X).fit()
        betas.append(float(m.params[1]))

    df["beta"] = betas
    df = df.dropna()

    # Spread
    df["spread"] = df["y"] - df["beta"] * df["x"]

    # Z-score standard sur le spread
    df["spread_moy"] = df["spread"].rolling(fenetre_z).mean()
    df["spread_std"] = df["spread"].rolling(fenetre_z).std()
    df["z"] = (df["spread"] - df["spread_moy"]) / df["spread_std"].replace(0, np.nan)

    # Demi-vie AR(1) rolling sur le spread
    demi_vies = []
    spread = df["spread"]

    for i in range(len(df)):
        if i < fenetre_ou:
            demi_vies.append(np.nan)
            continue
        sous = spread.iloc[i - fenetre_ou:i].dropna()
        hl = estimer_demi_vie_ar1(sous)
        demi_vies.append(hl)

    df["demi_vie"] = demi_vies

    # Régime ok
    df["regime_ok"] = (
        df["z"].notna()
        & df["demi_vie"].notna()
        & (df["demi_vie"] > 0)
        & (df["demi_vie"] <= demi_vie_max)
        & (df["z"].abs() <= z_max_abs)
        & df["spread_std"].notna()
        & (df["spread_std"] > 0)
    )

    # Règles de trading
    position = np.zeros(len(df))
    duree = 0

    for i in range(1, len(df)):
        ok = bool(df["regime_ok"].iloc[i])
        z = df["z"].iloc[i]
        z = float(z) if np.isfinite(z) else 0.0
        pos_prec = float(position[i - 1])

        # Si pas de régime, on est flat
        if not ok:
            position[i] = 0.0
            duree = 0
            continue

        # Stop
        if pos_prec != 0.0 and abs(z) >= z_stop:
            position[i] = 0.0
            duree = 0
            continue

        # Time stop
        if pos_prec != 0.0:
            duree += 1
            if duree >= max_hold:
                position[i] = 0.0
                duree = 0
                continue

        # Sortie
        if pos_prec > 0.0 and z >= -z_sortie:
            position[i] = 0.0
            duree = 0
            continue
        if pos_prec < 0.0 and z <= z_sortie:
            position[i] = 0.0
            duree = 0
            continue

        # Entrée
        if pos_prec == 0.0:
            if z <= -z_entree:
                position[i] = 1.0
                duree = 0
                continue
            if z >= z_entree:
                position[i] = -1.0
                duree = 0
                continue

        position[i] = pos_prec

    df["position"] = position

    # Exposition simple
    df["gross"] = 1.0 + df["beta"].abs()
    df["scale"] = 1.0 / df["gross"].replace(0, np.nan)

    return df.dropna()
