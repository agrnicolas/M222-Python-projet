# diagnostic_pvalues.py
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint
import config as cfg

df = pd.read_csv(cfg.RAW_FILE, index_col=0, parse_dates=True).sort_index()
assets = df.columns.tolist()

results = []

for a1, a2 in combinations(assets, 2):
    y = df[a1]
    x = df[a2]
    try:
        _, pval, _ = coint(y, x)
        if np.isfinite(pval):
            results.append((pval, a1, a2))
    except:
        pass

results.sort(key=lambda x: x[0])

print("\n===== DIAGNOSTIC COINTÉGRATION =====")
print("Nombre de paires testées :", len(results))
print("Min p-value :", results[0][0])

print("\nTop 10 paires (p-values les plus faibles) :")
for pval, a1, a2 in results[:10]:
    print(f"{a1}/{a2}  | p-value = {pval:.4f}")
