# main.py
import os
import config as cfg
import data_processor
import pair_selector
import robustness
import ou_runner


def fichier_ok(chemin: str) -> bool:
    return os.path.exists(chemin) and os.path.getsize(chemin) > 0


def status():
    print("\n=== STATUS ===")
    print(f"LOG_PRIX   : {cfg.FICHIER_LOG_PRIX} -> {'OK' if fichier_ok(cfg.FICHIER_LOG_PRIX) else 'MANQUANT'}")
    print(f"CANDIDATS  : {cfg.FICHIER_CANDIDATS} -> {'OK' if fichier_ok(cfg.FICHIER_CANDIDATS) else 'MANQUANT'}")
    print(f"VALIDES    : {cfg.FICHIER_VALIDES} -> {'OK' if fichier_ok(cfg.FICHIER_VALIDES) else 'MANQUANT'}")
    print("==============\n")


def pipeline_force():
    data_processor.telecharger_et_nettoyer()
    ok = pair_selector.trouver_paires()
    if ok:
        robustness.tester_robustesse()


def main():
    print("Pipeline Pair Trading (DATA -> PAIRS -> ROBUST -> OU)")
    status()

    while True:
        print("1) DATA")
        print("2) PAIRS")
        print("3) ROBUST")
        print("4) PIPELINE (force)")
        print("5) OU RUN")
        print("0) Quitter")

        choix = input("Choix: ").strip()

        if choix == "0":
            break
        if choix == "1":
            data_processor.telecharger_et_nettoyer()
            status()
        elif choix == "2":
            pair_selector.trouver_paires()
            status()
        elif choix == "3":
            robustness.tester_robustesse()
            status()
        elif choix == "4":
            pipeline_force()
            status()
        elif choix == "5":
            ou_runner.main()
        else:
            print("Choix invalide.")


if __name__ == "__main__":
    main()
