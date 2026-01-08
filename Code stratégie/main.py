# main.py
import sys
import os
import config as cfg

# Import des modules du projet
import data_processor
import pair_selector
import robustness
import backtester
import live_simulator

def menu():
    while True:
        print("\n" + "="*60)
        print("CRYPTO PAIR TRADING - Processus")
        print("="*60)
        print("0.LANCER TOUT LE PROCESSUS")
        print("-" * 60)
        print("1. [Données]   Télécharger (Tout l'historique)")
        print(f"2. [Screening] Sélection & Robustesse (Arrêt: {cfg.TRAIN_END_DATE})")
        print("3. [Backtest]  Optimisation IA (Training Set)")
        print("4. [Live]      Simulation Hors-Échantillon (Test Set)")
        print("5. Quitter")
        print("-" * 60)
        
        choice = input("Votre choix : ").strip()

        # OPTION 0 : TOUT LANCER
        if choice == '0':
            print("\DÉMARRAGE DU CODE COMPLET...")
            
            # 1. Données
            if data_processor.download_and_clean():
                
                # 2. Screening
                print("\n--- ÉTAPE 2 : SÉLECTION (PASSÉ) ---")
                if pair_selector.find_best_pairs():
                    print("\n--- ÉTAPE 2b : ROBUSTESSE (PASSÉ) ---")
                    robustness.run_robustness_test()
                    
                    # 3. Backtest
                    if os.path.exists(cfg.FINAL_FILE):
                        print("\n--- ÉTAPE 3 : OPTIMISATION (TRAINING) ---")
                        backtester.run_final_validation()
                        
                        # 4. Live
                        if os.path.exists("live_strategies.csv"):
                            print("\n--- ÉTAPE 4 : SIMULATION (TEST/FUTUR) ---")
                            live_simulator.run_simulation()
                        else:
                            print(" Arrêt : Aucune stratégie validée à l'étape 3.")
                    else:
                        print("Arrêt : Aucune paire robuste trouvée à l'étape 2.")
            else:
                print("Erreur lors du téléchargement des données.")

        # OPTION 1 : DONNÉES
        elif choice == '1':
            data_processor.download_and_clean()
            
        # OPTION 2 : SCREENING
        elif choice == '2':
            print("\n--- ÉTAPE 1 : SÉLECTION ---")
            if pair_selector.find_best_pairs():
                print("\n--- ÉTAPE 2 : ROBUSTESSE ---")
                robustness.run_robustness_test()

        #  OPTION 3 : SEUIL/BACKTEST 
        elif choice == '3':
            if os.path.exists(cfg.FINAL_FILE):
                backtester.run_final_validation()
            else:
                print(f"Erreur : Fichier '{cfg.FINAL_FILE}' manquant. Lancez l'étape 2 d'abord.")

        # OPTION 4 : LIVE STRATEGIE
        elif choice == '4':
            if os.path.exists("live_strategies.csv"):
                live_simulator.run_simulation()
            else:
                print("Erreur : Aucune stratégie sauvegardée. Lancez l'étape 3 d'abord.")

        # QUITTER 
        elif choice == '5':
            print("Arrêt du programme.")
            sys.exit()
            
        else:
            print("Choix invalide.")
        
        input("\nAppuyez sur Entrée pour revenir au menu...")

if __name__ == "__main__":
    if not os.path.exists('config.py'):
        print("Erreur : Le fichier config.py est introuvable.")
        sys.exit(1)
    menu()