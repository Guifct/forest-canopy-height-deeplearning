import os
import sys
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONDA_PYTHON = r"C:\Users\gui\Documents\Tese\.conda\python.exe"

SCRIPTS = [
    ("Figura 6.1", "fig_6_1_model_lineage_evolution.py", "model_lineage_evolution_bar_chart.png", False),
    ("Figura 6.2", "fig_6_2_dummy_regressor.py", "dummy_regressor_visualisation.png", False),
    ("Figura 6.3", "fig_6_3_minimalist_cnn.py", "minimalist_cnn_regression_to_mean.png", False),
    ("Figura 6.4", "fig_6_4_spatialcnn_residuals.py", "spatialcnn_error_residuals.png", False),
    ("Figura 6.5", "fig_6_5_rfe_trajectory_and_sensor_branches.py", "rfe_pruning_and_sensor_branches.png", False),
    ("Figura 6.6", "fig_6_6_ensemble_scatter_density.py", "ensemble_final_ca_scatter.png", False),
    ("Figura 6.7", "fig_6_7_height_distribution.py", "height_distribution_regression_to_mean.png", False),
    ("Figura 6.8", "fig_6_8_boxplot_residuals_classes.py", "boxplot_residuos_por_classe_altura_aug_on.png", False),
    ("Figura 6.9", "fig_6_9_weighted_loss_mitigation.py", "weighted_loss_rmse_reduction.png", False),
    ("Figura 6.10", "fig_6_10_patch_size_ablation.py", "patch_size_ablation_comparison.png", False),
    ("Figura 6.11", "fig_6_11_learning_curves_augmentation.py", "learning_curves_aug_off_vs_on.png", False),
    ("Figura 6.12", "fig_6_12_spatial_prediction_regional.py", "spatial_prediction_regional_validation.png", False),
    ("Figura 6.13", "fig_6_13_mapa_2d_serra_lousa.py", "SerraLousa_Cortado_2D_Canopy_Height_Map.png", True),
    ("Figura 6.14", "fig_6_14_loro_vs_stratified.py", "loro_vs_stratified_comparison.png", False),
]

def main():
    print("=" * 80)
    print("  EXECUTOR MESTRE: GERAR TODOS OS GRAFICOS DO CAPITULO 6 DA TESE")
    print("=" * 80)
    
    results = []
    total_start = time.time()
    
    for fig_id, script_name, out_name, needs_rasterio in SCRIPTS:
        script_path = os.path.join(BASE_DIR, script_name)
        
        # Escolha do executavel Python
        py_exec = sys.executable
        if needs_rasterio:
            try:
                import rasterio
            except ImportError:
                if os.path.exists(CONDA_PYTHON):
                    py_exec = CONDA_PYTHON
                else:
                    print(f"[AVISO] rasterio nao encontrado e {CONDA_PYTHON} nao existe.")

        print(f"\n>> Executando [{fig_id}]: {script_name}...")
        t0 = time.time()
        ret = subprocess.run([py_exec, script_path], cwd=BASE_DIR, capture_output=True, text=True)
        elapsed = time.time() - t0
        
        if ret.returncode == 0:
            status = "SUCESSO"
            print(f"   [OK] Concluido em {elapsed:.2f}s")
        else:
            status = "ERRO"
            print(f"   [ERRO] Falha com codigo {ret.returncode} em {elapsed:.2f}s")
            print("   Stderr:", ret.stderr.strip()[:300])

        results.append((fig_id, script_name, out_name, status, elapsed))

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print("  RESUMO DE EXECUCAO DE TODOS OS GRAFICOS")
    print("=" * 80)
    print(f"{'Figura':<12} | {'Script':<42} | {'Status':<8} | {'Tempo'}")
    print("-" * 80)
    for fig_id, script_name, out_name, status, elapsed in results:
        print(f"{fig_id:<12} | {script_name:<42} | {status:<8} | {elapsed:.2f}s")
    print("-" * 80)
    print(f"Tempo total de execucao: {total_time:.2f}s")
    print(f"Ficheiros guardados em: {os.path.join(BASE_DIR, 'outputs')}")
    print(f"Ficheiros atualizados em: c:\\Users\\gui\\Documents\\Tese\\Escrita_tese_Rascunho\\5-Figures\\Results")
    print("=" * 80)

if __name__ == '__main__':
    main()
