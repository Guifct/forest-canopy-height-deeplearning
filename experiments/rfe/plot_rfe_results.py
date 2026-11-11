import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurações de estilo para gráficos mais profissionais
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

# Caminhos
RFE_DIR = Path(r"C:\Users\gui\Documents\Tese\trabalho tese\organizado\RFE\RFE_results")
SUMMARY_CSV = RFE_DIR / "rfe_summary.csv"
FINAL_BANDS_CSV = RFE_DIR / "iter_31_bands.csv"

PLOTS_DIR = RFE_DIR / "figuras"
PLOTS_DIR.mkdir(exist_ok=True)

def plot_rfe_metrics():
    print("A gerar gráficos globais (RMSE e R2)...")
    
    # Carregar resumo
    df = pd.read_csv(SUMMARY_CSV)
    
    # Inverter ordem para que o eixo X vá do maior número de bandas para o menor (ordem cronológica)
    # Mas em gráficos RFE é comum ler da esquerda (muitas bandas) para a direita (poucas bandas)
    
    # 1. Gráfico RMSE
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n_bands', y='mean_rmse', marker='o', color='#d62728', linewidth=2)
    
    # Encontrar o ponto de RMSE mínimo
    min_rmse_idx = df['mean_rmse'].idxmin()
    min_rmse_row = df.loc[min_rmse_idx]
    plt.axvline(min_rmse_row['n_bands'], color='gray', linestyle='--', alpha=0.7)
    plt.scatter(min_rmse_row['n_bands'], min_rmse_row['mean_rmse'], color='black', s=100, zorder=5, 
                label=f'Min RMSE: {min_rmse_row["mean_rmse"]:.2f} ({int(min_rmse_row["n_bands"])} bandas)')
    
    plt.title('Evolução do Erro (RMSE) na Remoção Iterativa de Bandas', fontsize=14, pad=15)
    plt.xlabel('Número de Bandas Ativas', fontsize=12)
    plt.ylabel('RMSE Médio (Validação Cruzada)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'rfe_rmse_evolution.png')
    plt.close()
    
    # 2. Gráfico R2
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n_bands', y='mean_r2', marker='s', color='#1f77b4', linewidth=2)
    
    # Encontrar o ponto de R2 máximo
    max_r2_idx = df['mean_r2'].idxmax()
    max_r2_row = df.loc[max_r2_idx]
    plt.axvline(max_r2_row['n_bands'], color='gray', linestyle='--', alpha=0.7)
    plt.scatter(max_r2_row['n_bands'], max_r2_row['mean_r2'], color='black', s=100, zorder=5,
                label=f'Max R²: {max_r2_row["mean_r2"]:.3f} ({int(max_r2_row["n_bands"])} bandas)')
    
    plt.title('Evolução do R² na Remoção Iterativa de Bandas', fontsize=14, pad=15)
    plt.xlabel('Número de Bandas Ativas', fontsize=12)
    plt.ylabel('R² Médio (Validação Cruzada)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'rfe_r2_evolution.png')
    plt.close()

def plot_final_bands():
    print("A gerar gráfico das top bandas finais...")
    
    if not FINAL_BANDS_CSV.exists():
        print(f"Aviso: {FINAL_BANDS_CSV} não encontrado.")
        return
        
    df_bands = pd.read_csv(FINAL_BANDS_CSV)
    
    # Limpar um pouco os nomes removendo o prefixo "Mafra_20200000_" para ficar mais legível na tese
    # Esta função assume que o padrão é [Área]_[Ano]_NomeDaBanda
    def clean_band_name(name):
        parts = str(name).split('_', 2)
        if len(parts) >= 3 and parts[1].isdigit():
            return parts[2]
        return name
        
    df_bands['clean_name'] = df_bands['band_name'].apply(clean_band_name)
    
    # Ordenar por importância (crescente para o gráfico de barras horizontais ficar com a maior no topo)
    df_bands = df_bands.sort_values(by='importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_bands['clean_name'], df_bands['importance'], color='#2ca02c')
    
    plt.title('Importância (L1-Norm) das Últimas 9 Bandas Sobreviventes', fontsize=14, pad=15)
    plt.xlabel('Soma Absoluta dos Pesos (Importância)', fontsize=12)
    plt.ylabel('Variável / Banda', fontsize=12)
    
    # Adicionar o valor no fim de cada barra
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                 ha='left', va='center', fontsize=10)
                 
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'rfe_final_bands_importance.png')
    plt.close()

if __name__ == "__main__":
    print("A iniciar geração de gráficos RFE...")
    plot_rfe_metrics()
    plot_final_bands()
    print(f"Concluído! Os gráficos foram guardados em: {PLOTS_DIR}")
