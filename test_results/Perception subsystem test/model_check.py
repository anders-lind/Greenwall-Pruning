import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# --- 1. LOAD DATA ---
df_mahal = pd.read_csv('stats_mahalanobis.csv')
df_sam2 = pd.read_csv('stats_sam2.csv')
df_hybrid = pd.read_csv('stats_hybrid.csv')

# --- 2. FILTER DATASETS ---

# A. CRITERIA: Strictly isolate shared True Positives (GT present AND all models detected it)
tp_mahal_files = set(df_mahal[(df_mahal['gt_present'] == 1) & (df_mahal['iou'] > 0)]['filename'])
tp_sam2_files = set(df_sam2[(df_sam2['gt_present'] == 1) & (df_sam2['iou'] > 0)]['filename'])
tp_hybrid_files = set(df_hybrid[(df_hybrid['gt_present'] == 1) & (df_hybrid['iou'] > 0)]['filename'])

# Find the intersection where ALL three conditions are satisfied
shared_files = list(tp_mahal_files & tp_sam2_files & tp_hybrid_files)

iou_mahal = df_mahal[df_mahal['filename'].isin(shared_files)]['iou']
iou_sam2 = df_sam2[df_sam2['filename'].isin(shared_files)]['iou']
iou_hybrid = df_hybrid[df_hybrid['filename'].isin(shared_files)]['iou']

# B. CRITERIA: Inference Time filters
# 1. Mahalanobis: Filter away anything ABOVE 15 ms
time_mahal = df_mahal[df_mahal['time_ms'] <= 15]['time_ms']
# 2. SAM2: Keep standard
time_sam2 = df_sam2['time_ms']
# 3. Hybrid: Filter away anything BELOW 100 ms
time_hybrid = df_hybrid[df_hybrid['time_ms'] >= 100]['time_ms']

print(f"Verified Shared True Positive image count for IoU: {len(shared_files)}")
print(f"Mahalanobis time points remaining (<= 15ms): {len(time_mahal)}")
print(f"Hybrid time points remaining (>= 100ms):      {len(time_hybrid)}\n")


def generate_normality_grid(datasets, model_names, metric_label):
    """
    Creates a wide 2-row, 3-column grid layout.
    Row 0: Histograms (all models side-by-side)
    Row 1: Q-Q Plots (all models side-by-side)
    """
    print(f"========================================\n STATISTICAL TESTS FOR {metric_label.upper()}\n========================================")
    
    # Configure large fonts globally for the wide layout
    plt.rcParams.update({
        'axes.titlesize': 14,    
        'axes.labelsize': 12,    
        'xtick.labelsize': 11,   
        'ytick.labelsize': 11,   
        'figure.titlesize': 18   
    })
    
    # Changed grid structure to 2 rows (Hist vs QQ) and 3 columns (Models)
    # Made it wide (18) and less tall (10)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for i, (data, name) in enumerate(zip(datasets, model_names)):
        if len(data) < 3:
            print(f"[{name}] Warning: Not enough data points to run normality check!")
            continue
            
        # Run statistical calculations
        _, shapiro_p = stats.shapiro(data)
        _, k2_p = stats.normaltest(data)
        
        print(f"[{name}] Shapiro-Wilk p-value: {shapiro_p:.4e}")
        print(f"[{name}] D'Agostino K^2 p-value: {k2_p:.4e}\n")
        
        # Target specific coordinates in the 2x3 matrix
        ax_hist = axes[0, i]  # Top Row, Column i
        ax_qq = axes[1, i]   # Bottom Row, Column i
        
        # 1. Plot Histogram + KDE (Top Row)
        sns.histplot(data, kde=True, ax=ax_hist, color='teal', bins=20)
        ax_hist.set_title(f'{name}\nDistribution', pad=12)
        ax_hist.set_xlabel(f'{metric_label}', labelpad=8)
        ax_hist.set_ylabel('Count', labelpad=8)
        
        # 2. Plot Q-Q Plot (Bottom Row)
        stats.probplot(data, dist="norm", plot=ax_qq)
        ax_qq.set_title(f'{name}\nQ-Q Plot', pad=12)
        ax_qq.set_xlabel('Theoretical Quantiles', labelpad=8)
        ax_qq.set_ylabel('Ordered Values', labelpad=8)
        
        # Clean up text objects generated inside stats.probplot that cause crowding
        if ax_qq.texts:
            for text in ax_qq.texts:
                text.set_visible(False)
                
        # Tweak line color of Q-Q plot to match your layout style
        ax_qq.get_lines()[1].set_color('crimson')
        ax_qq.get_lines()[1].set_linewidth(2)

    # plt.suptitle(f'Normality Analysis Group Levels: {metric_label}', y=0.97, weight='bold')
    
    # Adjusted spacing parameters to suit the landscape alignment
    plt.subplots_adjust(left=0.06, right=0.96, bottom=0.08, top=0.88, hspace=0.35, wspace=0.25)
    
    # Save files
    file_tag = metric_label.lower().replace(' ', '_').replace('(ms)', '').replace('iou', '').replace('__', '_').strip('_')
    plt.savefig(f"figures/normality_{file_tag}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"figures/normality_{file_tag}.pdf", bbox_inches='tight')
    plt.show()

# --- 3. RUN GENERATION FOR BOTH METRICS ---
# Generate 2x3 grid layout for Inference Time
generate_normality_grid(
    datasets=[time_mahal, time_sam2, time_hybrid],
    model_names=['Mahalanobis', 'SAM2', 'Hybrid'],
    metric_label='Inference Time (ms)'
)

# Generate 2x3 grid layout for IoU
generate_normality_grid(
    datasets=[iou_mahal, iou_sam2, iou_hybrid],
    model_names=['Mahalanobis', 'SAM2', 'Hybrid'],
    metric_label='IoU'
)