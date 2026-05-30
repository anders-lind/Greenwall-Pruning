import os
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
CSV_PATH = "combined_results/experiment_results.csv"  # Replace with your actual file path
OUTPUT_PNG = "fail_case_pareto.png"
OUTPUT_PDF = "fail_case_pareto.pdf"

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find CSV file at '{CSV_PATH}'")
        return

    # 1. Load data and extract the fail_case column
    df = pd.read_csv(CSV_PATH)
    fail_series = df['fail_case'].dropna().str.strip()
    
    if fail_series.empty:
        print("Warning: No failure cases detected in the CSV file.")
        return

    # 2. Compute absolute and cumulative frequencies
    counts = fail_series.value_counts().to_frame(name='count')
    total_failures = counts['count'].sum()
    counts['cum_percentage'] = (counts['count'].cumsum() / total_failures) * 100

    # 3. Setup the plot with standard figure dimensions
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # --- Primary Axis: Bar Chart (Counts) ---
    # zorder=3 places the bars in front of the grid lines
    ax1.bar(counts.index, counts['count'], width=0.6, zorder=3)
    ax1.set_xlabel('Failure Category')
    ax1.set_ylabel('Count')
    
    # Configure horizontal-only grid lines tucked behind the bars (zorder=0)
    ax1.grid(True, axis='y', zorder=0)
    
    # Standard rotation for clear x-axis text layout
    plt.xticks(rotation=45, ha='right')

    # --- Secondary Axis: Line Chart (Cumulative Percentage) ---
    ax2 = ax1.twinx()
    # zorder=4 ensures the line chart stays on the absolute top layer
    ax2.plot(counts.index, counts['cum_percentage'], color='C1', marker='o', zorder=4)
    ax2.set_ylabel('Cumulative Percentage (%)')
    
    # Enforce a predictable y-axis upper ceiling limit for percentages
    ax2.set_ylim(0, 110)

    # 4. Save both formats with standard tight margins
    plt.title('Pareto Analysis of Failure Cases')
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PDF, bbox_inches='tight')
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    
    print(f"Saved default style Pareto plots:\n - {OUTPUT_PNG}\n - {OUTPUT_PDF}")
    plt.show()

if __name__ == '__main__':
    main()