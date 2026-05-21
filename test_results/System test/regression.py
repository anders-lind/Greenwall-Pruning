import pandas as pd
import numpy as np
import statsmodels.api as sm

# 1. Load your actual data file
# Replace 'your_robot_data.csv' with the actual path to your file (e.g., 'C:/Users/Name/Documents/data.csv')
file_path = '/home/anders/workspace/masters_thesis/Greenwall-Pruning/test_results/System test/combined_results/experiment_results.csv'

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {len(df)} rows of data.\n")
except FileNotFoundError:
    print(f"Error: Could not find the file at '{file_path}'. Please check the file name and path.")
    exit()

# 2. Define your independent variables (the factors)
factors = [
    'dx', 'dy', 'dz', 
    'robot_x', 'robot_y', 'robot_yaw', 
    'u_seed', 'v_seed', 
    'verif_percentile', 'verif_dist', 
    'user_accepted'
]

# 3. Separate your factors (X) and your success outcome (y)
X = df[factors]
y = df['pruning_success']

# Clean up any rows with missing values (NaN) in these specific columns so the math doesn't break
clean_indices = X.dropna().index
X = X.loc[clean_indices]
y = y.loc[clean_indices]

# Add a constant (intercept) to the model - required by statsmodels
X = sm.add_constant(X)

print("--- Running Multiple Logistic Regression ---")
try:
    # 4. Fit the logistic regression model
    model = sm.Logit(y, X).fit()
    
    # 5. Print the full detailed statistical summary
    print(model.summary())
    
    # 6. Extract Odds Ratios and p-values into a clean, readable table
    results_table = pd.DataFrame({
        'Odds Ratio': np.exp(model.params),
        'p-value': model.pvalues
    })
    
    print("\n" + "="*50)
    print("ALL FACTORS & ODDS RATIOS")
    print("="*50)
    print(results_table)
    
    # 7. Isolate the TRUE drivers (where p-value < 0.05)
    # We exclude the 'const' row because it's just the background intercept math
    significant_factors = results_table[(results_table['p-value'] < 0.05) & (results_table.index != 'const')]
    
    print("\n" + "="*50)
    print("SIGNIFICANT CONTRIBUTORS (p < 0.05)")
    print("="*50)
    if significant_factors.empty:
        print("No individual factors met the strict p < 0.05 threshold.")
        print("Tip: If your sample size is small, look for factors with p < 0.10 for trends.")
    else:
        print(significant_factors)
        print("\nInterpretation Note:")
        print("- Odds Ratio > 1: Increasing this factor INCREASES the odds of pruning success.")
        print("- Odds Ratio < 1: Increasing this factor DECREASES the odds of pruning success.")

except Exception as e:
    print(f"\nAn error occurred during modeling: {e}")
    print("\nPossible solutions:")
    print("1. Ensure your 'pruning_success' column only contains 0s and 1s.")
    print("2. Ensure you have a mix of both successes (1) and failures (0) in your file.")
    print("3. Check for 'Perfect Separation' (e.g., if 'user_accepted' is always 1 when success is 1, drop it from the factors list).")