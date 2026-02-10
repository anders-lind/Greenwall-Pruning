import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
FILENAME = 'motor_test_readings.csv'
FILTER_ALPHA = 0.8  # The alpha value we discussed earlier

def clean_and_parse(val):
    """
    Removes brackets '[' and ']' from the data string and converts to float.
    Example: '[2770]' -> 2770.0
    """
    if isinstance(val, str):
        return float(val.replace('[', '').replace(']', ''))
    return float(val)

def apply_ema_filter(data, alpha):
    """
    Applies an Exponential Moving Average filter.
    y[i] = alpha * x[i] + (1-alpha) * y[i-1]
    """
    # Initialize the filtered array with the first value
    filtered = [data[0]] 
    for i in range(1, len(data)):
        prev_val = filtered[-1]
        curr_val = data[i]
        
        # The EMA formula
        new_val = (alpha * curr_val) + ((1.0 - alpha) * prev_val)
        filtered.append(new_val)
    return filtered

def plot_metric(timestamps, raw_data, filtered_data, title, y_label, color):
    """Creates a new window and plots raw vs filtered data."""
    plt.figure(figsize=(10, 6))  # Create a new window/figure
    
    # Plot Raw Data (faint, dashed)
    plt.plot(timestamps, raw_data, label='Raw Signal', color='lightgray', linestyle='--', linewidth=1.5)
    
    # Plot Filtered Data (solid, colored)
    plt.plot(timestamps, filtered_data, label=f'Filtered (alpha={FILTER_ALPHA})', color=color, linewidth=2)
    
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

def main():
    # 1. Load the Data
    # We define converters to handle the brackets '[]' automatically during load
    converters = {
        'position': clean_and_parse,
        'current': clean_and_parse,
        'velocity': clean_and_parse
    }
    
    # Note: skipinitialspace=True handles spaces after commas if present
    df = pd.read_csv(FILENAME, converters=converters, skipinitialspace=True)

    # 2. Extract columns
    t = df['timestamp'].values
    pos = df['position'].values
    curr = df['current'].values
    vel = df['velocity'].values

    # 3. Apply Filters
    pos_filtered = apply_ema_filter(pos, FILTER_ALPHA)
    curr_filtered = apply_ema_filter(curr, FILTER_ALPHA)
    vel_filtered = apply_ema_filter(vel, FILTER_ALPHA)

    # 4. Plot in Separate Windows
    # Window 1: Position
    plot_metric(t, pos, pos_filtered, 'Motor Position', 'Position (Encoder Units)', 'blue')

    # Window 2: Current
    plot_metric(t, curr, curr_filtered, 'Motor Current', 'Current (mA)', 'red')

    # Window 3: Velocity
    plot_metric(t, vel, vel_filtered, 'Motor Velocity', 'Velocity (RPM/Units)', 'green')

    # Show all windows
    plt.show()

if __name__ == "__main__":
    main()