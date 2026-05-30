import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.path import Path
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import BoundaryNorm

# --- CONFIGURATION ---
CONFIGS = {
    'Kinematic': {
        'dir': 'data_trial2',
        'scale': 0.06,
        'cutoffs': {
            'bottom_left_trajectory': 62.8, 'bottom_right_trajectory': 57.8,
            'bottom_trajectory': 52.0, 'left_trajectory': 62.0,
            'right_trajectory': 53.0, 'top_left_trajectory': 64.5,
            'top_right_trajectory': 65.0, 'top_trajectory': 64.5
        },
        'vectors': {
            'top_right_trajectory': np.array([0.454, 0.454])+np.array([0.04, 0.11]),
            'top_left_trajectory': np.array([-0.454, 0.454])+np.array([-0.07, 0.10]),
            'bottom_right_trajectory': np.array([0.454, -0.490])+np.array([0.02, -0.03]),
            'bottom_left_trajectory': np.array([-0.454, -0.490])+np.array([-0.04, -0.075]),
            'top_trajectory': np.array([0.0, 0.454])+np.array([0.00, 0.13]),
            'bottom_trajectory': np.array([0.0, -0.490])+np.array([0.003, -0.09]),
            'right_trajectory': np.array([0.454, 0.0])+np.array([0.055, 0.00]),
            'left_trajectory': np.array([-0.454, 0.0])+np.array([-0.07, -0.015])
        }
    },
    'Force': {
        'dir': 'data_force',
        'scale': 0.0012,
        'cutoffs': {
            'bottom_left_trajectory': 21.0, 'bottom_right_trajectory': 22.0,
            'bottom_trajectory': 25.0, 'left_trajectory': 21.0,
            'right_trajectory': 12.8, 'top_left_trajectory': 26.0,
            'top_right_trajectory': 65.0, 'top_trajectory': 7.5
        },
        'vectors': {
            'top_right_trajectory': np.array([0.454, 0.454])+np.array([-0.04, -0.14]),
            'top_left_trajectory': np.array([-0.454, 0.454])+np.array([0.27, -0.28]),
            'bottom_right_trajectory': np.array([0.454, -0.490])+np.array([-0.28, 0.35]),
            'bottom_left_trajectory': np.array([-0.454, -0.490])+np.array([0.26, 0.27]),
            'top_trajectory': np.array([0.0, 0.454]) + np.array([0.00, -0.25]),
            'bottom_trajectory': np.array([0.0, -0.490]) + np.array([0.00, 0.3]),
            'right_trajectory': np.array([0.454, 0.0]) + np.array([-0.31, 0.00]),
            'left_trajectory': np.array([-0.454, 0.0]) + np.array([0.01, 0.00])
        }
    }
}

# --- VISUALIZATION CONTROLS ---
COLORMAP_NAME = 'YlOrRd'      
ERROR_THRESHOLD_MM = 10.0 

# Reachable Workspace Boundary (Outer Hull)
HULL_COLOR = 'black'
HULL_STYLE = '-'             
HULL_WIDTH = 1

# Usable Workspace Boundary (< 10mm threshold contour)
ACC_COLOR = 'navy'            
ACC_STYLE = '--'             
ACC_WIDTH = 2


def load_and_trim(filepath, scale_factor, cutoff_time):
    df = pd.read_csv(filepath, sep=';', decimal=',', skiprows=2, names=['t', 'x', 'y', 'empty']).dropna(axis=1, how='all')
    df = df.dropna(subset=['t', 'x', 'y'])
    df['x'] = df['x'] * scale_factor
    df['y'] = df['y'] * scale_factor
    df['t'] = df['t'] - df['t'].iloc[0]
    return df[df['t'] <= cutoff_time].copy()    

def calc_cross_track_error(df, traj_name, vectors):
    p1 = np.array([df['x'].iloc[0], df['y'].iloc[0]])
    line_vec = vectors[traj_name]
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        df['error_mm'] = 0.0
        return df
    dx = df['x'] - p1[0]
    dy = df['y'] - p1[1]
    cross_prod = np.abs(line_vec[0] * dy - dx * line_vec[1])
    df['error_mm'] = (cross_prod / line_len) * 1000 
    return df

def process_mode_data(mode_name):
    cfg = CONFIGS[mode_name]
    files = glob.glob(os.path.join(cfg['dir'], '*.txt'))
    
    all_x, all_y, all_err = [0.0], [0.0], [0.0]
    boundary_points = []

    for filepath in files:
        traj_name = os.path.basename(filepath).replace('.txt', '')
        if traj_name == 'top_trajectory' and 'top_trajectory_v2.txt' in [os.path.basename(f) for f in files]:
            continue
            
        cutoff = cfg['cutoffs'].get(traj_name, float('inf'))
        df = load_and_trim(filepath, cfg['scale'], cutoff)
        if df.empty: continue
            
        df = calc_cross_track_error(df, traj_name, cfg['vectors'])
        all_x.extend(df['x'].tolist())
        all_y.extend(df['y'].tolist())
        all_err.extend(df['error_mm'].tolist())
        boundary_points.append((df['x'].iloc[-1], df['y'].iloc[-1]))

    x, y, z = np.array(all_x), np.array(all_y), np.array(all_err)
    valid = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    x, y, z = x[valid], y[valid], z[valid]

    boundary_points = np.array(boundary_points)
    angles = np.arctan2(boundary_points[:, 1], boundary_points[:, 0])
    sorted_boundary = boundary_points[np.argsort(angles)]
    sorted_boundary = np.vstack((sorted_boundary, sorted_boundary[0]))

    grid_x, grid_y = np.mgrid[-0.5:0.5:500j, -0.5:0.5:500j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

    polygon_path = Path(sorted_boundary)
    grid_points = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    inside_polygon = polygon_path.contains_points(grid_points).reshape(grid_x.shape)
    grid_z[~inside_polygon] = np.nan

    return grid_x, grid_y, grid_z, sorted_boundary, inside_polygon, cfg['vectors']

def main():
    # Gather datasets
    kx, ky, kz, k_bound, k_inside, k_vecs = process_mode_data('Kinematic')
    fx, fy, fz, f_bound, f_inside, f_vecs = process_mode_data('Force')

    # --- THEORETICAL WORKSPACE LIMITS ---
    theo_x_m = (100.0 - (4.5 * 2) - 10.0) / 100.0   # 0.810 m
    theo_y_m = (100.0 - (3.2 * 2) - 3.0) / 100.0    # 0.906 m
    theo_area_m2 = theo_x_m * theo_y_m              # 0.7339 m^2

    # Resolution step values
    dx, dy = 1.0 / 500.0, 1.0 / 500.0
    pixel_area_m2 = dx * dy

    # Kinematic Workspace Areas
    kin_reachable_area = np.sum(k_inside) * pixel_area_m2
    kin_accurate_area = np.sum((kz <= ERROR_THRESHOLD_MM) & k_inside) * pixel_area_m2

    # Force Workspace Areas
    force_reachable_area = np.sum(f_inside) * pixel_area_m2
    force_accurate_area = np.sum((fz <= ERROR_THRESHOLD_MM) & f_inside) * pixel_area_m2

    # Display updated metrics with theoretical references
    print("="*55)
    print(f"Theoretical Maximum Area : {theo_area_m2:.4f} m^2 ({theo_x_m*100:.1f} x {theo_y_m*100:.1f} cm)")
    print("-"*55)
    print(f"Kinematic Reachable Area : {kin_reachable_area:.4f} m^2 ({ (kin_reachable_area/theo_area_m2)*100:.1f}% of theoretical)")
    print(f"Kinematic Accurate Zone  : {kin_accurate_area:.4f} m^2 ({ (kin_accurate_area/theo_area_m2)*100:.1f}% of theoretical)")
    print(f"Ratio of Usable Area     : { (kin_accurate_area/kin_reachable_area)*100:.1f}% usable")
    print("-"*55)
    print(f"Force Reachable Area     : {force_reachable_area:.4f} m^2 ({ (force_reachable_area/theo_area_m2)*100:.1f}% of theoretical)")
    print(f"Force Accurate Zone      : {force_accurate_area:.4f} m^2 ({ (force_accurate_area/theo_area_m2)*100:.1f}% of theoretical)")
    print(f"Ratio of Usable Area      : { (force_accurate_area/force_reachable_area)*100:.1f}% usable")
    print("="*55 + "\n")

    # --- PLOTTING CODE ---
    levels = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10, 15, 20, 30, 40, 60, 85]
    cmap = plt.colormaps[COLORMAP_NAME]
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig = plt.figure(figsize=(11, 5.8))
    ax1 = fig.add_axes([0.08, 0.14, 0.39, 0.67])
    ax2 = fig.add_axes([0.54, 0.14, 0.39, 0.67])
    
    # --- PANEL 1: KINEMATIC ---
    heatmap1 = ax1.contourf(kx, ky, kz, levels=levels, cmap=cmap, norm=norm, alpha=0.8)
    ax1.plot(k_bound[:, 0], k_bound[:, 1], color=HULL_COLOR, linestyle=HULL_STYLE, linewidth=HULL_WIDTH)
    k_z_filled = np.copy(kz)
    k_z_filled[~k_inside] = 9999.0
    ax1.contour(kx, ky, (k_z_filled <= ERROR_THRESHOLD_MM).astype(float), levels=[0.5], colors=ACC_COLOR, linewidths=ACC_WIDTH, linestyles=ACC_STYLE)
    for vec in k_vecs.values():
        ax1.plot([0, vec[0]], [0, vec[1]], 'k:', alpha=0.2)
    ax1.set_title('Kinematic Control', fontsize=11, pad=6)
    ax1.set_ylabel('y Position relative to Home (m)', labelpad=3)

    # --- PANEL 2: FORCE ---
    heatmap2 = ax2.contourf(fx, fy, fz, levels=levels, cmap=cmap, norm=norm, alpha=0.8)
    ax2.plot(f_bound[:, 0], f_bound[:, 1], color=HULL_COLOR, linestyle=HULL_STYLE, linewidth=HULL_WIDTH)
    f_z_filled = np.copy(fz)
    f_z_filled[~f_inside] = 9999.0
    ax2.contour(fx, fy, (f_z_filled <= ERROR_THRESHOLD_MM).astype(float), levels=[0.5], colors=ACC_COLOR, linewidths=ACC_WIDTH, linestyles=ACC_STYLE)
    for vec in f_vecs.values():
        ax2.plot([0, vec[0]], [0, vec[1]], 'k:', alpha=0.2)
    ax2.set_title('Force Control', fontsize=11, pad=6)
    ax2.set_ylabel('y Position relative to Home (m)', labelpad=3) 

    for ax in [ax1, ax2]:
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel('x Position relative to Home (m)', labelpad=3)
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.grid(True, which='major', color='black', alpha=0.15)

    cbar_ax = fig.add_axes([0.12, 0.89, 0.78, 0.025]) 
    cbar = fig.colorbar(heatmap1, cax=cbar_ax, orientation='horizontal', ticks=[0, 1, 2, 5, 10, 20, 40, 60, 85])
    cbar.set_label('Cross Track Error (mm)', labelpad=4, fontsize=9.5)
    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.xaxis.set_label_position('top')

    ax1.plot([], [], color=HULL_COLOR, linestyle=HULL_STYLE, linewidth=HULL_WIDTH, label='Reachable Workspace')
    ax1.plot([], [], color=ACC_COLOR, linestyle=ACC_STYLE, linewidth=ACC_WIDTH, label=f'Accurate Workspace (< {ERROR_THRESHOLD_MM} mm)')
    fig.legend(loc='lower center', bbox_to_anchor=(0.51, 0.01), ncol=2, frameon=False, fontsize=9.5)
    
    plt.savefig('combined_workspace_heatmaps.pdf', bbox_inches='tight')
    plt.savefig('combined_workspace_heatmaps.png', dpi=300, bbox_inches='tight')
    print("Execution complete!")
    
    plt.show()

if __name__ == '__main__':
    main()