import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import math

# --- HELPER FUNCTIONS ---
def get_bar_area(dia_mm):
    return (math.pi * (dia_mm/10)**2) / 4

def draw_support_symbol(ax, x, y, type, size=0.6): # เพิ่ม size ให้เห็นชัด
    """Draws engineering support symbols (Full & Unclipped)."""
    line_color = '#374151'
    fill_color = '#F3F4F6'
    
    if type == 'Pin':
        # Triangle
        triangle = patches.Polygon([[x, y], [x-size/2, y-size], [x+size/2, y-size]], 
                                   closed=True, edgecolor=line_color, facecolor=fill_color, linewidth=1.5, zorder=10)
        ax.add_patch(triangle)
        # Base
        ax.plot([x-size*0.8, x+size*0.8], [y-size, y-size], color=line_color, linewidth=2)
        # Hatching
        for i in np.linspace(x-size*0.6, x+size*0.6, 5):
             ax.plot([i, i-0.15], [y-size, y-size-0.25], color=line_color, linewidth=1, alpha=0.5)
            
    elif type == 'Roller':
        # Triangle
        triangle = patches.Polygon([[x, y], [x-size/2, y-size*0.7], [x+size/2, y-size*0.7]], 
                                   closed=True, edgecolor=line_color, facecolor=fill_color, linewidth=1.5, zorder=10)
        ax.add_patch(triangle)
        # Wheels
        r = size * 0.15
        c1 = patches.Circle((x-size/3, y-size*0.7-r), r, edgecolor=line_color, facecolor='white', linewidth=1.2)
        c2 = patches.Circle((x+size/3, y-size*0.7-r), r, edgecolor=line_color, facecolor='white', linewidth=1.2)
        ax.add_patch(c1)
        ax.add_patch(c2)
        # Ground
        ground_y = y - size*0.7 - 2*r - 0.05
        ax.plot([x-size*0.8, x+size*0.8], [ground_y, ground_y], color=line_color, linewidth=2)
        for i in np.linspace(x-size*0.6, x+size*0.6, 5):
             ax.plot([i, i-0.15], [ground_y, ground_y-0.25], color=line_color, linewidth=1, alpha=0.5)

    elif type == 'Fixed':
        # Vertical Wall
        ax.plot([x, x], [y-size, y+size], color=line_color, linewidth=3)
        # Hatching
        h_step = size/3
        direction = -1 if x < 0.1 else 1
        for i in np.arange(y-size, y+size, h_step):
            ax.plot([x, x + 0.3*direction], [i, i+0.3], color=line_color, linewidth=1, alpha=0.5)

# --- DRAW DIAGRAMS (Accepts probe_x) ---
def draw_diagrams(df, spans, sup_df, loads, unit_force, unit_len, probe_x=None):
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 12), sharex=True, 
                                        gridspec_kw={'height_ratios': [1, 2, 2]})
    plt.subplots_adjust(hspace=0.15)
    
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # 1. SETUP MARGINS (แก้ปัญหา Support แหว่ง)
    margin = total_len * 0.10  # เพิ่มขอบข้างละ 10%
    x_limit_min = -margin
    x_limit_max = total_len + margin
    
    # Colors
    c_beam = '#111827'
    c_shear = '#F59E0B'
    c_moment = '#2563EB'
    c_probe = '#DC2626' # สีของเส้น Probe
    
    # === AX0: MODEL ===
    ax0.set_title("Structural Model", loc='left', fontweight='bold', fontsize=11, pad=20)
    ax0.set_ylim(-1.5, 2.5)
    ax0.axis('off')
    
    # Draw Beam (0 to L only)
    ax0.plot([0, total_len], [0, 0], color=c_beam, linewidth=4, zorder=5)
    
    # Draw Supports
    for _, row in sup_df.iterrows():
        node_idx = int(row['id'])
        x_sup = cum_spans[node_idx]
        draw_support_symbol(ax0, x_sup, 0, row['type'], size=0.8)

    # Draw Loads
    for l in loads:
        span_idx = int(l['span_idx'])
        x_s = cum_spans[span_idx]
        if l['type'] == 'P':
            px = x_s + l['x']
            ax0.arrow(px, 1.8, 0, -1.5, head_width=0.2, fc='#EF4444', ec='#EF4444', width=0.03, zorder=20)
            ax0.text(px, 2.0, f"P={l['P']}", ha='center', fontweight='bold', color='#EF4444')
        elif l['type'] == 'U':
            x_e = cum_spans[span_idx+1]
            rect = patches.Rectangle((x_s, 0.1), x_e-x_s, 0.8, facecolor='#EF4444', alpha=0.1)
            ax0.add_patch(rect)
            ax0.text((x_s+x_e)/2, 1.1, f"w={l['w']}", ha='center', fontweight='bold', color='#EF4444')

    # === AX1: SHEAR ===
    ax1.set_ylabel(f"Shear ({unit_force})", fontweight='bold')
    ax1.plot(df['x'], df['shear'], color=c_shear, linewidth=2)
    ax1.fill_between(df['x'], df['shear'], 0, color=c_shear, alpha=0.2)
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # === AX2: MOMENT ===
    ax2.set_ylabel(f"Moment ({unit_force}-{unit_len})", fontweight='bold')
    ax2.set_xlabel(f"Distance ({unit_len})", fontweight='bold')
    ax2.plot(df['x'], df['moment'], color=c_moment, linewidth=2)
    ax2.fill_between(df['x'], df['moment'], 0, color=c_moment, alpha=0.2)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # === PROBE LOGIC (ส่วนสำคัญที่เพิ่มเข้ามา) ===
    if probe_x is not None:
        # 1. Draw Vertical Line on all axes
        for ax in [ax0, ax1, ax2]:
            ax.axvline(probe_x, color=c_probe, linestyle='--', linewidth=1.5, alpha=0.8)
            
        # 2. Get Values at Probe
        # Find closest row in dataframe
        idx = (df['x'] - probe_x).abs().idxmin()
        val_v = df.loc[idx, 'shear']
        val_m = df.loc[idx, 'moment']
        
        # 3. Annotate Values (Bubble Box)
        # On Model (Position Indicator)
        ax0.text(probe_x, -1.2, f"x = {probe_x:.2f} m", ha='center', 
                 bbox=dict(boxstyle="round,pad=0.3", fc=c_probe, ec=c_probe, alpha=0.1), color=c_probe, fontweight='bold')
        
        # On Shear
        ax1.plot(probe_x, val_v, 'o', color=c_probe, markersize=6)
        ax1.annotate(f"{val_v:.2f}", xy=(probe_x, val_v), xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=c_probe), color=c_probe, fontweight='bold')
                     
        # On Moment
        ax2.plot(probe_x, val_m, 'o', color=c_probe, markersize=6)
        ax2.annotate(f"{val_m:.2f}", xy=(probe_x, val_m), xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=c_probe), color=c_probe, fontweight='bold')

    # Apply Limits with Margin
    ax0.set_xlim(x_limit_min, x_limit_max)
    ax1.set_xlim(x_limit_min, x_limit_max)
    ax2.set_xlim(x_limit_min, x_limit_max)

    st.pyplot(fig)

# (Render design function keeps same logic)
def render_design_results(df, params, spans, span_props, sup_df): pass
def _design_flexure(Mu, b, d, fc, fy, phi): pass
