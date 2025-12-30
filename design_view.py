import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import pandas as pd
import numpy as np
import math

# --- HELPER FUNCTIONS ---
def get_bar_area(dia_mm):
    return (math.pi * (dia_mm/10)**2) / 4

def draw_support_symbol(ax, x, y, type, size=0.5):
    """Draws engineering support symbols stylized."""
    line_color = '#374151' # Dark Gray
    fill_color = '#E5E7EB' # Light Gray
    
    if type == 'Pin':
        triangle = patches.Polygon([[x, y], [x-size/2, y-size*0.9], [x+size/2, y-size*0.9]], 
                                   closed=True, edgecolor=line_color, facecolor=fill_color, zorder=10, linewidth=1.5)
        ax.add_patch(triangle)
        ax.plot([x-size*0.8, x+size*0.8], [y-size*0.9, y-size*0.9], color=line_color, linewidth=2)
        # Hatching
        for i in np.linspace(x-size*0.6, x+size*0.6, 5):
            ax.plot([i, i-0.15], [y-size*0.9, y-size*1.2], color=line_color, linewidth=1, alpha=0.6)
            
    elif type == 'Roller':
        triangle = patches.Polygon([[x, y], [x-size/2, y-size*0.7], [x+size/2, y-size*0.7]], 
                                   closed=True, edgecolor=line_color, facecolor=fill_color, zorder=10, linewidth=1.5)
        ax.add_patch(triangle)
        # Wheels
        r = size * 0.12
        c1 = patches.Circle((x-size/3.5, y-size*0.7-r), r, edgecolor=line_color, facecolor='white', linewidth=1.5)
        c2 = patches.Circle((x+size/3.5, y-size*0.7-r), r, edgecolor=line_color, facecolor='white', linewidth=1.5)
        ax.add_patch(c1)
        ax.add_patch(c2)
        # Ground
        ground_y = y - size*0.7 - 2*r - 0.05
        ax.plot([x-size*0.8, x+size*0.8], [ground_y, ground_y], color=line_color, linewidth=2)
        for i in np.linspace(x-size*0.6, x+size*0.6, 5):
             ax.plot([i, i-0.15], [ground_y, ground_y-0.3], color=line_color, linewidth=1, alpha=0.6)

    elif type == 'Fixed':
        # Wall Line
        ax.plot([x, x], [y-size*1.2, y+size*1.2], color=line_color, linewidth=3)
        # Hatching
        h_step = size/2.5
        direction = -1 if x < 0.1 else 1 # Hatch direction based on side
        for i in np.arange(y-size, y+size, h_step):
            ax.plot([x, x + 0.25*direction], [i, i+0.25], color=line_color, linewidth=1, alpha=0.6)

def annotate_peak(ax, x_val, y_val, color, unit="", position='top'):
    """Helper for smart annotations with background box."""
    offset_y = 15 if position == 'top' else -15
    va = 'bottom' if position == 'top' else 'top'
    
    text = ax.annotate(f"{y_val:.2f} {unit}", xy=(x_val, y_val), xytext=(0, offset_y),
                 textcoords='offset points', ha='center', va=va,
                 color='white', fontweight='bold', fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc=color, ec=color, lw=1))
    
    # Add a small marker dot at the peak
    ax.plot(x_val, y_val, marker='o', markersize=5, color=color, markeredgecolor='white', markeredgewidth=1)

# --- DRAW DIAGRAMS ---
def draw_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    
    # Use a clean, modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Adjust height ratios for better proportion
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 12), sharex=True, 
                                        gridspec_kw={'height_ratios': [1, 2, 2]})
    plt.subplots_adjust(hspace=0.15) # Tighter spacing
    
    x = df['x']
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # --- Modern Color Palette ---
    c_beam = '#1F2937'
    c_shear_line = '#F97316'  # Vibrant Orange
    c_shear_fill = '#FED7AA'  # Light Orange
    c_moment_line = '#2563EB' # Vibrant Blue
    c_moment_fill = '#BFDBFE' # Light Blue
    c_load = '#DC2626'        # Red
    c_grid = '#E5E7EB'        # Light Gray for grids
    
    # === AX0: STRUCTURAL MODEL (Full Edge) ===
    ax0.set_title("Structural Model & Loading", loc='left', fontweight='bold', fontsize=11, color=c_beam, pad=15)
    ax0.set_ylim(-1.8, 2.8)
    
    # Remove spines for edge-to-edge look
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.tick_params(left=False, labelleft=False, bottom=False) # Hide ticks
    ax0.grid(False)
    
    # Draw thick beam line exactly from 0 to total_len
    ax0.plot([0, total_len], [0, 0], color=c_beam, linewidth=4, solid_capstyle='butt', zorder=5)
    
    # Draw Supports
    for _, row in sup_df.iterrows():
        node_idx = int(row['id'])
        x_sup = cum_spans[node_idx]
        # Ensure fixed supports at edges are drawn correctly
        draw_support_symbol(ax0, x_sup, 0, row['type'], size=0.7)

    # Draw Loads (Smart styling)
    for l in loads:
        span_idx = int(l['span_idx'])
        x_start_span = cum_spans[span_idx]
        
        if l['type'] == 'P':
            px = x_start_span + l['x']
            ax0.arrow(px, 2.0, 0, -1.7, head_width=0.25, head_length=0.3, fc=c_load, ec=c_load, width=0.03, zorder=15)
            ax0.text(px, 2.2, f"P={l['P']}", ha='center', color=c_load, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=c_load, lw=1))
            
        elif l['type'] == 'U':
            x_s, x_e = x_start_span, cum_spans[span_idx+1]
            rect = patches.Rectangle((x_s, 0.05), x_e-x_s, 1.0, facecolor=c_load, alpha=0.1, edgecolor=c_load, linestyle='--')
            ax0.add_patch(rect)
            ax0.plot([x_s, x_e], [1.05, 1.05], color=c_load, linewidth=2)
            # Distributed arrows
            for xa in np.linspace(x_s+0.1, x_e-0.1, num=max(3, int((x_e-x_s)*1.5))):
                ax0.arrow(xa, 1.0, 0, -0.8, head_width=0.15, head_length=0.15, fc=c_load, ec=c_load, alpha=0.8)
            mid = (x_s + x_e)/2
            ax0.text(mid, 1.25, f"w={l['w']}", ha='center', color=c_load, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=c_load, lw=1))

    # === AX1: SHEAR (Smart Style) ===
    ax1.set_ylabel(f"Shear V ({unit_force})", fontweight='bold', color='#4B5563')
    ax1.plot(x, df['shear'], color=c_shear_line, linewidth=2.5, zorder=10)
    ax1.fill_between(x, df['shear'], 0, color=c_shear_fill, alpha=0.5, zorder=9)
    ax1.axhline(0, color=c_beam, linewidth=1, zorder=11)
    
    # Clean Grid
    ax1.grid(True, which='major', axis='y', linestyle='-', linewidth=0.5, color=c_grid)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(c_grid)
    ax1.spines['bottom'].set_visible(False)

    # Smart Annotations
    v_max, v_min = df['shear'].max(), df['shear'].min()
    if abs(v_max) > 1e-1: annotate_peak(ax1, x[df['shear'].idxmax()], v_max, c_shear_line, position='top')
    if abs(v_min) > 1e-1: annotate_peak(ax1, x[df['shear'].idxmin()], v_min, c_shear_line, position='bottom')

    # === AX2: MOMENT (Smart Style) ===
    ax2.set_ylabel(f"Moment M ({unit_force}-{unit_len})", fontweight='bold', color='#4B5563')
    ax2.set_xlabel(f"Distance along beam ({unit_len})", fontweight='bold', color='#4B5563', labelpad=10)
    ax2.plot(x, df['moment'], color=c_moment_line, linewidth=2.5, zorder=10)
    ax2.fill_between(x, df['moment'], 0, color=c_moment_fill, alpha=0.5, zorder=9)
    ax2.axhline(0, color=c_beam, linewidth=1, zorder=11)
    
    # Clean Grid
    ax2.grid(True, which='major', axis='both', linestyle='-', linewidth=0.5, color=c_grid)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(c_grid)
    ax2.spines['bottom'].set_color(c_grid)

    # Smart Annotations
    m_max, m_min = df['moment'].max(), df['moment'].min()
    if abs(m_max) > 1e-1: annotate_peak(ax2, x[df['moment'].idxmax()], m_max, c_moment_line, position='top')
    if abs(m_min) > 1e-1: annotate_peak(ax2, x[df['moment'].idxmin()], m_min, c_moment_line, position='bottom')

    # Final Touches: Vertical Grid lines for supports
    for xc in cum_spans:
        ax1.axvline(xc, color=c_grid, linestyle='-', linewidth=1, zorder=0)
        ax2.axvline(xc, color=c_grid, linestyle='-', linewidth=1, zorder=0)

    # Hard limits for edge-to-edge look
    ax0.set_xlim(0, total_len)
    ax1.set_xlim(0, total_len)
    ax2.set_xlim(0, total_len)

    st.pyplot(fig)

# (คง function อื่นๆ ไว้เหมือนเดิม)
def render_design_results(df, params, spans, span_props, sup_df): pass
def _design_flexure(Mu, b, d, fc, fy, phi): pass
