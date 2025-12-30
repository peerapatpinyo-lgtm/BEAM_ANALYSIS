import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import math

# --- HELPER FUNCTIONS ---
def get_bar_area(dia_mm):
    return (math.pi * (dia_mm/10)**2) / 4

def draw_support_symbol(ax, x, y, type, size=0.5):
    """
    Draws standard engineering support symbols.
    """
    if type == 'Pin':
        # Triangle
        triangle = patches.Polygon([[x, y], [x-size/2, y-size], [x+size/2, y-size]], 
                                   closed=True, edgecolor='black', facecolor='#CFD8DC', zorder=10)
        ax.add_patch(triangle)
        # Base Line
        ax.plot([x-size, x+size], [y-size, y-size], color='black', linewidth=1.5)
        # Hatching (Diagonal lines)
        for i in np.linspace(x-size, x+size, 6):
            ax.plot([i, i-0.1], [y-size, y-size-0.2], color='black', linewidth=0.8)
            
    elif type == 'Roller':
        # Triangle
        triangle = patches.Polygon([[x, y], [x-size/2, y-size*0.8], [x+size/2, y-size*0.8]], 
                                   closed=True, edgecolor='black', facecolor='#CFD8DC', zorder=10)
        ax.add_patch(triangle)
        # Wheels (Circles)
        r = size * 0.15
        c1 = patches.Circle((x-size/3, y-size*0.8-r), r, edgecolor='black', facecolor='white')
        c2 = patches.Circle((x+size/3, y-size*0.8-r), r, edgecolor='black', facecolor='white')
        ax.add_patch(c1)
        ax.add_patch(c2)
        # Base Line (Ground)
        ground_y = y - size*0.8 - 2*r
        ax.plot([x-size, x+size], [ground_y, ground_y], color='black', linewidth=1.5)
        # Hatching
        for i in np.linspace(x-size, x+size, 6):
            ax.plot([i, i-0.1], [ground_y, ground_y-0.2], color='black', linewidth=0.8)

    elif type == 'Fixed':
        # Vertical Line
        ax.plot([x, x], [y-size, y+size], color='black', linewidth=2.5)
        # Hatching (Wall style)
        h_step = size/3
        for i in np.arange(y-size, y+size, h_step):
            # Check direction (usually left or right depending on span, but generic here)
            ax.plot([x, x-0.2], [i, i+0.2], color='black', linewidth=0.8)

# --- DRAW DIAGRAMS ---
def draw_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 12), sharex=True, 
                                        gridspec_kw={'height_ratios': [1.2, 2, 2]})
    plt.subplots_adjust(hspace=0.25)
    
    x = df['x']
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # Styles
    color_shear = '#F59E0B'
    color_moment = '#3B82F6'
    color_load = '#DC2626'
    
    # === AX0: STRUCTURAL MODEL ===
    ax0.set_title("Structural Model", loc='left', fontweight='bold', fontsize=12, pad=10)
    ax0.set_ylim(-1.5, 2.5)
    ax0.axis('off')
    
    # Draw Beam
    ax0.plot([0, total_len], [0, 0], color='#1F2937', linewidth=3, solid_capstyle='butt', zorder=5)
    
    # Draw Supports (Correct Engineering Symbols)
    for _, row in sup_df.iterrows():
        node_idx = int(row['id'])
        x_sup = cum_spans[node_idx]
        stype = row['type']
        draw_support_symbol(ax0, x_sup, 0, stype, size=0.6)

    # Draw Loads
    for l in loads:
        span_idx = int(l['span_idx'])
        x_start_span = cum_spans[span_idx]
        
        if l['type'] == 'P':
            px = x_start_span + l['x']
            p_mag = l['P']
            # Large Arrow
            ax0.arrow(px, 1.5, 0, -1.3, head_width=0.2, head_length=0.2, fc=color_load, ec=color_load, width=0.02)
            ax0.text(px, 1.7, f"P={p_mag}", ha='center', color=color_load, fontweight='bold')
            
        elif l['type'] == 'U':
            w_mag = l['w']
            x_s = x_start_span
            x_e = cum_spans[span_idx+1]
            
            # Draw Load Block
            rect = patches.Rectangle((x_s, 0), x_e-x_s, 0.8, facecolor=color_load, alpha=0.15, edgecolor=None)
            ax0.add_patch(rect)
            # Top Line
            ax0.plot([x_s, x_e], [0.8, 0.8], color=color_load, linewidth=1.5)
            # Arrows
            for xa in np.linspace(x_s, x_e, num=int((x_e-x_s)*2)+3):
                ax0.arrow(xa, 0.8, 0, -0.65, head_width=0.1, head_length=0.15, fc=color_load, ec=color_load, alpha=0.7)
            
            mid = (x_s + x_e)/2
            ax0.text(mid, 0.95, f"w={w_mag}", ha='center', color=color_load, fontweight='bold')

    # === AX1: SHEAR ===
    ax1.plot(x, df['shear'], color=color_shear, linewidth=2)
    ax1.fill_between(x, df['shear'], 0, color=color_shear, alpha=0.1)
    ax1.set_ylabel(f"Shear ({unit_force})", fontweight='bold')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Annotate Max/Min Shear
    v_max = df['shear'].max()
    v_min = df['shear'].min()
    ax1.text(x[df['shear'].idxmax()], v_max, f"{v_max:.2f}", color=color_shear, fontweight='bold', ha='center', va='bottom')
    ax1.text(x[df['shear'].idxmin()], v_min, f"{v_min:.2f}", color=color_shear, fontweight='bold', ha='center', va='top')

    # === AX2: MOMENT ===
    ax2.plot(x, df['moment'], color=color_moment, linewidth=2)
    ax2.fill_between(x, df['moment'], 0, color=color_moment, alpha=0.1)
    ax2.set_ylabel(f"Moment ({unit_force}-{unit_len})", fontweight='bold')
    ax2.set_xlabel(f"Length ({unit_len})", fontweight='bold')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Annotate Max/Min Moment
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    ax2.text(x[df['moment'].idxmax()], m_max, f"{m_max:.2f}", color=color_moment, fontweight='bold', ha='center', va='bottom')
    ax2.text(x[df['moment'].idxmin()], m_min, f"{m_min:.2f}", color='red', fontweight='bold', ha='center', va='top')

    # Vertical Grid Lines for Spans
    for xc in cum_spans:
        for ax in [ax0, ax1, ax2]:
            ax.axvline(xc, color='gray', linestyle=':', alpha=0.5)

    ax0.set_xlim(-0.5, total_len + 0.5)
    ax1.set_xlim(0, total_len)
    ax2.set_xlim(0, total_len)

    st.pyplot(fig)

# (ส่วน render_design_results และ _design_flexure ให้คงเดิมไว้ครับ)
def render_design_results(df, params, spans, span_props, sup_df):
    # ... (Copy logic เดิมมาใส่) ...
    pass 
def _design_flexure(Mu, b, d, fc, fy, phi):
    # ... (Copy logic เดิมมาใส่) ...
    pass
