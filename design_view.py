import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import math

# --- HELPER FUNCTIONS ---
def get_bar_area(dia_mm):
    return (math.pi * (dia_mm/10)**2) / 4

# --- 1. DRAW DIAGRAMS (Full Structural Model) ---
def draw_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    """
    Plots:
    1. Beam Model (Supports + Loads)
    2. Shear Force Diagram (SFD)
    3. Bending Moment Diagram (BMD)
    """
    # Create 3 subplots: Model, Shear, Moment
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 10), sharex=True, 
                                        gridspec_kw={'height_ratios': [1, 2, 2]})
    plt.subplots_adjust(hspace=0.2)
    
    x = df['x']
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # Styles
    color_beam = '#374151'
    color_shear = '#F59E0B'  # Amber
    color_shear_fill = '#FEF3C7'
    color_moment = '#3B82F6' # Blue
    color_moment_fill = '#DBEAFE'
    color_load = '#EF4444'   # Red for loads
    
    # === AX0: BEAM MODEL ===
    ax0.set_title("Structural Model & Loading", loc='left', fontsize=10, fontweight='bold', color='#374151')
    ax0.set_ylim(-1.5, 2.0) # Fixed height for drawing
    ax0.axis('off') # Hide axis box
    
    # Draw Beam Line
    ax0.plot([0, total_len], [0, 0], color='black', linewidth=4, solid_capstyle='round')
    
    # Draw Supports
    for _, row in sup_df.iterrows():
        node_idx = int(row['id'])
        x_sup = cum_spans[node_idx]
        stype = row['type']
        
        # Draw Triangle for Pin/Roller
        triangle = patches.Polygon([[x_sup-0.2, -0.4], [x_sup+0.2, -0.4], [x_sup, 0]], 
                                   closed=True, edgecolor='black', facecolor='#9CA3AF')
        ax0.add_patch(triangle)
        
        # Label Support
        ax0.text(x_sup, -0.7, stype, ha='center', fontsize=8, color='#4B5563')
        
        # If Fixed, add hatching
        if stype == 'Fixed':
            rect = patches.Rectangle((x_sup-0.25, -0.4), 0.5, 0.05, color='black')
            ax0.add_patch(rect)

    # Draw Loads
    max_load_val = 1.0 # For scaling arrows
    if loads:
        vals = [l['w'] if l['type']=='U' else l['P'] for l in loads]
        if vals: max_load_val = max(vals)

    for l in loads:
        span_idx = int(l['span_idx'])
        x_start_span = cum_spans[span_idx]
        
        if l['type'] == 'P':
            # Point Load
            px = x_start_span + l['x']
            p_mag = l['P']
            # Draw Arrow
            ax0.arrow(px, 1.0, 0, -0.8, head_width=0.15, head_length=0.2, fc=color_load, ec=color_load)
            ax0.text(px, 1.1, f"{p_mag}", ha='center', fontsize=9, color=color_load, fontweight='bold')
            
        elif l['type'] == 'U':
            # Uniform Load
            w_mag = l['w']
            x_start = x_start_span
            x_end = cum_spans[span_idx+1]
            
            # Draw Block
            rect = patches.Rectangle((x_start, 0), x_end-x_start, 0.6, alpha=0.3, color=color_load)
            ax0.add_patch(rect)
            # Draw Arrows inside
            mid = (x_start + x_end)/2
            ax0.text(mid, 0.7, f"w = {w_mag}", ha='center', fontsize=9, color=color_load, fontweight='bold')
            # Small arrows
            for xa in np.linspace(x_start, x_end, 5):
                ax0.arrow(xa, 0.6, 0, -0.4, head_width=0.05, head_length=0.1, fc=color_load, ec=color_load, alpha=0.6)

    # === AX1: SHEAR (SFD) ===
    ax1.plot(x, df['shear'], color=color_shear, linewidth=2)
    ax1.fill_between(x, df['shear'], 0, color=color_shear_fill, alpha=0.8)
    ax1.set_ylabel(f"Shear V ({unit_force})", fontsize=9, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Label Max/Min
    v_max = df['shear'].max()
    v_min = df['shear'].min()
    if abs(v_max) > 1e-3:
        ax1.text(x[df['shear'].idxmax()], v_max, f"{v_max:.2f}", color=color_shear, fontweight='bold', ha='center', va='bottom')
    if abs(v_min) > 1e-3:
        ax1.text(x[df['shear'].idxmin()], v_min, f"{v_min:.2f}", color=color_shear, fontweight='bold', ha='center', va='top')

    # === AX2: MOMENT (BMD) ===
    ax2.plot(x, df['moment'], color=color_moment, linewidth=2)
    ax2.fill_between(x, df['moment'], 0, color=color_moment_fill, alpha=0.8)
    ax2.set_ylabel(f"Moment M ({unit_force}-{unit_len})", fontsize=9, fontweight='bold')
    ax2.set_xlabel(f"Distance x ({unit_len})", fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Label Max/Min
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    if abs(m_max) > 1e-3:
        ax2.text(x[df['moment'].idxmax()], m_max, f"{m_max:.2f}", color=color_moment, fontweight='bold', ha='center', va='bottom')
    if abs(m_min) > 1e-3:
        ax2.text(x[df['moment'].idxmin()], m_min, f"{m_min:.2f}", color='#DC2626', fontweight='bold', ha='center', va='top')

    # Draw vertical lines for supports on all plots
    for xc in cum_spans:
        ax0.axvline(xc, color='gray', linestyle=':', alpha=0.3)
        ax1.axvline(xc, color='gray', linestyle=':', alpha=0.3)
        ax2.axvline(xc, color='gray', linestyle=':', alpha=0.3)

    ax0.set_xlim(0, total_len)
    ax1.set_xlim(0, total_len)
    ax2.set_xlim(0, total_len)

    st.pyplot(fig)

# --- 2. RENDER DESIGN TABLE (UNCHANGED) ---
def render_design_results(df, params, spans, span_props, sup_df):
    # (ใช้ Code เดิมจากรอบที่แล้วได้เลยครับ ส่วนนี้ logic การออกแบบยังเหมือนเดิม)
    # เพื่อความกระชับ ผมขอละไว้ในคำตอบนี้ แต่ในไฟล์จริงต้องมีนะครับ
    # ...
    # ... 
    pass 

# (อย่าลืม copy function _design_flexure และอื่นๆ มาด้วยนะครับ)
def _design_flexure(Mu, b, d, fc, fy, phi):
     # (ใช้ Code เดิม)
     pass
