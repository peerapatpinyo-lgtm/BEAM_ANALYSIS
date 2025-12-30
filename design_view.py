import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import math

# --- CONSTANTS & CONFIG (Clean Engineering Style) ---
C_BG = '#FFFFFF'       # Pure White
C_GRID = '#E5E7EB'     # Very Light Gray
C_TEXT = '#1F2937'     # Dark Gray (Not pure black for softness)
C_BEAM = '#374151'     # Structure
C_LOAD = '#DC2626'     # Red
C_SHEAR = '#F59E0B'    # Orange (Standard for Shear)
C_MOMENT = '#2563EB'   # Blue (Standard for Moment)
C_PROBE = '#EF4444'    # Probe Line Color

# --- HELPER FUNCTIONS ---
def get_bar_area(dia_mm):
    return (math.pi * (dia_mm/10)**2) / 4

def get_rho_limits(fc, fy):
    rho_min = max(14/fy, 0.8 * math.sqrt(fc) / fy)
    beta1 = 0.85 
    if fc > 280: beta1 = max(0.65, 0.85 - 0.05 * (fc - 280) / 70)
    rho_max = 0.75 * (0.85 * beta1 * fc / fy * (6120 / (6120 + fy)))
    return rho_min, rho_max

def annotate_extremes(ax, x_series, y_series, color, unit, title):
    """
    Automatically labels the global Maximum (+) and Minimum (-) values on the chart.
    This ensures the user sees critical values immediately.
    """
    # 1. Max Positive
    if y_series.max() > 0.01:
        idx_max = y_series.idxmax()
        x_val = x_series[idx_max]
        y_val = y_series[idx_max]
        
        ax.annotate(f"{y_val:.2f}", xy=(x_val, y_val), xytext=(0, 5),
                    textcoords="offset points", ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=color,
                    bbox=dict(boxstyle="round,pad=0.1", fc='white', ec=color, alpha=0.9))
        # Dot
        ax.plot(x_val, y_val, 'o', color=color, markersize=4)

    # 2. Min Negative
    if y_series.min() < -0.01:
        idx_min = y_series.idxmin()
        x_val = x_series[idx_min]
        y_val = y_series[idx_min]
        
        ax.annotate(f"{y_val:.2f}", xy=(x_val, y_val), xytext=(0, -10),
                    textcoords="offset points", ha='center', va='top',
                    fontsize=9, fontweight='bold', color=color,
                    bbox=dict(boxstyle="round,pad=0.1", fc='white', ec=color, alpha=0.9))
        # Dot
        ax.plot(x_val, y_val, 'o', color=color, markersize=4)

def draw_support_symbol(ax, x, y, type, size=0.6):
    """Draws clear, standard support symbols on white background."""
    line_c = 'black'
    fill_c = '#F3F4F6' # Very light gray fill
    
    if type == 'Pin':
        ax.add_patch(patches.Polygon([[x, y], [x-size/2, y-size], [x+size/2, y-size]], 
                                     closed=True, ec=line_c, fc=fill_c, lw=1.2, zorder=10))
        ax.plot([x-size*0.8, x+size*0.8], [y-size, y-size], color=line_c, lw=1.5)
        # Hatching
        for i in np.linspace(x-size*0.6, x+size*0.6, 4):
             ax.plot([i, i-0.15], [y-size, y-size-0.25], color=line_c, lw=0.8)
            
    elif type == 'Roller':
        ax.add_patch(patches.Polygon([[x, y], [x-size/2, y-size*0.7], [x+size/2, y-size*0.7]], 
                                     closed=True, ec=line_c, fc=fill_c, lw=1.2, zorder=10))
        # Wheels
        r = size * 0.15
        ax.add_patch(patches.Circle((x-size/3, y-size*0.7-r), r, ec=line_c, fc='white', lw=1.2))
        ax.add_patch(patches.Circle((x+size/3, y-size*0.7-r), r, ec=line_c, fc='white', lw=1.2))
        # Ground
        ground_y = y - size*0.7 - 2*r - 0.05
        ax.plot([x-size*0.8, x+size*0.8], [ground_y, ground_y], color=line_c, lw=1.5)
        for i in np.linspace(x-size*0.6, x+size*0.6, 4):
             ax.plot([i, i-0.15], [ground_y, ground_y-0.25], color=line_c, lw=0.8)

    elif type == 'Fixed':
        ax.plot([x, x], [y-size, y+size], color=line_c, lw=2.5)
        h_step = size/3
        direction = -1 if x < 0.1 else 1
        for i in np.arange(y-size, y+size, h_step):
            ax.plot([x, x + 0.25*direction], [i, i+0.25], color=line_c, lw=0.8)

# =========================================
# DRAWING MAIN FUNCTION
# =========================================
def draw_diagrams(df, spans, sup_df, loads, unit_force, unit_len, probe_x=None):
    
    # Reset to Clean White Style
    plt.rcParams.update({
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'text.color': C_TEXT,
        'axes.labelcolor': C_TEXT,
        'xtick.color': C_TEXT,
        'ytick.color': C_TEXT,
        'grid.color': C_GRID,
        'font.family': 'sans-serif',
    })
    
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 11), sharex=True, 
                                        gridspec_kw={'height_ratios': [0.8, 2, 2]})
    plt.subplots_adjust(hspace=0.25) # Give some breathing room
    
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # Margin for full support visibility
    margin = total_len * 0.08
    x_lims = [-margin, total_len + margin]
    
    # === AX0: MODEL (Simple & Clear) ===
    ax0.set_title("Structural Model", loc='left', fontweight='bold', fontsize=10)
    ax0.set_ylim(-1.5, 2.5)
    ax0.axis('off')
    
    # Beam
    ax0.plot([0, total_len], [0, 0], color='black', linewidth=3, zorder=5)
    
    # Supports
    for _, row in sup_df.iterrows():
        draw_support_symbol(ax0, cum_spans[int(row['id'])], 0, row['type'], size=0.8)

    # Loads
    for l in loads:
        x_s = cum_spans[int(l['span_idx'])]
        if l['type'] == 'P':
            px = x_s + l['x']
            ax0.arrow(px, 1.8, 0, -1.4, head_width=0.2, fc=C_LOAD, ec=C_LOAD, width=0.03)
            ax0.text(px, 2.0, f"P={l['P']}", ha='center', color=C_LOAD, fontweight='bold', fontsize=9)
        elif l['type'] == 'U':
            x_e = cum_spans[int(l['span_idx'])+1]
            rect = patches.Rectangle((x_s, 0.1), x_e-x_s, 0.8, facecolor=C_LOAD, alpha=0.15)
            ax0.add_patch(rect)
            ax0.plot([x_s, x_e], [0.9, 0.9], color=C_LOAD, lw=1.5)
            ax0.text((x_s+x_e)/2, 1.05, f"w={l['w']}", ha='center', color=C_LOAD, fontweight='bold', fontsize=9)

    # === AX1: SHEAR (Clean Orange) ===
    ax1.set_ylabel(f"Shear ({unit_force})", fontweight='bold')
    ax1.plot(df['x'], df['shear'], color=C_SHEAR, linewidth=2)
    ax1.fill_between(df['x'], df['shear'], 0, color=C_SHEAR, alpha=0.2)
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.grid(True, linestyle='-', linewidth=0.5, color=C_GRID)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Auto-Label Max/Min
    annotate_extremes(ax1, df['x'], df['shear'], C_SHEAR, unit_force, "V")

    # === AX2: MOMENT (Clean Blue) ===
    ax2.set_ylabel(f"Moment ({unit_force}-{unit_len})", fontweight='bold')
    ax2.set_xlabel(f"Distance ({unit_len})", fontweight='bold')
    ax2.plot(df['x'], df['moment'], color=C_MOMENT, linewidth=2)
    ax2.fill_between(df['x'], df['moment'], 0, color=C_MOMENT, alpha=0.2)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.grid(True, linestyle='-', linewidth=0.5, color=C_GRID)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Auto-Label Max/Min
    annotate_extremes(ax2, df['x'], df['moment'], C_MOMENT, unit_force, "M")

    # === INTERACTIVE PROBE (Optional detail check) ===
    if probe_x is not None:
        for ax in [ax0, ax1, ax2]:
            ax.axvline(probe_x, color=C_TEXT, linestyle=':', linewidth=1, alpha=0.6)
            
        idx = (df['x'] - probe_x).abs().idxmin()
        val_v = df.loc[idx, 'shear']
        val_m = df.loc[idx, 'moment']
        
        # Probe bubbles (only appear on the lines)
        ax1.plot(probe_x, val_v, 'o', color='white', markeredgecolor=C_TEXT, markersize=6)
        ax2.plot(probe_x, val_m, 'o', color='white', markeredgecolor=C_TEXT, markersize=6)

    for ax in [ax0, ax1, ax2]: ax.set_xlim(x_lims)
    st.pyplot(fig)

# --- DESIGN LOGIC (Senior Level) ---
# (ส่วนนี้ Logic ดีแล้ว ผมแค่ก๊อปปี้ส่วน Render Table ให้มาอยู่ไฟล์นี้ให้ครบครับ)
def render_design_results(df, params, spans, span_props, sup_df):
    fc, fy = params['fc'], params['fy']
    phi_b, phi_v = 0.90, 0.75 

    results_data = []
    cum_dist = 0.0
    
    st.markdown("### 🧱 Design Calculation Sheet")
    
    for i, L in enumerate(spans):
        prop = span_props[i]
        b, h, cv = prop['b'], prop['h'], prop['cv']
        db_main, db_stir = prop['main_bar_dia'], prop['stirrup_dia']
        
        mask = (df['x'] >= cum_dist) & (df['x'] <= cum_dist + L)
        span_data = df[mask]
        Mu_pos = max(0, span_data['moment'].max()) * 100
        Vu_max = span_data['shear'].abs().max()

        # Calculations
        Ab_main = get_bar_area(db_main)
        Ab_stir = get_bar_area(db_stir)
        d = h - cv - (db_stir/10) - (db_main/20)
        
        # Flexure
        As_approx = Mu_pos / (phi_b * fy * 0.9 * d) if Mu_pos > 0 else 0
        num_bars = max(2, math.ceil(As_approx / Ab_main)) if As_approx > 0 else 2
        As_provided = num_bars * Ab_main
        
        a = (As_provided * fy) / (0.85 * fc * b)
        Mn = As_provided * fy * (d - a/2)
        phi_Mn = phi_b * Mn
        dc_flex = Mu_pos / phi_Mn if phi_Mn > 0 else 0
        
        # Checks
        rho = As_provided / (b * d)
        rho_min, rho_max = get_rho_limits(fc, fy)
        status_flex = "✅ OK" if dc_flex <= 1.0 else "❌ Fail"
        if rho < rho_min: status_flex = "⚠️ < Min"
        if rho > rho_max: status_flex = "❌ > Max"

        # Shear
        Vc = 0.53 * math.sqrt(fc) * b * d 
        phi_Vc = phi_v * Vc
        stirrup_txt = "-"
        
        if Vu_max > phi_Vc / 2:
            Av = 2 * Ab_stir
            if Vu_max > phi_Vc:
                Vs_req = (Vu_max / phi_v) - Vc
                s_calc = (Av * fy * d) / Vs_req
            else:
                s_calc = 999 
            
            s_max = min(d/2, 60)
            s_prov = max(5, int(min(s_calc, s_max)))
            stirrup_txt = f"RB{db_stir}@{s_prov}"
            
            Vs_prov = (Av * fy * d) / s_prov
            phi_Vn = phi_v * (Vc + Vs_prov)
            dc_shear = Vu_max / phi_Vn
        else:
            dc_shear = Vu_max / phi_Vc
            stirrup_txt = "Min"

        results_data.append({
            "Span": f"{i+1}",
            "Section": f"{b:.0f}x{h:.0f}",
            "Reinforcement": f"{num_bars}-DB{db_main}",
            "Mu+ (kg-m)": f"{Mu_pos/100:.0f}",
            "φMn": f"{phi_Mn/100:.0f}",
            "D/C (M)": f"{dc_flex:.2f} {status_flex}",
            "Vu (kg)": f"{Vu_max:.0f}",
            "Stirrups": stirrup_txt,
            "D/C (V)": f"{dc_shear:.2f}"
        })
        cum_dist += L

    st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)
