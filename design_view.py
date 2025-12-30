import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import math

# --- STYLING CONSTANTS ---
C_BG = '#FFFFFF'
C_GRID = '#F3F4F6'     # Very light gray for minimal distraction
C_TEXT = '#374151'     # Charcoal gray
C_BEAM = '#111827'     # Almost black
C_LOAD = '#DC2626'     # Engineering Red
C_SHEAR = '#F59E0B'    # Safety Orange
C_MOMENT = '#2563EB'   # Structural Blue
C_PROBE = '#EF4444'

# --- HELPER FUNCTIONS ---
def get_bar_area(dia_mm):
    return (math.pi * (dia_mm/10)**2) / 4

def get_rho_limits(fc, fy):
    rho_min = max(14/fy, 0.8 * math.sqrt(fc) / fy)
    beta1 = 0.85 
    if fc > 280: beta1 = max(0.65, 0.85 - 0.05 * (fc - 280) / 70)
    rho_max = 0.75 * (0.85 * beta1 * fc / fy * (6120 / (6120 + fy)))
    return rho_min, rho_max

def set_smart_ylim(ax, data, margin_factor=0.2):
    """Sets y-limits with symmetric or balanced padding."""
    y_min, y_max = data.min(), data.max()
    span = y_max - y_min
    if span == 0: span = 1.0 # Prevent singular matrix error
    
    # Add breathing room
    pad = span * margin_factor
    ax.set_ylim(y_min - pad, y_max + pad * 1.5) # Extra pad top for labels

def annotate_extremes(ax, x, y, color, unit, label_prefix=""):
    """Annotates Max and Min values nicely."""
    # Max
    if y.max() > 1e-3:
        idx = y.idxmax()
        val = y[idx]
        ax.annotate(f"{val:.2f}", xy=(x[idx], val), xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=color,
                    bbox=dict(boxstyle="round,pad=0.1", fc='white', ec='none', alpha=0.8))
        ax.plot(x[idx], val, '.', color=color)

    # Min
    if y.min() < -1e-3:
        idx = y.idxmin()
        val = y[idx]
        ax.annotate(f"{val:.2f}", xy=(x[idx], val), xytext=(0, -10), textcoords="offset points",
                    ha='center', va='top', fontsize=9, fontweight='bold', color=color,
                    bbox=dict(boxstyle="round,pad=0.1", fc='white', ec='none', alpha=0.8))
        ax.plot(x[idx], val, '.', color=color)

def draw_support_symbol(ax, x, y, type, size=1.0):
    """Draws standardized structural supports."""
    line_c = '#4B5563'
    fill_c = 'white'
    
    if type == 'Pin':
        ax.add_patch(patches.Polygon([[x, y], [x-size*0.4, y-size*0.8], [x+size*0.4, y-size*0.8]], 
                                     closed=True, ec=line_c, fc=fill_c, lw=1.5, zorder=10))
        ax.plot([x-size*0.6, x+size*0.6], [y-size*0.8, y-size*0.8], color=line_c, lw=2)
            
    elif type == 'Roller':
        ax.add_patch(patches.Polygon([[x, y], [x-size*0.4, y-size*0.6], [x+size*0.4, y-size*0.6]], 
                                     closed=True, ec=line_c, fc=fill_c, lw=1.5, zorder=10))
        # Wheels
        r = size * 0.12
        ax.add_patch(patches.Circle((x-size*0.25, y-size*0.6-r), r, ec=line_c, fc=fill_c, lw=1.2))
        ax.add_patch(patches.Circle((x+size*0.25, y-size*0.6-r), r, ec=line_c, fc=fill_c, lw=1.2))
        # Ground
        ground_y = y - size*0.6 - 2*r - 0.05
        ax.plot([x-size*0.6, x+size*0.6], [ground_y, ground_y], color=line_c, lw=2)

    elif type == 'Fixed':
        ax.plot([x, x], [y-size*0.8, y+size*0.8], color=line_c, lw=3)
        # Hatching
        h_step = size/4
        direction = -1 if x < 0.1 else 1
        for i in np.arange(y-size*0.8, y+size*0.8, h_step):
            ax.plot([x, x + 0.2*direction], [i, i+0.2], color=line_c, lw=1)

# =========================================
# MAIN PLOT FUNCTION
# =========================================
def draw_diagrams(df, spans, sup_df, loads, unit_force, unit_len, probe_x=None):
    
    # 1. Setup Canvas (Portrait Ratio is best for stacked graphs)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9})
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 10), sharex=True, 
                                        gridspec_kw={'height_ratios': [0.7, 1, 1]})
    
    # Smart spacing
    plt.subplots_adjust(hspace=0.25, top=0.95, bottom=0.05, left=0.1, right=0.95)
    
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # Set X-Limits with fixed padding (prevent edge cropping)
    margin_x = total_len * 0.05
    x_limits = [-margin_x, total_len + margin_x]

    # === AX0: STRUCTURAL MODEL ===
    ax0.set_title("Structural Model", loc='left', fontweight='bold', color=C_TEXT)
    ax0.axis('off') # Turn off box, we will draw what we need
    ax0.set_ylim(-1.5, 3.0) # Fixed visual height for supports(neg) and loads(pos)

    # Beam Line
    ax0.plot([0, total_len], [0, 0], color=C_BEAM, linewidth=4, solid_capstyle='butt', zorder=5)
    
    # Supports
    for _, row in sup_df.iterrows():
        draw_support_symbol(ax0, cum_spans[int(row['id'])], 0, row['type'], size=1.0)

    # Loads (Dynamic Scaling)
    # Find max load to scale arrow sizes relatively
    max_P = max([l['P'] for l in loads if l['type']=='P'] + [0])
    max_w = max([l['w'] for l in loads if l['type']=='U'] + [0])
    
    for l in loads:
        x_s = cum_spans[int(l['span_idx'])]
        if l['type'] == 'P':
            px = x_s + l['x']
            # Arrow size relative to max
            scale = 0.5 + 1.0 * (l['P'] / max_P if max_P > 0 else 1)
            arrow_len = min(2.0, max(1.0, scale)) # Cap size
            
            ax0.arrow(px, 0.2 + arrow_len, 0, -arrow_len, head_width=0.25, fc=C_LOAD, ec=C_LOAD, width=0.04, zorder=20)
            ax0.text(px, 0.3 + arrow_len, f"P={l['P']}", ha='center', color=C_LOAD, fontweight='bold')
            
        elif l['type'] == 'U':
            x_e = cum_spans[int(l['span_idx'])+1]
            rect = patches.Rectangle((x_s, 0.1), x_e-x_s, 0.8, facecolor=C_LOAD, alpha=0.1)
            ax0.add_patch(rect)
            ax0.plot([x_s, x_e], [0.9, 0.9], color=C_LOAD, lw=1.5)
            # Center label
            ax0.text((x_s+x_e)/2, 1.1, f"w={l['w']}", ha='center', color=C_LOAD, fontweight='bold', 
                     bbox=dict(fc='white', ec='none', pad=0, alpha=0.7))
            # Mini arrows
            for xa in np.linspace(x_s+0.1, x_e-0.1, num=max(3, int((x_e-x_s)))):
                ax0.arrow(xa, 0.9, 0, -0.6, head_width=0.1, fc=C_LOAD, ec=C_LOAD, alpha=0.6)

    # === AX1: SHEAR DIAGRAM ===
    ax1.set_ylabel(f"Shear ({unit_force})", fontweight='bold', color=C_TEXT)
    ax1.plot(df['x'], df['shear'], color=C_SHEAR, linewidth=2)
    ax1.fill_between(df['x'], df['shear'], 0, color=C_SHEAR, alpha=0.15)
    ax1.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    
    # Clean Grid
    ax1.grid(True, which='major', axis='y', color=C_GRID, linestyle='-', linewidth=1)
    ax1.grid(True, which='major', axis='x', color=C_GRID, linestyle=':', linewidth=1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    
    set_smart_ylim(ax1, df['shear']) # Auto-fit height
    annotate_extremes(ax1, df['x'], df['shear'], C_SHEAR, unit_force)

    # === AX2: MOMENT DIAGRAM ===
    ax2.set_ylabel(f"Moment ({unit_force}-{unit_len})", fontweight='bold', color=C_TEXT)
    ax2.set_xlabel(f"Beam Length ({unit_len})", fontweight='bold', color=C_TEXT)
    
    ax2.plot(df['x'], df['moment'], color=C_MOMENT, linewidth=2)
    ax2.fill_between(df['x'], df['moment'], 0, color=C_MOMENT, alpha=0.15)
    ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    
    # Clean Grid
    ax2.grid(True, which='major', axis='y', color=C_GRID, linestyle='-', linewidth=1)
    ax2.grid(True, which='major', axis='x', color=C_GRID, linestyle=':', linewidth=1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    set_smart_ylim(ax2, df['moment']) # Auto-fit height
    annotate_extremes(ax2, df['x'], df['moment'], C_MOMENT, unit_force)

    # === PROBE ===
    if probe_x is not None:
        for ax in [ax0, ax1, ax2]:
            ax.axvline(probe_x, color='#9CA3AF', linestyle='--', linewidth=1)
        
        # Get values
        idx = (df['x'] - probe_x).abs().idxmin()
        val_v = df.loc[idx, 'shear']
        val_m = df.loc[idx, 'moment']
        
        # Draw Dots on Diagrams
        ax1.plot(probe_x, val_v, 'o', color='white', markeredgecolor=C_SHEAR, markersize=6, zorder=100)
        ax2.plot(probe_x, val_m, 'o', color='white', markeredgecolor=C_MOMENT, markersize=6, zorder=100)

    # Final X-Limits
    for ax in [ax0, ax1, ax2]: ax.set_xlim(x_limits)

    st.pyplot(fig)

# --- DESIGN TABLE RENDERER (Clean Table) ---
def render_design_results(df, params, spans, span_props, sup_df):
    fc, fy = params['fc'], params['fy']
    phi_b, phi_v = 0.90, 0.75 

    results_data = []
    cum_dist = 0.0
    
    st.markdown("#### 🧱 RC Design Checks")
    
    for i, L in enumerate(spans):
        prop = span_props[i]
        b, h, cv = prop['b'], prop['h'], prop['cv']
        db_main, db_stir = prop['main_bar_dia'], prop['stirrup_dia']
        
        mask = (df['x'] >= cum_dist) & (df['x'] <= cum_dist + L)
        span_data = df[mask]
        Mu_pos = max(0, span_data['moment'].max()) * 100
        Vu_max = span_data['shear'].abs().max()

        # Calculation
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
        
        status_flex = "OK" if dc_flex <= 1.0 else "Fail"

        # Shear
        Vc = 0.53 * math.sqrt(fc) * b * d 
        phi_Vc = phi_v * Vc
        
        if Vu_max > phi_Vc / 2:
            s_prov = 15 # Placeholder logic for dashboard summary
            stirrup_txt = f"RB{db_stir}@{s_prov}"
            Av = 2 * Ab_stir
            Vs_prov = (Av * fy * d) / s_prov
            phi_Vn = phi_v * (Vc + Vs_prov)
            dc_shear = Vu_max / phi_Vn
        else:
            stirrup_txt = "Min"
            dc_shear = Vu_max / phi_Vc

        results_data.append({
            "Span": f"{i+1}",
            "Section": f"{b:.0f}x{h:.0f}",
            "Rebar": f"{num_bars}-DB{db_main}",
            "Mu+": f"{Mu_pos/100:.0f}",
            "φMn": f"{phi_Mn/100:.0f}",
            "Flex Check": f"{dc_flex:.2f} ({status_flex})",
            "Vu": f"{Vu_max:.0f}",
            "Stirrup": stirrup_txt,
            "Shear D/C": f"{dc_shear:.2f}"
        })
        cum_dist += L

    # Styling the table
    st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)
