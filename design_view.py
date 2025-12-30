import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import pandas as pd
import numpy as np
import math

# --- CONSTANTS & CONFIG ---
# Modern Professional Color Palette (Dark Theme Optimized)
C_BG = '#111827'       # Dark background like engineering blueprint space
C_GRID = '#374151'     # Subtle grid lines
C_TEXT = '#E5E7EB'     # High contrast text
C_BEAM = '#6B7280'     # Structure color
C_LOAD = '#EF4444'     # Alert red for loads
C_SHEAR = '#06B6D4'    # Cyan/Teal for Shear (Professional look)
C_MOMENT = '#6366F1'   # Indigo/Purple for Moment (Trustworthy look)
C_PROBE = '#FACC15'    # Bright Yellow for interactive probe

# --- HELPER FUNCTIONS ---
def get_bar_area(dia_mm):
    """Area of one bar in cm2"""
    return (math.pi * (dia_mm/10)**2) / 4

def get_rho_limits(fc, fy):
    """Calculate rho_min and approximate rho_max (ductility limit)"""
    # fc, fy in ksc
    # rho_min (ACI/EIT metric)
    rho_min = max(14/fy, 0.8 * math.sqrt(fc) / fy)
    
    # rho_balanced (approximate beta1 for typical concrete)
    beta1 = 0.85 
    if fc > 280: beta1 = max(0.65, 0.85 - 0.05 * (fc - 280) / 70)
    
    # rho_max for tension controlled (approx 0.75 rho_b or strain limit 0.005)
    # Using simplified tension-controlled limit (~0.63 rho_b for fy=4000)
    # A common practical limit for seniors is around 1.5% - 2.5% depending on beam size to avoid congestion
    # Let's use a calculated ductility limit based on strain 0.005
    # rho_t = 0.85 * beta1 * (fc/fy) * (6120 / (6120 + fy)) # for E_s = 2.04e6 ksc
    # Simplified approach for dashboard:
    rho_max = 0.75 * (0.85 * beta1 * fc / fy * (6120 / (6120 + fy)))

    return rho_min, rho_max

def add_glow_effect(line):
    """Adds a subtle neon glow to plot lines."""
    line.set_path_effects([path_effects.Stroke(linewidth=4, foreground=line.get_color(), alpha=0.3),
                           path_effects.Normal()])

# --- SUPPORT DRAWING (Modernized) ---
def draw_support_symbol(ax, x, y, type, size=0.6):
    """Draws stylized support symbols."""
    line_color = C_TEXT
    fill_color = C_BEAM
    
    if type == 'Pin':
        # Modern Triangle
        ax.add_patch(patches.Polygon([[x, y], [x-size/2, y-size*0.8], [x+size/2, y-size*0.8]], 
                                     closed=True, ec=line_color, fc=fill_color, lw=1.5, zorder=10))
        # Base with stylized hatching
        ax.plot([x-size*0.8, x+size*0.8], [y-size*0.8, y-size*0.8], color=line_color, lw=2)
        for i in np.linspace(x-size*0.6, x+size*0.6, 4):
             ax.plot([i, i-0.15], [y-size*0.8, y-size*1.1], color=line_color, lw=1, alpha=0.5)
            
    elif type == 'Roller':
        # Triangle
        ax.add_patch(patches.Polygon([[x, y], [x-size/2, y-size*0.6], [x+size/2, y-size*0.6]], 
                                     closed=True, ec=line_color, fc=fill_color, lw=1.5, zorder=10))
        # Modern Wheels (Circles with dots)
        r = size * 0.12
        for offset in [-size/3.5, size/3.5]:
            c = patches.Circle((x+offset, y-size*0.6-r), r, ec=line_color, fc=C_BG, lw=1.5)
            ax.add_patch(c)
            ax.add_patch(patches.Circle((x+offset, y-size*0.6-r), r/3, fc=line_color)) # Center dot

        # Ground
        ground_y = y - size*0.6 - 2*r - 0.05
        ax.plot([x-size*0.9, x+size*0.9], [ground_y, ground_y], color=line_color, lw=2)

    elif type == 'Fixed':
        # Thick stylized wall
        ax.add_patch(patches.Rectangle((x-size*0.1, y-size*1.2), size*0.2, size*2.4, fc=fill_color, ec=line_color, lw=2))
        # Hatching
        h_step = size/2.5
        direction = -1 if x < 0.1 else 1
        for i in np.arange(y-size, y+size, h_step):
            ax.plot([x, x + 0.3*direction], [i, i+0.3], color=line_color, lw=1, alpha=0.5)

# =========================================
# 1. MODERN DIAGRAM PLOTTING
# =========================================
def draw_diagrams(df, spans, sup_df, loads, unit_force, unit_len, probe_x=None):
    
    # Set dark theme aesthetic within Matplotlib
    plt.rcParams.update({
        'axes.facecolor': C_BG,
        'figure.facecolor': C_BG,
        'text.color': C_TEXT,
        'axes.labelcolor': C_TEXT,
        'xtick.color': C_TEXT,
        'ytick.color': C_TEXT,
        'grid.color': C_GRID,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Roboto', 'Arial', 'sans-serif']
    })
    
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 12), sharex=True, 
                                        gridspec_kw={'height_ratios': [1, 2, 2]})
    fig.patch.set_facecolor(C_BG)
    plt.subplots_adjust(hspace=0.15)
    
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    margin = total_len * 0.05
    x_lims = [-margin, total_len + margin]
    
    # === AX0: STRUCTURAL MODEL (Modern Blueprint Style) ===
    ax0.set_title("STRUCTURAL MODEL & LOADING", loc='left', fontweight='bold', fontsize=10, color=C_TEXT, pad=20)
    ax0.set_ylim(-1.5, 2.5)
    ax0.axis('off')
    
    # Beam Line with Glow
    beam_line, = ax0.plot([0, total_len], [0, 0], color=C_BEAM, linewidth=5, solid_capstyle='butt', zorder=5)
    
    # Supports
    for _, row in sup_df.iterrows():
        draw_support_symbol(ax0, cum_spans[int(row['id'])], 0, row['type'], size=0.8)

    # Loads (High Contrast)
    for l in loads:
        x_s = cum_spans[int(l['span_idx'])]
        if l['type'] == 'P':
            px = x_s + l['x']
            ax0.arrow(px, 2.0, 0, -1.6, head_width=0.25, fc=C_LOAD, ec=C_LOAD, width=0.04, zorder=20)
            ax0.text(px, 2.2, f"P={l['P']}", ha='center', fontweight='bold', color=C_BG,
                     bbox=dict(boxstyle="square,pad=0.3", fc=C_LOAD, ec='none'))
        elif l['type'] == 'U':
            x_e = cum_spans[int(l['span_idx'])+1]
            # Gradient-like block
            rect = patches.Rectangle((x_s, 0.1), x_e-x_s, 0.9, facecolor=C_LOAD, alpha=0.2, edgecolor=C_LOAD, lw=2, linestyle='--')
            ax0.add_patch(rect)
            # Stylized arrows
            for xa in np.linspace(x_s+0.1, x_e-0.1, max(3, int((x_e-x_s)*2))):
                 ax0.arrow(xa, 0.8, 0, -0.5, head_width=0.1, fc=C_LOAD, ec=C_LOAD, alpha=0.8)
            ax0.text((x_s+x_e)/2, 1.1, f"w={l['w']}", ha='center', fontweight='bold', color=C_LOAD)

    # === AX1: SHEAR (Modern Cyan) ===
    ax1.set_ylabel(f"SHEAR ({unit_force})", fontweight='bold', fontsize=9)
    line_v, = ax1.plot(df['x'], df['shear'], color=C_SHEAR, linewidth=2.5)
    add_glow_effect(line_v)
    ax1.fill_between(df['x'], df['shear'], 0, color=C_SHEAR, alpha=0.15)
    ax1.axhline(0, color=C_TEXT, linewidth=0.5, alpha=0.5)
    ax1.grid(True, which='major', linestyle='-', linewidth=0.5)
    for spine in ['top', 'right', 'bottom']: ax1.spines[spine].set_visible(False)
    ax1.spines['left'].set_color(C_GRID)

    # === AX2: MOMENT (Modern Indigo) ===
    ax2.set_ylabel(f"MOMENT ({unit_force}-{unit_len})", fontweight='bold', fontsize=9)
    ax2.set_xlabel(f"DISTANCE ({unit_len})", fontweight='bold', fontsize=9)
    line_m, = ax2.plot(df['x'], df['moment'], color=C_MOMENT, linewidth=2.5)
    add_glow_effect(line_m)
    ax2.fill_between(df['x'], df['moment'], 0, color=C_MOMENT, alpha=0.15)
    ax2.axhline(0, color=C_TEXT, linewidth=0.5, alpha=0.5)
    ax2.grid(True, which='major', linestyle='-', linewidth=0.5)
    for spine in ['top', 'right']: ax2.spines[spine].set_visible(False)
    ax2.spines['left'].set_color(C_GRID)
    ax2.spines['bottom'].set_color(C_GRID)

    # === INTERACTIVE PROBE (Bright Yellow) ===
    if probe_x is not None:
        for ax in [ax0, ax1, ax2]:
            ax.axvline(probe_x, color=C_PROBE, linestyle='--', linewidth=1, alpha=0.8, zorder=30)
            
        idx = (df['x'] - probe_x).abs().idxmin()
        val_v = df.loc[idx, 'shear']
        val_m = df.loc[idx, 'moment']
        
        # Probe Indicators
        ax0.text(probe_x, -1.2, f"x = {probe_x:.2f}", ha='center', fontsize=9, fontweight='bold', color=C_PROBE,
                 bbox=dict(boxstyle="round,pad=0.2", fc=C_BG, ec=C_PROBE, lw=1.5))

        for ax, val, unit, offset in [(ax1, val_v, unit_force, 15), (ax2, val_m, f"{unit_force}-{unit_len}", 15)]:
            ax.plot(probe_x, val, 'o', color=C_BG, markeredgecolor=C_PROBE, markeredgewidth=2, markersize=8, zorder=35)
            ax.annotate(f"{val:,.0f}", xy=(probe_x, val), xytext=(5, offset), textcoords='offset points',
                        color=C_BG, fontweight='bold', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc=C_PROBE, ec='none', alpha=0.9), zorder=40)

    for ax in [ax0, ax1, ax2]: ax.set_xlim(x_lims)
    st.pyplot(fig)

# =========================================
# 2. SENIOR ENGINEER RC DESIGN LOGIC
# =========================================
def render_design_results(df, params, spans, span_props, sup_df):
    """
    Performs detailed RC design checks (Capacity vs Demand) for Senior Engineers.
    Follows ACI 318 / EIT Strength Design Method.
    """
    fc, fy = params['fc'], params['fy']
    phi_b, phi_v = 0.90, 0.75 # ACI 318-19 Shear phi is 0.75
    Es = 2.04e6 # ksc (Steel Modulus)

    results_data = []
    cum_dist = 0.0
    
    st.markdown("### 🧠 Detailed Engineer's Calculation Sheet (Strength Design)")
    st.caption(f"**Design Parameters:** f'c = {fc} ksc, fy = {fy} ksc, φ_flexure = {phi_b}, φ_shear = {phi_v}")

    for i, L in enumerate(spans):
        # 1. Get Inputs & Forces
        prop = span_props[i]
        b, h, cv = prop['b'], prop['h'], prop['cv']
        db_main, db_stir = prop['main_bar_dia'], prop['stirrup_dia']
        
        mask = (df['x'] >= cum_dist) & (df['x'] <= cum_dist + L)
        span_data = df[mask]
        # Get critical forces (convert kg-m to kg-cm for calculation)
        Mu_pos = max(0, span_data['moment'].max()) * 100
        # Note: A full design would check Mu_neg at supports too. 
        # For dashboard simplicity, we focus on midspan positive moment for main bars.
        Vu_max = span_data['shear'].abs().max()

        # 2. Section Properties & Effective Depth
        Ab_main = get_bar_area(db_main)
        Ab_stir = get_bar_area(db_stir)
        # d = h - cover - stirrup - main/2 (Assume 1 layer)
        d = h - cv - (db_stir/10) - (db_main/20)
        
        # 3. Flexural Design (Capacity Check Mode)
        # Calculate required As first to estimate number of bars
        # Simplified Mu = phi * As * fy * 0.9d
        As_approx = Mu_pos / (phi_b * fy * 0.9 * d) if Mu_pos > 0 else 0
        num_bars = max(2, math.ceil(As_approx / Ab_main)) if As_approx > 0 else 2
        As_provided = num_bars * Ab_main
        
        # --- SENIOR CHECK 1: Spacing & Rho ---
        # Required width for 'num_bars' in one layer
        # width = 2*cv + 2*db_stir + n*db_main + (n-1)*spacing_min
        s_min_clear = max(2.5, db_main/10) # Max of 2.5cm or bar diameter
        req_width = 2*cv + 2*(db_stir/10) + num_bars*(db_main/10) + (num_bars-1)*s_min_clear
        
        fit_status = "✅ OK" if b >= req_width else "❌ Congested"
        
        # Reinforcement Ratio Checks
        rho = As_provided / (b * d)
        rho_min, rho_max = get_rho_limits(fc, fy)
        
        rho_status = "✅ OK"
        if rho < rho_min: rho_status = "⚠️ < Min (Use As_min)"
        elif rho > rho_max: rho_status = "❌ > Max (Brittle!)"
        
        # --- SENIOR CHECK 2: Moment Capacity (phi*Mn) ---
        # Depth of Stress Block (a) = (As * fy) / (0.85 * fc * b)
        a = (As_provided * fy) / (0.85 * fc * b)
        # Nominal Moment Mn = As * fy * (d - a/2)
        Mn = As_provided * fy * (d - a/2) # kg-cm
        phi_Mn = phi_b * Mn
        
        # D/C Ratio
        dc_flex = Mu_pos / phi_Mn if phi_Mn > 0 else 999.0
        flex_status = "✅ PASS" if dc_flex <= 1.0 else "❌ FAIL"
        if Mu_pos == 0: flex_status, dc_flex = "-", 0

        # 4. Shear Design (Detailed Stirrup Spacing)
        # Vc = 0.53 * sqrt(fc) * b * d (Simplified ACI metric) -> Result in kg
        # Note: ACI 318-19 uses slightly different formula involving nu_s and rho_w. Sticking to classic for simplicity here.
        Vc = 0.53 * math.sqrt(fc) * b * d 
        phi_Vc = phi_v * Vc
        
        # Determine stirrup requirement
        stirrup_txt = "-"
        s_provided = 0
        shear_status = "Concrete OK"
        dc_shear = Vu_max / phi_Vc # Base D/C on concrete only first

        if Vu_max > phi_Vc / 2:
            # Stirrups needed (at least min)
            Av = 2 * Ab_stir # 2 legs
            
            if Vu_max > phi_Vc:
                # Design for Vs
                Vs_req = (Vu_max / phi_v) - Vc
                # Check max Vs allowed (Vs <= 2.1 * sqrt(fc) * b * d)
                Vs_max_limit = 2.1 * math.sqrt(fc) * b * d
                if Vs_req > Vs_max_limit:
                     shear_status = "❌ Section Too Small (Vs > Vs_max)"
                     s_provided = 5 # Theoretical
                else:
                     # s = (Av * fy * d) / Vs
                     s_calc = (Av * fy * d) / Vs_req
                     shear_status = "Designed"
            else:
                # Min stirrups apply (Vu between phiVc/2 and phiVc)
                # s_min_req = (Av * fy) / (3.5 * bw) or (Av*fy)/(0.2*sqrt(fc)*bw) -> Simplified max spacing controls usually
                s_calc = 999.0 # Let max spacing control
                shear_status = "Min Reinf."

            # Spacing Limits (ACI)
            # If Vs is high, tighter spacing
            s_max_limit = d / 2
            if Vu_max > phi_Vc + (1.06 * math.sqrt(fc) * b * d): # High shear zone
                s_max_limit = d / 4
                
            s_max_limit = min(s_max_limit, 60.0) # Absolute max 60cm (or 30cm depending on code ver)
            
            # Final Spacing (Round down to nearest cm, min 5cm practical)
            s_final = math.floor(min(s_calc, s_max_limit))
            s_provided = max(5, int(s_final))
            stirrup_txt = f"RB{db_stir}@{s_provided}c/c"
            
            # Recalculate Capacity with provided spacing
            Vs_prov = (Av * fy * d) / s_provided
            phi_Vn = phi_v * (Vc + Vs_prov)
            dc_shear = Vu_max / phi_Vn

        # Prepare Data Row
        results_data.append({
            "Span": f"Span {i+1}",
            "Section": f"{b:.0f}x{h:.0f}",
            "d_eff (cm)": f"{d:.1f}",
            
            # Flexure Data
            "Mu+ (kg-m)": f"{Mu_pos/100:.1f}",
            "As Prov.": f"{num_bars}-DB{db_main} ({As_provided:.1f} cm²)",
            "ρ Check": rho_status,
            "φMn (kg-m)": f"{phi_Mn/100:.1f}",
            "D/C (Flex)": f"{dc_flex:.2f} ({flex_status})",
            "Fit Check": fit_status,

            # Shear Data
            "Vu (kg)": f"{Vu_max:.1f}",
            "φVc (kg)": f"{phi_Vc:.1f}",
            "Stirrups": stirrup_txt,
            "D/C (Shear)": f"{dc_shear:.2f}",
            "Shear Status": shear_status
        })
        
        cum_dist += L

    # Display Professional Table
    df_results = pd.DataFrame(results_data)
    
    # Function to style the dataframe columns for Pass/Fail
    def highlight_cols(val):
        if isinstance(val, str):
            if '❌' in val or 'FAIL' in val or 'Congested' in val or '> Max' in val:
                return 'color: #DC2626; font-weight: bold;'
            elif '⚠️' in val:
                 return 'color: #D97706; font-weight: bold;'
            elif '✅' in val or 'PASS' in val:
                return 'color: #059669; font-weight: bold;'
        # Highlight high D/C ratios
        try:
            val_fl = float(val.split(' ')[0])
            if val_fl > 1.0: return 'color: #DC2626; font-weight: bold;'
            if val_fl > 0.9: return 'color: #D97706; font-weight: bold;'
        except: pass
        return ''

    styled_df = df_results.style.map(highlight_cols, 
        subset=['ρ Check', 'D/C (Flex)', 'Fit Check', 'Shear Status', 'D/C (Shear)']) \
        .format(precision=2)

    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)

    # Add Engineering Notes Footer
    st.info("""
    **💡 Senior Engineer Notes:**
    1.  **D/C Ratio (Demand/Capacity):** Values > 1.0 indicate failure. Ideal range is 0.8 - 0.95 for economic design.
    2.  **ρ Check:** Ensures the beam is tension-controlled (ductile failure). '⚠️ < Min' means use standard minimum reinforcement.
    3.  **Fit Check:** Crude check if bars fit in a single layer with standard spacing. Does not account for splices or multiple layers.
    4.  **Shear:** Stirrup spacing calculated based on $V_u$ demand, limited by code maximums ($d/2$ or $d/4$).
    """)
