import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# --- HELPER: Rebar Area ---
def get_bar_area(dia_mm):
    """Return area of rebar in cm2"""
    return (math.pi * (dia_mm/10)**2) / 4

def get_bar_text(n, dia):
    return f"{n}-DB{dia}" if dia >= 10 else f"{n}-RB{dia}"

# --- 1. DRAW DIAGRAMS (SFD/BMD) ---
def draw_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    """
    Plot Shear Force and Bending Moment Diagrams with professional formatting.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # X axis
    x = df['x']
    
    # --- Shear Diagram ---
    ax1.plot(x, df['shear'], color='#E65100', linewidth=2, label='Shear Force (V)')
    ax1.fill_between(x, df['shear'], 0, color='#FFE0B2', alpha=0.5)
    ax1.set_ylabel(f"Shear Force ({unit_force})", fontsize=10, fontweight='bold')
    ax1.set_title("Shear Force Diagram (SFD)", fontsize=12, fontweight='bold', color='#E65100')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.axhline(0, color='black', linewidth=1)
    
    # Annotate Max Shear
    v_max = df['shear'].max()
    v_min = df['shear'].min()
    ax1.annotate(f"{v_max:.2f}", xy=(x[df['shear'].idxmax()], v_max), xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax1.annotate(f"{v_min:.2f}", xy=(x[df['shear'].idxmin()], v_min), xytext=(5, -15), textcoords='offset points', fontsize=9)

    # --- Moment Diagram ---
    # Note: Invert Y for engineering convention (Sagging Positive down? usually plotted positive up in software, but negative moment is top tension)
    # Let's stick to standard math plot: Positive Up. User knows + is Sagging (Bottom Steel), - is Hogging (Top Steel).
    
    ax2.plot(x, df['moment'], color='#1565C0', linewidth=2, label='Bending Moment (M)')
    ax2.fill_between(x, df['moment'], 0, color='#BBDEFB', alpha=0.5)
    ax2.set_ylabel(f"Moment ({unit_force}-{unit_len})", fontsize=10, fontweight='bold')
    ax2.set_xlabel(f"Distance ({unit_len})", fontsize=10, fontweight='bold')
    ax2.set_title("Bending Moment Diagram (BMD)", fontsize=12, fontweight='bold', color='#1565C0')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axhline(0, color='black', linewidth=1)
    
    # Annotate Supports
    for xc in np.cumsum(spans):
        ax2.axvline(xc, color='gray', linestyle=':', alpha=0.7)
        ax1.axvline(xc, color='gray', linestyle=':', alpha=0.7)

    st.pyplot(fig)

# --- 2. RENDER DESIGN CALCULATION ---
def render_design_results(df, params, spans, span_props, sup_df):
    """
    Perform RC Design for each span based on Analysis Results (df) and User Inputs (span_props).
    """
    
    # Material Properties
    fc = params['fc']  # ksc
    fy = params['fy']  # ksc
    # Strength Reduction Factors (ACI 318 / EIT)
    phi_b = 0.90  # Flexure
    phi_v = 0.85  # Shear
    
    results = []
    
    cum_dist = 0.0
    
    st.markdown("### 📋 Detailed Design Calculation (Working Stress / Ultimate Strength)")
    st.caption("*Assumption: Input Loads are Ultimate Loads ($M_u, V_u$). If not, please apply load factors manually.*")

    for i, L in enumerate(spans):
        # 1. Get Span Properties
        prop = span_props[i]
        b = prop['b']      # cm
        h = prop['h']      # cm
        cv = prop['cv']    # cm
        db_main = prop['main_bar_dia']  # mm
        db_stir = prop['stirrup_dia']   # mm
        
        # Effective Depth (d)
        # d approx = h - cover - stirrup - main/2
        d = h - cv - (db_stir/10) - (db_main/20)
        
        # 2. Get Internal Forces for this span
        # Filter dataframe for range [cum_dist, cum_dist + L]
        # We add small buffer to avoid capturing neighbor span peaks at supports
        mask = (df['x'] >= cum_dist) & (df['x'] <= cum_dist + L)
        span_data = df[mask]
        
        # Design Moments
        mu_pos = span_data['moment'].max() # Max Sagging (+Moment) -> Bottom Steel
        mu_neg = span_data['moment'].min() # Max Hogging (-Moment) -> Top Steel (Usually at supports)
        
        # Design Shear
        vu_max = span_data['shear'].abs().max() # Critical Shear
        
        # --- FLEXURE DESIGN (Positive Moment / Bottom Steel) ---
        # Only design for Positive Moment in mid-span for simplicity presentation
        # Real design would do Top Steel at supports too.
        
        req_As_pos, note_pos = _design_flexure(mu_pos, b, d, fc, fy, phi_b)
        
        # Calculate Number of Bars
        area_one_bar = get_bar_area(db_main)
        if req_As_pos > 0:
            num_bars = math.ceil(req_As_pos / area_one_bar)
            # Min bars = 2
            num_bars = max(num_bars, 2)
            provided_As = num_bars * area_one_bar
            
            # Check Spacing (b must accommodate n bars)
            # clear_space = b - 2*cover - 2*stirrup - n*db
            req_width = 2*cv + 2*(db_stir/10) + num_bars*(db_main/10) + (num_bars-1)*2.5 # assume 2.5cm gap
            
            if req_width > b:
                spacing_status = "❌ Too Tight! Increase b"
            else:
                spacing_status = "✅ OK"
                
            txt_main = f"{num_bars}-DB{db_main}"
        else:
            txt_main = "Min Reinf."
            provided_As = 0
            spacing_status = "-"

        # --- SHEAR DESIGN (Stirrups) ---
        # Vc = 0.53 * sqrt(fc) * b * d (ACI Metric) -> unit kg/cm2
        # fc is in ksc. 
        vc_stress = 0.53 * math.sqrt(fc) # kg/cm2
        Vc = vc_stress * b * d # kg
        phi_Vc = phi_v * Vc
        
        # Vs required
        # Vu <= phi(Vc + Vs)  =>  Vs >= (Vu/phi) - Vc
        if vu_max > phi_Vc / 2:
            # Need Stirrups (at least min)
            if vu_max > phi_Vc:
                Vs_req = (vu_max / phi_v) - Vc
                # Spacing s = (Av * fy * d) / Vs
                Av = 2 * get_bar_area(db_stir) # 2 legs
                s_calc = (Av * fy * d) / Vs_req
                
                # Max Spacing Limits
                s_max = d / 2
                s_final = min(s_calc, s_max, 30.0) # Cap at 30cm
                s_final = math.floor(s_final) # Round down to integer
                if s_final < 5: s_final = 5 # Min practical spacing
                
                txt_stirrup = f"RB{db_stir} @ {int(s_final)} cm"
                status_shear = "Designed"
            else:
                # Min Stirrups
                s_max = d / 2
                s_final = min(s_max, 30.0)
                txt_stirrup = f"RB{db_stir} @ {int(s_final)} cm (Min)"
                status_shear = "Min Reinf."
        else:
            txt_stirrup = "Theoretical None"
            status_shear = "Concrete OK"
            
        
        # Append Result
        results.append({
            "Span": f"Span {i+1}",
            "Size (cm)": f"{b:.0f} x {h:.0f}",
            "Mu+ (kg-m)": f"{mu_pos:.2f}",
            "As Req (cm2)": f"{req_As_pos:.2f}",
            "Bottom Bars": f"**{txt_main}**",
            "Fit?": spacing_status,
            "Vu (kg)": f"{vu_max:.2f}",
            "Stirrups": f"**{txt_stirrup}**"
        })
        
        cum_dist += L

    # Display Table
    st.table(pd.DataFrame(results))
    
    # Legend
    st.info("""
    **Note:** 1. **Bottom Bars**: Main reinforcement for positive moment (mid-span). Top bars at supports should be checked separately.
    2. **Fit?**: Checks if bars fit in width `b` with standard spacing.
    3. **Stirrups**: Calculated for Vertical Shear ($V_u$).
    """)

def _design_flexure(Mu, b, d, fc, fy, phi):
    """
    Return Required As (cm2) for Singly Reinforced Beam
    Mu in kg-m
    """
    if Mu <= 0: return 0, "Compression/Min"
    
    Mu_kgcm = Mu * 100
    
    # Iterative or Formula Design (Rn)
    # Mn_req = Mu / phi
    Mn_req = Mu_kgcm / phi
    
    # Rn = Mn / (b * d^2)
    Rn = Mn_req / (b * d**2) # kg/cm2
    
    # rho = (0.85 fc / fy) * (1 - sqrt(1 - 2*Rn / (0.85*fc)))
    term = 1 - (2 * Rn) / (0.85 * fc)
    
    if term < 0:
        return 0, "Section too small! (Concrete Crush)"
    
    rho = (0.85 * fc / fy) * (1 - math.sqrt(term))
    
    # Check Min Steel
    # rho_min = 14/fy or 0.8 sqrt(fc)/fy
    rho_min = max(14/fy, 0.8*math.sqrt(fc)/fy)
    
    # Check Max Steel (approx 0.75 rho_b, simplified to 0.025 for now)
    rho_max = 0.025 # Simplified limit
    
    final_rho = max(rho, rho_min)
    
    As_req = final_rho * b * d
    
    # Warning if section too small
    if rho > rho_max:
        return As_req, "Over Reinforced!"
        
    return As_req, "OK"
