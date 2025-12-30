import numpy as np
import math
import re

def parse_bars(bar_str):
    """Parses '2-DB12' into (2, 12). Returns None if invalid."""
    try:
        if not bar_str or "Over" in bar_str: return None
        match = re.search(r'(\d+)-DB(\d+)', bar_str)
        if match:
            return int(match.group(1)), int(match.group(2))
    except:
        pass
    return None

def calculate_flexure_sdm(Mu, type_str, params):
    """Design flexural reinforcement (SDM) with detailed logging."""
    fc = params['fc']
    fy = params['fy']
    b = params['b']
    h = params['h']
    d = h - params['cv']
    db_select = params['db_main']
    
    # Factors (ACI 318 / EIT)
    phi_b = 0.90
    beta1 = 0.85 if fc <= 280 else max(0.65, 0.85 - 0.05*(fc-280)/70)
    
    is_metric = 'Metric' in params['unit']
    
    # Unit Conversion for Calculation
    if is_metric:
        M_design = abs(Mu) * 100 # kg-m -> kg-cm
        fc_c, fy_c = fc, fy
        b_c, d_c = b, d
        # ACI/EIT Metric Rho Min
        rho_min = max(14/fy_c, 0.25*np.sqrt(fc_c)/fy_c) 
    else: 
        M_design = abs(Mu) * 1e6 # kN-m -> N-mm
        fc_c, fy_c = fc, fy
        b_c, d_c = b*10, d*10
        rho_min = max(1.4/fy_c, 0.25*np.sqrt(fc_c)/fy_c)

    # Balanced Ratio
    bal_const = 6120 if is_metric else 6000
    rho_bal = 0.85 * beta1 * (fc_c/fy_c) * (bal_const/(bal_const+fy_c))
    rho_max = 0.75 * rho_bal 

    rho = 999
    Rn = 0
    try:
        Rn = M_design / (phi_b * b_c * d_c**2)
        term = 1 - 2*Rn/(0.85*fc_c)
        if term >= 0:
            rho = (0.85 * fc_c / fy_c) * (1 - np.sqrt(term))
    except:
        pass
        
    As_calc = rho * b_c * d_c
    As_min = rho_min * b_c * d_c
    
    status = "✅ OK"
    control_As = As_calc
    note = "(Calculated)"
    
    if rho == 999 or rho > rho_max:
        status = "❌ Over Reinforced"
        note = "(Section too small, Rho > Rho_max)"
        control_As = As_calc
    elif rho < rho_min:
        control_As = As_min
        status = "⚠️ Min Steel Governs"
        note = "(Minimum Required)"

    # Bar Selection
    select_str = ""
    As_provided = 0
    if "Over" not in status:
        unit_area = 3.1416 * (db_select/10)**2 / 4 if is_metric else 3.1416 * db_select**2 / 4
        num = math.ceil(control_As / unit_area)
        
        # Check Layer Width
        width_avail = b_c - 2*params['cv'] if is_metric else b_c - 2*params['cv']*10
        bar_dia = db_select/10 if is_metric else db_select
        # Simple check for 1 layer spacing (approx 2.5cm gap or db)
        spacing_req = max(2.5, bar_dia)
        max_bars_layer = int((width_avail + spacing_req) / (bar_dia + spacing_req))
        
        if num > max_bars_layer and num > 2:
             # Just a warning logic, theoretically can do 2 layers but kept simple here
             pass 
        
        select_str = f"{int(num)}-DB{db_select}"
        As_provided = num * unit_area
            
    return_As = control_As if is_metric else control_As/100
    
    # --- Engineering Calculation Log ---
    u_len = "cm" if is_metric else "mm"
    u_area = "cm²" if is_metric else "mm²"
    
    calc_log = [
        f"**Design Parameters ({type_str})**",
        f"- $M_u = {abs(Mu):.2f}$ {'kg-m' if is_metric else 'kN-m'}",
        f"- $d = {d_c:.1f}$ {u_len}, $b = {b_c:.1f}$ {u_len}",
        f"- $f'_c = {fc:.1f}$, $f_y = {fy:.0f}$",
        f"---",
        f"**1. Flexural Strength Check**",
        f"- $R_n = \\frac{{M_u}}{{\phi b d^2}} = {Rn:.4f}$",
        f"- $\\rho_{{req}} = {rho:.5f}$",
        f"- $\\rho_{{min}} = {rho_min:.5f}$",
        f"- $\\rho_{{max}} (0.75\\rho_b) = {rho_max:.5f}$",
        f"- Check: {'OK' if rho <= rho_max else 'NG (Section too small)'}",
        f"---",
        f"**2. Reinforcement Area ($A_s$)**",
        f"- $A_{{s,calc}} = {As_calc:.2f}$ {u_area}",
        f"- $A_{{s,min}} = {As_min:.2f}$ {u_area}",
        f"- **Control $A_s$ = {control_As:.2f} {u_area}** {note}",
        f"---",
        f"**3. Selected Reinforcement**",
        f"- Use: **{select_str}**",
        f"- $A_{{s,prov}} = {As_provided:.2f}$ {u_area} (> Required)"
    ]

    return { 
        "Type": type_str, "Mu": abs(Mu), "As_req": return_As, 
        "Status": status, "Bars": select_str, "Log": calc_log 
    }

def calculate_shear_capacity(Vu, params):
    """Shear design with detailed logging."""
    fc = params['fc']
    fy_stir = params['fys']
    b = params['b']
    d = params['h'] - params['cv']
    db_stir = params['db_stirrup']
    step = params.get('s_step', 2.5)
    
    spacing_txt = ""
    v_cap_display = 0
    calc_log = []
    
    is_metric = 'Metric' in params['unit']
    
    if is_metric:
        # ACI Simplified Vc
        vc = 0.53 * np.sqrt(fc) * b * d 
        phi_vc = 0.85 * vc 
        vu_val = abs(Vu)
        v_cap_display = phi_vc
        
        calc_log.append(f"**Shear Design Parameters**")
        calc_log.append(f"- $V_u = {vu_val:.2f}$ kg")
        calc_log.append(f"- $\phi = 0.85$ (Shear)")
        calc_log.append(f"---")
        
        calc_log.append(f"**1. Concrete Capacity**")
        calc_log.append(f"- $V_c = 0.53\sqrt{{f'c}} bd = {vc:.2f}$ kg")
        calc_log.append(f"- $\phi V_c = {phi_vc:.2f}$ kg")
        calc_log.append(f"- $0.5 \phi V_c = {phi_vc/2:.2f}$ kg")
        
        av = 2 * (3.1416 * (db_stir/10)**2 / 4)
        
        if vu_val > phi_vc:
            vs = (vu_val - phi_vc) / 0.85
            s_req = (av * fy_stir * d) / vs
            
            s_max = d/2
            s_use = min(s_req, s_max, 30.0)
            s_use = math.floor(s_use / step) * step
            
            calc_log.append(f"---")
            calc_log.append(f"**2. Stirrup Calculation**")
            calc_log.append(f"- Condition: $V_u > \phi V_c$ (Stirrups Required)")
            calc_log.append(f"- $V_s = (V_u - \phi V_c)/\phi = {vs:.2f}$ kg")
            calc_log.append(f"- $A_v (2 legs) = {av:.2f}$ cm²")
            calc_log.append(f"- $s_{{req}} = \\frac{{A_v f_y d}}{{V_s}} = {s_req:.2f}$ cm")
            calc_log.append(f"- $s_{{max}} (d/2) = {s_max:.2f}$ cm")
            
            if s_use < 5.0: 
                spacing_txt = f"RB{db_stir} - Fail/Close"
                calc_log.append(f"- **Result:** Spacing too close! Increase Section.")
            else: 
                spacing_txt = f"RB{db_stir}@{s_use:.0f}cm"
                calc_log.append(f"- **Select:** {spacing_txt}")

        elif vu_val > phi_vc/2:
            s_max = d/2
            s_use = min(s_max, 30.0)
            s_use = math.floor(s_use / step) * step
            spacing_txt = f"RB{db_stir}@{s_use:.0f}cm (Min)"
            
            calc_log.append(f"---")
            calc_log.append(f"**2. Stirrup Calculation**")
            calc_log.append(f"- Condition: $0.5\phi V_c < V_u \le \phi V_c$")
            calc_log.append(f"- Use Minimum Shear Reinforcement")
            calc_log.append(f"- **Select:** {spacing_txt}")
        else:
            spacing_txt = "None Req."
            calc_log.append(f"---")
            calc_log.append(f"**2. Stirrup Calculation**")
            calc_log.append(f"- Condition: $V_u \le 0.5\phi V_c$")
            calc_log.append(f"- Theoretical: No stirrups required")
            
    else:
        # SI Placeholder
        vc = 0.17 * np.sqrt(fc) * b*10 * d*10 
        phi_vc = 0.75 * (vc / 1000) 
        vu_val = abs(Vu)
        v_cap_display = phi_vc
        if vu_val > phi_vc: spacing_txt = f"RB{db_stir} (Calc SI)"
        else: spacing_txt = "Min / None"
        calc_log.append("SI Units Simplified Calculation")
        
    return abs(Vu), v_cap_display, spacing_txt, calc_log
