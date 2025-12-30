import numpy as np
import math
import re

def parse_bars(bar_str):
    """แปลง string เช่น '4-DB20' เป็นตัวเลข (4, 20)"""
    try:
        if not bar_str or "Over" in bar_str: return None
        match = re.search(r'(\d+)-DB(\d+)', bar_str)
        if match:
            return int(match.group(1)), int(match.group(2))
    except:
        pass
    return None

def calculate_flexure_sdm(Mu, type_str, params):
    """คำนวณเหล็กรับแรงดัด พร้อมคืนค่ารายการคำนวณละเอียด"""
    fc = params['fc']
    fy = params['fy']
    b = params['b']
    h = params['h']
    d = h - params['cv']
    db_select = params['db_main']
    
    phi_b = 0.90
    beta1 = 0.85 if fc <= 280 else max(0.65, 0.85 - 0.05*(fc-280)/70)
    
    # Unit Conversion Setup
    is_metric = 'Metric' in params['unit']
    if is_metric:
        M_design = abs(Mu) * 100 # kg-m -> kg-cm
        fc_c, fy_c = fc, fy
        b_c, d_c = b, d
        rho_min_1 = 14/fy_c
        rho_min_2 = 0.25*np.sqrt(fc_c)/fy_c
        bal_const = 6120
    else: 
        M_design = abs(Mu) * 1e6 # kN-m -> N-mm
        fc_c, fy_c = fc, fy
        b_c, d_c = b*10, d*10
        rho_min_1 = 1.4/fy_c
        rho_min_2 = 0.25*np.sqrt(fc_c)/fy_c
        bal_const = 6000

    rho_min = max(rho_min_1, rho_min_2)
    rho_bal = 0.85 * beta1 * (fc_c/fy_c) * (bal_const/(bal_const+fy_c))
    rho_max = 0.75 * rho_bal 

    # Calculation
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
    
    note = ""
    if rho == 999 or rho > rho_max:
        status = "❌ Over Reinforced"
        note = "(Section too small)"
        control_As = As_calc
    elif rho < rho_min:
        control_As = As_min
        status = "⚠️ Min Steel Governs"
        note = "(Use As,min)"

    # Bar Selection
    select_str = ""
    As_provided = 0
    if "Over" not in status:
        unit_area = 3.1416 * (db_select/10)**2 / 4 if is_metric else 3.1416 * db_select**2 / 4
        num = math.ceil(control_As / unit_area)
        
        width_avail = b_c - 2*params['cv'] if is_metric else b_c - 2*params['cv']*10
        bar_dia = db_select/10 if is_metric else db_select
        max_bars = int(width_avail / (bar_dia * 2.5)) # Approximation spacing
        
        if num > max_bars + 2 and num > 2: # Check crowding roughly
            select_str = f"Too many DB{db_select}"
            status = "⚠️ Section Too Small (Crowded)"
        else:
            select_str = f"{int(num)}-DB{db_select}"
            As_provided = num * unit_area
            
    return_As = control_As if is_metric else control_As/100
    
    # Generate Detailed Calculation Steps
    calc_log = [
        f"**1. Design Parameters:**",
        f"- $M_u = {abs(Mu):.2f}$ {'kg-m' if is_metric else 'kN-m'}",
        f"- $d = h - cover = {h} - {params['cv']} = {d:.2f}$ cm",
        f"- $R_n = M_u / (\phi b d^2) = {Rn:.2f}$ (ksc/MPa)",
        f"**2. Reinforcement Ratio ($\mu$):**",
        f"- $\\rho_{{req}} = {rho:.5f}$",
        f"- $\\rho_{{min}} = {rho_min:.5f}$ | $\\rho_{{max}} = {rho_max:.5f}$",
        f"- Check: {'$\\rho < \\rho_{max}$ OK' if rho <= rho_max else '$\\rho > \\rho_{max}$ NG'}",
        f"**3. Steel Area ($A_s$):**",
        f"- $A_{{s,req}} = {As_calc:.2f}$ cm²",
        f"- $A_{{s,min}} = {As_min:.2f}$ cm²",
        f"- **Control $A_s$ = {return_As:.2f} cm²** {note}",
        f"**4. Selection:**",
        f"- Try DB{db_select}: Use **{select_str}** ($A_{{s,prov}}={As_provided:.2f}$ cm²)"
    ]

    return { 
        "Type": type_str, "Mu": abs(Mu), "As_req": return_As, 
        "Status": status, "Bars": select_str, "Log": calc_log 
    }

def calculate_shear_capacity(Vu, params):
    """คำนวณ Shear พร้อม Spacing Step"""
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
        vc = 0.53 * np.sqrt(fc) * b * d 
        phi_vc = 0.85 * vc 
        vu_val = abs(Vu)
        v_cap_display = phi_vc
        
        calc_log.append(f"$\phi V_c = 0.85(0.53\sqrt{{f'c}} bd) = {phi_vc:.2f}$ kg")
        
        if vu_val > phi_vc:
            vs = (vu_val - phi_vc) / 0.85
            av = 2 * (3.1416 * (db_stir/10)**2 / 4)
            s_req = (av * fy_stir * d) / vs
            s_max = d/2
            s_use = min(s_req, s_max, 30.0)
            s_use = math.floor(s_use / step) * step
            
            calc_log.append(f"$V_u ({vu_val:.0f}) > \phi V_c$ -> Stirrups req.")
            calc_log.append(f"$V_s = (V_u - \phi V_c)/\phi = {vs:.2f}$ kg")
            calc_log.append(f"$S_{{req}} = (A_v f_y d)/V_s = {s_req:.2f}$ cm")
            
            if s_use < 5.0: 
                spacing_txt = f"RB{db_stir} - Increase Section"
            else: 
                spacing_txt = f"RB{db_stir}@{s_use:.0f}cm"
        elif vu_val > phi_vc/2:
            s_max = d/2
            s_use = min(s_max, 30.0)
            s_use = math.floor(s_use / step) * step
            spacing_txt = f"RB{db_stir}@{s_use:.0f}cm (Min)"
            calc_log.append(f"$V_u < \phi V_c$ but $> 0.5\phi V_c$ -> Min Stirrups")
        else:
            spacing_txt = "None Req."
            calc_log.append(f"$V_u < 0.5\phi V_c$ -> No Stirrups theor.")
            
    else:
        # SI Simplified
        vc = 0.17 * np.sqrt(fc) * b*10 * d*10 
        phi_vc = 0.75 * (vc / 1000) 
        vu_val = abs(Vu)
        v_cap_display = phi_vc
        if vu_val > phi_vc: spacing_txt = f"RB{db_stir} (Calc SI)"
        else: spacing_txt = "Min / None"
        
    return abs(Vu), v_cap_display, spacing_txt, calc_log
