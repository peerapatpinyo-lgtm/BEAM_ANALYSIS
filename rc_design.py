import numpy as np
import math
import re

def parse_bars(bar_str):
    try:
        if not bar_str or "Over" in bar_str or "Too many" in bar_str: return None
        match = re.search(r'(\d+)-DB(\d+)', str(bar_str))
        if match:
            return int(match.group(1)), int(match.group(2))
    except:
        pass
    return None

def calculate_flexure_sdm(Mu, type_str, b_in, h_in, cv_in, params):
    # Force convert inputs to float to prevent formatting errors
    try:
        fc = float(params['fc'])
        fy = float(params['fy'])
        b = float(b_in)
        h = float(h_in)
        cv = float(cv_in)
        d = h - cv
        db_select = int(params['db_main'])
        Mu = float(Mu)
    except ValueError:
        return {"Type": type_str, "Mu": 0, "As_req": 0, "Status": "❌ Data Error", "Bars": "", "Log": ["Error converting inputs"]}

    phi_b = 0.90
    beta1 = 0.85 if fc <= 280 else max(0.65, 0.85 - 0.05*(fc-280)/70)
    
    is_metric = 'Metric' in params['unit']
    
    if is_metric:
        M_design = abs(Mu) * 100 # kg-m -> kg-cm
        fc_c, fy_c = fc, fy
        b_c, d_c = b, d
        rho_min = max(14/fy_c, 0.25*np.sqrt(fc_c)/fy_c) 
    else: 
        M_design = abs(Mu) * 1e6 # kN-m -> N-mm
        fc_c, fy_c = fc, fy
        b_c, d_c = b*10, d*10
        rho_min = max(1.4/fy_c, 0.25*np.sqrt(fc_c)/fy_c)

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
        note = "(Section too small > Rho_max)"
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
        
        width_avail = b_c - 2*cv if is_metric else b_c - 2*cv*10
        bar_dia = db_select/10 if is_metric else db_select
        spacing_req = max(2.5, bar_dia)
        max_bars_layer = int((width_avail + spacing_req) / (bar_dia + spacing_req))
        
        if num > max_bars_layer and num > 1:
             status = "⚠️ Crowded / Multi-layer needed"
        
        select_str = f"{int(num)}-DB{db_select}"
        As_provided = num * unit_area
            
    return_As = control_As if is_metric else control_As/100
    u_len = "cm" if is_metric else "mm"
    u_area = "cm²" if is_metric else "mm²"
    
    # Safe formatting
    calc_log = [
        f"**Design Parameters ({type_str})**",
        f"- Section: {b:.1f}x{h:.1f} {u_len}, d={d:.1f} {u_len}",
        f"- $M_u = {abs(Mu):.2f}$",
        f"---",
        f"**Checks**",
        f"- $\\rho_{{req}} = {rho:.4f}$",
        f"- $\\rho_{{max}} = {rho_max:.4f}$ ({'OK' if rho <= rho_max else 'NG'})",
        f"---",
        f"**Steel Area**",
        f"- $A_{{s,req}} = {control_As:.2f}$ {u_area} {note}",
        f"- Use: **{select_str}** ($A_{{s,prov}} = {As_provided:.2f}$)"
    ]

    return { 
        "Type": type_str, "Mu": abs(Mu), "As_req": return_As, 
        "Status": status, "Bars": select_str, "Log": calc_log 
    }

def calculate_shear_capacity(Vu, b_in, h_in, cv_in, params):
    try:
        fc = float(params['fc'])
        fy_stir = float(params['fys'])
        b = float(b_in)
        h = float(h_in)
        cv = float(cv_in)
        d = h - cv
        db_stir = int(params['db_stirrup'])
        step = float(params.get('s_step', 2.5))
        vu_val = abs(float(Vu))
    except:
        return 0, 0, "Error", ["Check Inputs"]
    
    spacing_txt = ""
    v_cap_display = 0
    calc_log = []
    
    is_metric = 'Metric' in params['unit']
    
    if is_metric:
        vc = 0.53 * np.sqrt(fc) * b * d 
        phi_vc = 0.85 * vc 
        v_cap_display = phi_vc
        
        calc_log.append(f"**Shear Check** ($d={d:.1f}$ cm)")
        calc_log.append(f"- $V_u = {vu_val:.2f}$ kg")
        calc_log.append(f"- $\phi V_c = {phi_vc:.2f}$ kg")
        
        av = 2 * (3.1416 * (db_stir/10)**2 / 4)
        
        if vu_val > phi_vc:
            vs = (vu_val - phi_vc) / 0.85
            s_req = (av * fy_stir * d) / vs
            s_max = d/2
            s_use = min(s_req, s_max, 30.0)
            try:
                s_use = math.floor(s_use / step) * step
            except: s_use = 10.0
            
            if s_use < 5.0: 
                spacing_txt = f"RB{db_stir} - Fail"
            else: 
                spacing_txt = f"RB{db_stir}@{s_use:.0f}cm"
            calc_log.append(f"- $V_u > \phi V_c$ -> **Use {spacing_txt}**")

        elif vu_val > phi_vc/2:
            s_use = math.floor(min(d/2, 30.0) / step) * step
            spacing_txt = f"RB{db_stir}@{s_use:.0f}cm (Min)"
            calc_log.append(f"- $V_u > 0.5\phi V_c$ -> **Use {spacing_txt}**")
        else:
            spacing_txt = "None Req."
            calc_log.append(f"- $V_u \le 0.5\phi V_c$ -> No stirrups req.")
    else:
        vc = 0.17 * np.sqrt(fc) * b*10 * d*10 
        phi_vc = 0.75 * (vc / 1000) 
        v_cap_display = phi_vc
        spacing_txt = "Check SI Manual"
        
    return vu_val, v_cap_display, spacing_txt, calc_log
