import numpy as np
import math
import re

def parse_bars(bar_str):
    """ Extract number and size from '4-DB20' """
    try:
        if not bar_str or "Over" in bar_str: return None
        match = re.search(r'(\d+)-DB(\d+)', bar_str)
        if match:
            return int(match.group(1)), int(match.group(2))
    except:
        pass
    return None

def calculate_flexure_sdm(Mu, type_str, params):
    # (Logic เดิม คงไว้ แต่ตรวจสอบให้มั่นใจว่า return ครบ)
    fc = params['fc']
    fy = params['fy']
    b = params['b']
    h = params['h']
    d = h - params['cv']
    db_select = params['db_main']
    
    phi_b = 0.90
    beta1 = 0.85 if fc <= 280 else max(0.65, 0.85 - 0.05*(fc-280)/70)
    
    if 'Metric' in params['unit']:
        M_design = abs(Mu) * 100 
        fc_calc, fy_calc = fc, fy
        b_calc, d_calc = b, d
        rho_min = max(14/fy_calc, 0.25*np.sqrt(fc_calc)/fy_calc) 
    else: 
        M_design = abs(Mu) * 1e6 
        fc_calc, fy_calc = fc, fy
        b_calc, d_calc = b*10, d*10
        rho_min = max(1.4/fy_calc, 0.25*np.sqrt(fc_calc)/fy_calc)

    try:
        Rn = M_design / (phi_b * b_calc * d_calc**2)
        term = 1 - 2*Rn/(0.85*fc_calc)
        if term < 0: rho = 999 
        else: rho = (0.85 * fc_calc / fy_calc) * (1 - np.sqrt(term))
    except:
        rho = 999
        
    bal_const = 6120 if 'Metric' in params['unit'] else 6000
    rho_bal = 0.85 * beta1 * (fc_calc/fy_calc) * (bal_const/(bal_const+fy_calc))
    rho_max = 0.75 * rho_bal 
    
    As_req = rho * b_calc * d_calc
    As_min = rho_min * b_calc * d_calc
    
    status = "✅ OK"
    final_As = As_req
    
    if rho == 999 or rho > rho_max:
        status = "❌ Over Reinforced"
    elif rho < rho_min:
        final_As = As_min
        status = "⚠️ Min Steel Governs"

    select_str = ""
    if "Over" not in status:
        # Calculate Number of Bars
        unit_area = 3.1416 * (db_select/10)**2 / 4 if 'Metric' in params['unit'] else 3.1416 * db_select**2 / 4
        num = math.ceil(final_As / unit_area)
        # Check Fit
        if num > max(b/3.0, 15): # Rough check
            select_str = f"Too many DB{db_select}"
            status = "⚠️ Section Too Small"
        else:
            select_str = f"{int(num)}-DB{db_select}"
            
    return_As = final_As if 'Metric' in params['unit'] else final_As/100
    return { "Type": type_str, "Mu": abs(Mu), "As_req": return_As, "Status": status, "Bars": select_str }

def calculate_shear_capacity(Vu, params):
    fc = params['fc']
    fy_stir = params['fys']
    b = params['b']
    d = params['h'] - params['cv']
    db_stir = params['db_stirrup']
    step = params.get('s_step', 2.5) # ค่า Step ที่เพิ่มเข้ามา
    
    spacing_txt = ""
    
    if 'Metric' in params['unit']:
        vc = 0.53 * np.sqrt(fc) * b * d 
        phi_vc = 0.85 * vc 
        vu_val = abs(Vu)
        
        if vu_val > phi_vc:
            vs = (vu_val - phi_vc) / 0.85
            av = 2 * (3.1416 * (db_stir/10)**2 / 4)
            s_req = (av * fy_stir * d) / vs
            s_max = d/2
            s_use = min(s_req, s_max, 30.0)
            
            # Apply Rounding Step (Floor)
            s_use = math.floor(s_use / step) * step
            
            if s_use < 5.0: spacing_txt = f"RB{db_stir} - Increase Section"
            else: spacing_txt = f"RB{db_stir}@{s_use:.0f}cm"
            
        elif vu_val > phi_vc/2:
            s_max = d/2
            # Min stirrup logic simplified
            s_use = min(s_max, 30.0)
            s_use = math.floor(s_use / step) * step
            spacing_txt = f"RB{db_stir}@{s_use:.0f}cm (Min)"
        else:
            spacing_txt = "None Req."
            
    else:
        # SI Unit Logic (Simulated for brevity, similar structure)
        vc = 0.17 * np.sqrt(fc) * b*10 * d*10 
        phi_vc = 0.75 * (vc / 1000) 
        vu_val = abs(Vu) 
        if vu_val > phi_vc:
            spacing_txt = f"RB{db_stir} (Calc)" # Simplified for this demo
        elif vu_val > phi_vc/2:
            spacing_txt = f"RB{db_stir} (Min)"
        else:
            spacing_txt = "None"
        
    return vu_val, phi_vc, spacing_txt
