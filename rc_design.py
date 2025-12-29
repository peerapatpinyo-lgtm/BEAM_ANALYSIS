import numpy as np
import math
import re

def parse_bars(bar_str):
    try:
        if not bar_str or "Over" in bar_str: return None
        match = re.search(r'(\d+)-DB(\d+)', bar_str)
        if match:
            return int(match.group(1)), int(match.group(2))
    except:
        pass
    return None

def calculate_flexure_sdm(Mu, type_str, params):
    fc = params['fc']
    fy = params['fy']
    b = params['b']
    h = params['h']
    d = h - params['cv']
    db_select = params['db_main'] # ใช้ค่าที่ User เลือก
    
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

    Rn = M_design / (phi_b * b_calc * d_calc**2)
    
    calc_logs = []
    
    try:
        term = 1 - 2*Rn/(0.85*fc_calc)
        if term < 0:
            rho = 999 
        else:
            rho = (0.85 * fc_calc / fy_calc) * (1 - np.sqrt(term))
    except:
        rho = 999
        
    bal_const = 6120 if 'Metric' in params['unit'] else 6000
    rho_bal = 0.85 * beta1 * (fc_calc/fy_calc) * (bal_const/(bal_const+fy_calc))
    rho_max = 0.75 * rho_bal 
    
    As_req = rho * b_calc * d_calc
    As_min = rho_min * b_calc * d_calc
    
    select_str = ""
    status = ""
    final_As = 0
    
    if rho == 999 or rho > rho_max:
        status = "❌ Over Reinforced"
        final_As = As_req
    elif rho < rho_min:
        final_As = As_min
        status = "⚠️ Min Steel Governs"
    else:
        final_As = As_req
        status = "✅ OK"
        
    if "Over" not in status:
        As_target = final_As # cm2 or mm2
        # คำนวณจำนวนเส้นจากเหล็กที่เลือก
        unit_area = 3.1416 * (db_select/10)**2 / 4 if 'Metric' in params['unit'] else 3.1416 * db_select**2 / 4
        num = math.ceil(As_target / unit_area)
        
        # Check Fit in Width
        # spacing check simplified
        if num > max(b/2.5, 10):
            select_str = f"Too many DB{db_select}"
        else:
            select_str = f"{num}-DB{db_select}"
            
    return_As = final_As if 'Metric' in params['unit'] else final_As/100

    return {
        "Type": type_str,
        "Mu": abs(Mu),
        "As_req": return_As, 
        "Status": status,
        "Bars": select_str,
        "Logs": calc_logs
    }

def calculate_shear_capacity(Vu, params):
    fc = params['fc']
    fy_stir = params['fys']
    b = params['b']
    d = params['h'] - params['cv']
    db_stir = params['db_stirrup']
    
    spacing_txt = ""
    
    if 'Metric' in params['unit']:
        # kg, cm
        vc = 0.53 * np.sqrt(fc) * b * d 
        phi_vc = 0.85 * vc 
        vu_val = abs(Vu)
        
        # Design Stirrup
        if vu_val > phi_vc:
            vs = (vu_val - phi_vc) / 0.85
            av = 2 * (3.1416 * (db_stir/10)**2 / 4) # 2 legs
            s_req = (av * fy_stir * d) / vs
            s_max = d/2
            s_use = min(s_req, s_max, 30.0) # Cap at 30cm
            # Round down to nearest 2.5cm
            s_use = math.floor(s_use / 2.5) * 2.5
            spacing_txt = f"RB{db_stir}@{s_use:.0f}cm"
        elif vu_val > phi_vc/2:
            spacing_txt = f"RB{db_stir}@Min" # Simplified Min
        else:
            spacing_txt = "None"
            
    else:
        # N, mm
        vc = 0.17 * np.sqrt(fc) * b*10 * d*10 
        phi_vc = 0.75 * (vc / 1000) # kN
        vu_val = abs(Vu) 
        
        if vu_val > phi_vc:
            vs = (vu_val - phi_vc) / 0.75 * 1000 # N
            av = 2 * (3.1416 * db_stir**2 / 4)
            s_req = (av * fy_stir * d*10) / vs
            s_max = (d*10)/2
            s_use = min(s_req, s_max, 300.0)
            s_use = math.floor(s_use / 25) * 25
            spacing_txt = f"RB{db_stir}@{s_use:.0f}mm"
        elif vu_val > phi_vc/2:
            spacing_txt = f"RB{db_stir}@Min"
        else:
            spacing_txt = "None"
        
    return vu_val, phi_vc, spacing_txt
