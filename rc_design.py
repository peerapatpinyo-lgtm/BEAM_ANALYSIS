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
    
    phi_b = 0.90
    beta1 = 0.85 if fc <= 280 else max(0.65, 0.85 - 0.05*(fc-280)/70)
    
    # Unit Consistency
    if 'Metric' in params['unit']:
        M_design = abs(Mu) * 100 # kg-m -> kg-cm
        fc_calc, fy_calc = fc, fy
        b_calc, d_calc = b, d
        rho_min = max(14/fy_calc, 0.25*np.sqrt(fc_calc)/fy_calc) 
        bal_const = 6120
    else: 
        M_design = abs(Mu) * 1e6 # kN-m -> N-mm
        fc_calc, fy_calc = fc, fy
        b_calc, d_calc = b*10, d*10
        rho_min = max(1.4/fy_calc, 0.25*np.sqrt(fc_calc)/fy_calc)
        bal_const = 6000

    Rn = M_design / (phi_b * b_calc * d_calc**2)
    
    calc_logs = []
    calc_logs.append(f"Mu = {M_design:.0f} (converted)")
    calc_logs.append(f"Rn = {Rn:.4f}")
    
    try:
        term = 1 - 2*Rn/(0.85*fc_calc)
        if term < 0:
            rho = 999 
        else:
            rho = (0.85 * fc_calc / fy_calc) * (1 - np.sqrt(term))
            calc_logs.append(f"rho_req = {rho:.5f}")
    except:
        rho = 999
        
    rho_bal = 0.85 * beta1 * (fc_calc/fy_calc) * (bal_const/(bal_const+fy_calc))
    rho_max = 0.75 * rho_bal 
    
    As_req = rho * b_calc * d_calc
    As_min = rho_min * b_calc * d_calc
    
    bars = [12, 16, 20, 25, 28]
    select_str = ""
    status = ""
    final_As = 0
    
    if rho == 999 or rho > rho_max:
        status = "❌ Over Reinforced"
        final_As = As_req
    elif rho < rho_min:
        final_As = As_min
        status = "⚠️ Min Steel Governs"
        calc_logs.append(f"rho < rho_min ({rho_min:.5f})")
    else:
        final_As = As_req
        status = "✅ OK"
        
    if "Over" not in status:
        As_target_cm2 = final_As if 'Metric' in params['unit'] else final_As/100
        found = False
        for db in bars:
            area = 3.1416 * (db/10)**2 / 4
            num = math.ceil(As_target_cm2 / area)
            if num <= max(b/2.5, 10): 
                 select_str = f"{num}-DB{db}"
                 found = True
                 break
        if not found: select_str = f"{math.ceil(As_target_cm2/3.14)}+-DB25"
    
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
    b = params['b']
    d = params['h'] - params['cv']
    
    if 'Metric' in params['unit']:
        vc = 0.53 * np.sqrt(fc) * b * d 
        phi_vc = 0.85 * vc 
        vu_val = abs(Vu) 
    else:
        vc = 0.17 * np.sqrt(fc) * b*10 * d*10 # N
        phi_vc = 0.75 * (vc / 1000) # kN
        vu_val = abs(Vu) 
        
    return vu_val, phi_vc
