import numpy as np
import math
import re

def parse_bars(bar_str):
    """Parses '4-DB20' to (4, 20). Returns None if invalid."""
    try:
        if not bar_str or "Over" in bar_str: return None
        match = re.search(r'(\d+)-DB(\d+)', bar_str)
        if match:
            return int(match.group(1)), int(match.group(2))
    except:
        pass
    return None

def calculate_flexure_sdm(Mu, type_str, params):
    """Calculates reinforcement for a given Moment (Mu) using SDM."""
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

    # Required Rho
    Rn = M_design / (phi_b * b_calc * d_calc**2)
    rho_bal = 0.85 * beta1 * (fc_calc/fy_calc) * (bal_const/(bal_const+fy_calc))
    rho_max = 0.75 * rho_bal 
    
    try:
        term = 1 - 2*Rn/(0.85*fc_calc)
        if term < 0:
            rho = 999 
        else:
            rho = (0.85 * fc_calc / fy_calc) * (1 - np.sqrt(term))
    except:
        rho = 999
        
    As_req = rho * b_calc * d_calc
    As_min = rho_min * b_calc * d_calc
    
    # Logic for Bar Selection
    bars = [12, 16, 20, 25, 28]
    select_str = ""
    status = ""
    final_As = 0
    
    if rho == 999 or rho > rho_max:
        status = "❌ Over Reinforced (Section too small)"
        final_As = As_req
    elif rho < rho_min:
        final_As = As_min
        status = "⚠️ Minimum Steel Governs"
    else:
        final_As = As_req
        status = "✅ OK"
        
    # Select Bars
    if "Over" not in status:
        # Convert As to cm2 for checking
        As_target_cm2 = final_As if 'Metric' in params['unit'] else final_As/100
        
        # Simple selection logic: Pick the bar that gives reasonable spacing/qty
        found = False
        for db in bars:
            area = 3.1416 * (db/10)**2 / 4
            num = math.ceil(As_target_cm2 / area)
            if num <= max(b/3, 8): # Limit max bars for sanity
                 select_str = f"{num}-DB{db}"
                 found = True
                 break
        if not found: select_str = f"{math.ceil(As_target_cm2/3.14)}+-DB25"
    
    # Return Area in cm2 always for display
    return_As = final_As if 'Metric' in params['unit'] else final_As/100

    return {
        "Type": type_str,
        "Mu": abs(Mu),
        "As_req": return_As, # cm2
        "Status": status,
        "Bars": select_str
    }

def calculate_shear_capacity(Vu, params):
    """Calculates Shear Capacity."""
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
