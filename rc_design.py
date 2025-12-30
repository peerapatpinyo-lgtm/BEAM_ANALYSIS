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
    """คำนวณเหล็กรับแรงดัด (SDM) พร้อมเช็ค Min/Max"""
    fc = params['fc']
    fy = params['fy']
    b = params['b']
    h = params['h']
    d = h - params['cv']
    db_select = params['db_main']
    
    phi_b = 0.90
    beta1 = 0.85 if fc <= 280 else max(0.65, 0.85 - 0.05*(fc-280)/70)
    
    # Unit Conversion
    if 'Metric' in params['unit']:
        M_design = abs(Mu) * 100 # kg-m -> kg-cm
        fc_c, fy_c = fc, fy
        b_c, d_c = b, d
        rho_min = max(14/fy_c, 0.25*np.sqrt(fc_c)/fy_c) 
    else: 
        M_design = abs(Mu) * 1e6 # kN-m -> N-mm
        fc_c, fy_c = fc, fy
        b_c, d_c = b*10, d*10
        rho_min = max(1.4/fy_c, 0.25*np.sqrt(fc_c)/fy_c)

    # Design Calculation
    rho = 999
    try:
        Rn = M_design / (phi_b * b_c * d_c**2)
        term = 1 - 2*Rn/(0.85*fc_c)
        if term >= 0:
            rho = (0.85 * fc_c / fy_c) * (1 - np.sqrt(term))
    except:
        pass
        
    bal_const = 6120 if 'Metric' in params['unit'] else 6000
    rho_bal = 0.85 * beta1 * (fc_c/fy_c) * (bal_const/(bal_const+fy_c))
    rho_max = 0.75 * rho_bal 
    
    As_req = rho * b_c * d_c
    As_min = rho_min * b_c * d_c
    
    status = "✅ OK"
    final_As = As_req
    
    if rho == 999 or rho > rho_max:
        status = "❌ Over Reinforced"
        final_As = As_req # Just to show value
    elif rho < rho_min:
        final_As = As_min
        status = "⚠️ Min Steel Governs"

    select_str = ""
    if "Over" not in status:
        # Bar Selection
        unit_area = 3.1416 * (db_select/10)**2 / 4 if 'Metric' in params['unit'] else 3.1416 * db_select**2 / 4
        num = math.ceil(final_As / unit_area)
        
        # Spacing/Crowding Check
        width_avail = b_c - 2*params['cv'] if 'Metric' in params['unit'] else b_c - 2*params['cv']*10
        # Rough check: spacing between bars should be approx db
        max_bars = width_avail / ((db_select/10)*2) if 'Metric' in params['unit'] else width_avail / (db_select*2)
        
        if num > max_bars + 1: 
            select_str = f"Too many DB{db_select}"
            status = "⚠️ Section Too Small"
        else:
            select_str = f"{int(num)}-DB{db_select}"
            
    return_As = final_As if 'Metric' in params['unit'] else final_As/100
    return { "Type": type_str, "Mu": abs(Mu), "As_req": return_As, "Status": status, "Bars": select_str }

def calculate_shear_capacity(Vu, params):
    """คำนวณเหล็กปลอก พร้อม Spacing Step"""
    fc = params['fc']
    fy_stir = params['fys']
    b = params['b']
    d = params['h'] - params['cv']
    db_stir = params['db_stirrup']
    step = params.get('s_step', 2.5) # Feature: Spacing Step
    
    spacing_txt = ""
    v_cap_display = 0
    
    if 'Metric' in params['unit']:
        vc = 0.53 * np.sqrt(fc) * b * d 
        phi_vc = 0.85 * vc 
        vu_val = abs(Vu)
        v_cap_display = phi_vc
        
        if vu_val > phi_vc:
            vs = (vu_val - phi_vc) / 0.85
            av = 2 * (3.1416 * (db_stir/10)**2 / 4)
            s_req = (av * fy_stir * d) / vs
            s_max = d/2
            s_use = min(s_req, s_max, 30.0)
            
            # Feature: Rounding Step (ปัดลงให้ลงตัว)
            s_use = math.floor(s_use / step) * step
            
            if s_use < 5.0: 
                spacing_txt = f"RB{db_stir} - Increase Section"
            else: 
                spacing_txt = f"RB{db_stir}@{s_use:.0f}cm"
            
        elif vu_val > phi_vc/2:
            s_max = d/2
            s_use = min(s_max, 30.0)
            s_use = math.floor(s_use / step) * step
            spacing_txt = f"RB{db_stir}@{s_use:.0f}cm (Min)"
        else:
            spacing_txt = "None Req."
            
    else:
        # SI Unit (Simplified)
        vc = 0.17 * np.sqrt(fc) * b*10 * d*10 
        phi_vc = 0.75 * (vc / 1000) 
        vu_val = abs(Vu)
        v_cap_display = phi_vc
        if vu_val > phi_vc: spacing_txt = f"RB{db_stir} (Calc SI)"
        else: spacing_txt = "Min / None"
        
    return abs(Vu), v_cap_display, spacing_txt
