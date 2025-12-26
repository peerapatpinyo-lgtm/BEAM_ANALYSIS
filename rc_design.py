import numpy as np

def get_default_factors(code_name):
    # แยกปี ACI และ EIT
    if "WSD" in code_name: return 1.0, 1.0, "WSD"
    elif "ACI 318-99" in code_name: return 1.4, 1.7, "SDM" # Old ACI
    elif "ACI 318" in code_name: return 1.2, 1.6, "SDM" # 14, 19 New ACI
    else: return 1.4, 1.7, "SDM" # EIT SDM default

def calculate_rc_design(max_M, max_V, fc, fy, b, h, cov, method, unit_sys, main_bar_area, stirrup_area, manual_s=0):
    logs = []
    logs.append(f"**1. Design Parameters:**")
    logs.append(f"- Code: {unit_sys} | Method: {method}")
    
    # 1. Convert Units to MKS (kg, cm) for calculation
    if "kN" in unit_sys:
        Mu_calc = max_M * 1000 * 100 / 9.80665
        Vu_calc = max_V * 1000 / 9.80665
        fc_c, fy_c = fc * 10.197, fy * 10.197
        logs.append(f"- Convert: Mu={Mu_calc:.0f} kg-cm, Vu={Vu_calc:.0f} kg")
    else:
        Mu_calc, Vu_calc = max_M * 100, max_V
        fc_c, fy_c = fc, fy
    
    d = h - cov - 0.9 # Effective depth
    result = {}

    # --- A. Flexural Design ---
    logs.append(f"\n**2. Flexure (Moment):**")
    if method == "SDM":
        phi_b = 0.9
        Rn = Mu_calc / (phi_b * b * d**2)
        term = 1 - (2*Rn)/(0.85*fc_c)
        
        if term < 0:
            result.update({'As_req': 9999, 'msg_flex': "❌ Section Too Small"})
            logs.append(f"❌ Rn too high ({Rn:.2f}). Increase section.")
        else:
            rho = (0.85*fc_c/fy_c) * (1 - np.sqrt(term))
            min_rho = 14/fy_c
            rho_design = max(rho, min_rho)
            As_req = rho_design * b * d
            result.update({'As_req': As_req, 'msg_flex': "✅ OK"})
            logs.append(f"- As_req = {As_req:.2f} cm²")
    else: # WSD
        j = 0.875 
        As_req = Mu_calc / (0.5 * fy_c * j * d)
        result.update({'As_req': As_req, 'msg_flex': "✅ OK (WSD)"})
        logs.append(f"- As_req = {As_req:.2f} cm²")

    # Bar Selection
    if result['As_req'] == 9999:
        result['nb'] = 0
    else:
        nb_calc = result['As_req'] / main_bar_area
        result['nb'] = max(2, int(np.ceil(nb_calc)))

    # --- B. Shear Design ---
    logs.append(f"\n**3. Shear:**")
    # Vc Calculation
    if method == "SDM": 
        Vc = 0.85 * 0.53 * np.sqrt(fc_c) * b * d
    else:
        Vc = 0.29 * np.sqrt(fc_c) * b * d

    logs.append(f"- Vc = {Vc:.2f} kg, Vu = {Vu_calc:.2f} kg")
    
    # Check Vu vs Vc
    if Vu_calc > Vc:
        Vs_req = Vu_calc - Vc
        Av = 2 * stirrup_area # 2 legs
        
        # Calculate Required Spacing
        stress = fy_c if method == "SDM" else 0.5*fy_c
        s_req_calc = (Av * stress * d) / Vs_req
        logs.append(f"- Need Stirrups. s_req = {s_req_calc:.2f} cm")
        
        # Determine Used Spacing (Manual or Auto)
        if manual_s > 0:
            s_use = manual_s
            status = "✅ User OK" if s_use <= s_req_calc else "❌ Unsafe"
            logs.append(f"- User Input @{s_use} cm -> {status}")
            if s_use > s_req_calc: result['msg_shear'] = "⚠️ Spacing > Required"
            else: result['msg_shear'] = "✅ User Defined"
        else:
            # Auto round down
            step = 5 if s_req_calc > 15 else 2.5
            s_use = int(step * round(min(s_req_calc, d/2, 60)/step)) or 5
            result['msg_shear'] = "⚠️ Shear Reinf."
    
    else:
        # Min Stirrups
        if manual_s > 0:
            s_use = manual_s
            logs.append(f"- Vu < Vc. User Input @{s_use} cm")
        else:
            s_use = int(d/2)
            logs.append(f"- Vu < Vc. Use Min @{s_use} cm")
        result['msg_shear'] = "✅ Min Shear"

    result['stirrup_text'] = f"@{s_use} cm"
    result['s_value'] = s_use
    result['logs'] = logs
    return result
