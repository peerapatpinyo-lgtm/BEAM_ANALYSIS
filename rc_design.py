import numpy as np
import math

def get_default_factors(code_name):
    if "WSD" in code_name: return 1.0, 1.0, "WSD"
    elif "ACI" in code_name: return 1.2, 1.6, "SDM"
    else: return 1.4, 1.7, "SDM"

def calculate_rc_design(max_M, max_V, fc, fy, b, h, cov, method, unit_sys, main_bar_area, stirrup_area):
    logs = [] # เก็บรายการคำนวณ
    logs.append(f"**1. Design Parameters:**")
    logs.append(f"- Method: {method} | Code: {unit_sys}")
    
    # 1. Convert Units to MKS (kg, cm)
    if "kN" in unit_sys:
        Mu_calc = max_M * 1000 * 100 / 9.80665
        Vu_calc = max_V * 1000 / 9.80665
        fc_c, fy_c = fc * 10.197, fy * 10.197
        logs.append(f"- Convert Input: Mu = {Mu_calc:.2f} kg-cm, Vu = {Vu_calc:.2f} kg")
        logs.append(f"- Material: fc' = {fc_c:.2f} ksc, fy = {fy_c:.2f} ksc")
    else:
        Mu_calc, Vu_calc = max_M * 100, max_V
        fc_c, fy_c = fc, fy
        logs.append(f"- Input: Mu = {Mu_calc:.2f} kg-cm, Vu = {Vu_calc:.2f} kg")

    d = h - cov - 0.9 # Effective depth
    logs.append(f"- Effective Depth (d) = {h} - {cov} - 0.9 = {d:.2f} cm")
    
    result = {}

    # --- A. Flexural Design ---
    logs.append(f"\n**2. Flexural Design (Moment):**")
    
    if method == "SDM":
        phi_b = 0.9
        logs.append(f"- Strength Design Method (SDM): phi = {phi_b}")
        Rn = Mu_calc / (phi_b * b * d**2)
        logs.append(f"- Rn = Mu / (phi*b*d^2) = {Mu_calc:.2f} / ({phi_b}*{b}*{d**2:.2f}) = {Rn:.2f} ksc")
        
        term = 1 - (2*Rn)/(0.85*fc_c)
        if term < 0:
            result.update({'As_req': 9999, 'msg_flex': "❌ Section too small (Rn > Max)"})
            logs.append(f"❌ Error: Section too small (Term under sqrt is negative)")
        else:
            rho = (0.85*fc_c/fy_c) * (1 - np.sqrt(term))
            logs.append(f"- rho_req = (0.85*fc/fy)*(1 - sqrt(1 - 2Rn/0.85fc)) = {rho:.5f}")
            
            min_rho = 14/fy_c
            logs.append(f"- rho_min = 14/fy = {min_rho:.5f}")
            
            rho_design = max(rho, min_rho)
            As_req = rho_design * b * d
            result.update({'As_req': As_req, 'msg_flex': "✅ Moment OK"})
            logs.append(f"- As_req = rho * b * d = {rho_design:.5f} * {b} * {d} = **{As_req:.2f} cm²**")
            
    else: # WSD
        logs.append(f"- Working Stress Design (WSD)")
        n = 135 / np.sqrt(fc_c) # Est n
        k = 1 / (1 + (fy_c*0.5)/(fc_c*0.45*n)) # Approximation
        j = 0.875 # Simplify j
        logs.append(f"- Est parameters: j ≈ {j}")
        
        As_req = Mu_calc / (0.5 * fy_c * j * d)
        result.update({'As_req': As_req, 'msg_flex': "✅ Moment OK (WSD)"})
        logs.append(f"- As_req = M / (fs * j * d) = {Mu_calc:.2f} / ({0.5*fy_c:.2f} * {j} * {d}) = **{As_req:.2f} cm²**")

    # Bar Selection
    if result['As_req'] == 9999:
        result['nb'] = 0
    else:
        nb_calc = result['As_req'] / main_bar_area
        result['nb'] = max(2, int(np.ceil(nb_calc)))
        logs.append(f"- Select Rebar: Need {result['As_req']:.2f} / {main_bar_area} = {nb_calc:.2f} -> Use **{result['nb']} bars**")

    # --- B. Shear Design ---
    logs.append(f"\n**3. Shear Design:**")
    Vc = 0.53 * np.sqrt(fc_c) * b * d
    if method == "SDM": 
        Vc *= 0.85 # phi shear
        logs.append(f"- Vc = 0.85 * 0.53 * sqrt(fc') * b * d = {Vc:.2f} kg")
    else:
        Vc = 0.29 * np.sqrt(fc_c) * b * d # WSD Vc approx
        logs.append(f"- Vc = 0.29 * sqrt(fc') * b * d = {Vc:.2f} kg")
    
    logs.append(f"- Vu = {Vu_calc:.2f} kg")
    
    if Vu_calc > Vc:
        Vs_req = Vu_calc - Vc
        logs.append(f"- Vu > Vc: Need Stirrups. Vs_req = {Vs_req:.2f} kg")
        
        # Spacing
        Av = 2 * stirrup_area
        if method == "SDM":
            s_req = (Av * fy_c * d) / Vs_req # using fy for stirrup
        else:
            s_req = (Av * (0.5*fy_c) * d) / Vs_req

        logs.append(f"- Av (2 legs) = {Av:.2f} cm²")
        logs.append(f"- s_req = (Av * fs * d) / Vs = {s_req:.2f} cm")
        
        s = int(5 * round(min(s_req, d/2, 60)/5)) or 5
        result.update({'stirrup_text': f"@{s} cm", 'msg_shear': "⚠️ Shear Reinf."})
        logs.append(f"- Use spacing @{s} cm")
    else:
        result.update({'stirrup_text': f"@{int(d/2)} cm (Min)", 'msg_shear': "✅ Min Shear"})
        logs.append(f"- Vu < Vc: Concrete OK. Use Min Stirrups @{int(d/2)} cm")
        
    result['logs'] = logs
    return result
