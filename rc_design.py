import numpy as np

def calculate_rc_design(max_M, max_V, fc, fy, b, h, cov, method, unit_sys, main_bar_area, stirrup_area, manual_s=0):
    logs = []
    
    stress_unit = "MPa" if "kN" in unit_sys else "ksc"
    
    logs.append("### 1. Design Parameters")
    logs.append(f"- **Material:** $f'_c = {fc}$ {stress_unit}, $f_y = {fy}$ {stress_unit}")
    # รับค่ามาเป็น cm (จาก input_handler) แต่โชว์เป็น mm ใน Log
    logs.append(f"- **Section:** $b={b*10:.0f}$ mm, $h={h*10:.0f}$ mm") 

    # Convert Units for Internal Calculation (kg, cm)
    if "kN" in unit_sys:
        Mu_calc = max_M * 1000 * 100 / 9.80665
        Vu_calc = max_V * 1000 / 9.80665
        fc_c = fc * 10.197 
        fy_c = fy * 10.197
    else:
        Mu_calc = max_M * 100
        Vu_calc = max_V
        fc_c, fy_c = fc, fy
    
    d = h - cov - 0.9 
    logs.append(f"- **Effective Depth ($d$):** {d*10:.0f} mm")
    
    result = {}

    # --- A. Flexural Design ---
    logs.append("### 2. Flexural Design")
    if method == "SDM":
        phi_b = 0.9
        Rn = Mu_calc / (phi_b * b * d**2)
        term = 1 - (2*Rn)/(0.85*fc_c)
        if term < 0:
            result.update({'As_req': 9999, 'nb': 0, 'msg_flex': "Fail"})
            logs.append(f"❌ **Fail:** Section too small.")
        else:
            rho = (0.85*fc_c/fy_c) * (1 - np.sqrt(term))
            min_rho = 14/fy_c
            rho_design = max(rho, min_rho)
            As_req = rho_design * b * d
            result.update({'As_req': As_req, 'msg_flex': "OK"})
            logs.append(f"- $A_{{s,req}} = {As_req:.2f}$ cm²")
    else: # WSD
        j = 0.875 
        As_req = Mu_calc / (0.5 * fy_c * j * d)
        result.update({'As_req': As_req, 'msg_flex': "OK (WSD)"})
        logs.append(f"- $A_{{s,req}} = {As_req:.2f}$ cm²")

    if result.get('As_req', 0) == 9999:
        result['nb'] = 0
    else:
        nb_calc = result['As_req'] / main_bar_area
        nb_use = max(2, int(np.ceil(nb_calc)))
        result['nb'] = nb_use
        logs.append(f"- **Use:** {nb_use} Bars ($A_s={nb_use*main_bar_area:.2f}$ cm²)")

    # --- B. Shear Design ---
    logs.append("### 3. Shear Design")
    if method == "SDM": 
        Vc = 0.85 * 0.53 * np.sqrt(fc_c) * b * d 
        stress = fy_c
    else:
        Vc = 0.29 * np.sqrt(fc_c) * b * d
        stress = 0.5 * fy_c

    if Vu_calc > Vc:
        Vs_req = Vu_calc - Vc
        Av = 2 * stirrup_area 
        if Vs_req <= 0: s_req_cm = 999
        else: s_req_cm = (Av * stress * d) / Vs_req
        
        logs.append(f"- $V_u > V_c$: Need Stirrups")
        
        if manual_s > 0:
            s_use_cm = manual_s
        else:
            limit_s = min(s_req_cm, d/2, 60)
            if limit_s < 2.5: limit_s = 2.5
            s_use_cm = int(5 * round(limit_s/5)) 
            if s_use_cm == 0: s_use_cm = 5
            if s_use_cm > limit_s: s_use_cm = int(limit_s)
    else:
        logs.append(f"- $V_u < V_c$: Min Stirrups")
        s_use_cm = manual_s if manual_s > 0 else int(d/2)

    # Output mm
    s_use_mm = int(s_use_cm * 10)
    result['stirrup_text'] = f"@{s_use_mm} mm"
    result['s_value_mm'] = s_use_mm
    result['logs'] = logs
    return result
