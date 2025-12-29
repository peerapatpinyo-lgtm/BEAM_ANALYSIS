import numpy as np

def calculate_rc_design(mu_input, vu_input, fc, fy, b, h, cov, method, unit_sys, main_bar_area, stirrup_area, manual_s=0):
    """
    mu_input: Moment in System Units (kg-m or kN-m)
    vu_input: Shear in System Units (kg or kN)
    """
    logs = []
    result = {}
    
    # ส่งค่า Input กลับไปให้ app.py แสดงผล (แก้ KeyError ตรงนี้)
    result['Mu'] = mu_input
    result['Vu'] = vu_input

    # 1. แปลงหน่วยเข้าสูตร (kg, cm)
    # Metric: Input kg-m -> *100 -> kg-cm
    # SI: Input kN-m -> *1000(N)*100(cm)/9.81(g) -> kg-cm (approx)
    if "kN" in unit_sys:
        # 1 kN-m = 1000 N-m = 100,000 N-cm. 1 kgf ~= 9.81 N
        # conversion factor: 1000 * 100 / 9.806
        k_moment = 1000 * 100 / 9.80665
        k_shear = 1000 / 9.80665
        k_stress = 10.197 # MPa to ksc
        
        Mu_calc = abs(mu_input) * k_moment
        Vu_calc = abs(vu_input) * k_shear
        fc_c = fc * k_stress 
        fy_c = fy * k_stress
    else:
        # Metric (kg, m)
        Mu_calc = abs(mu_input) * 100.0 # kg-m to kg-cm
        Vu_calc = abs(vu_input)         # kg
        fc_c, fy_c = fc, fy
    
    d = h - cov - 0.9 # Effective depth estimate
    
    # --- Flexural Design (M) ---
    if method == "SDM":
        phi_b = 0.9
        Rn = Mu_calc / (phi_b * b * d**2)
        term = 1 - (2*Rn)/(0.85*fc_c)
        
        if term < 0:
            result.update({'As_req': 9999, 'nb': 0, 'msg_flex': "❌ Fail (Sec. Small)"})
            logs.append("Section too small for Moment!")
        else:
            rho = (0.85*fc_c/fy_c) * (1 - np.sqrt(term))
            min_rho = 14/fy_c
            rho_design = max(rho, min_rho)
            As_req = rho_design * b * d
            result.update({'As_req': As_req, 'msg_flex': "✅ OK"})
    else: # WSD
        n = 135 / np.sqrt(fc_c) # Modular ratio approx
        k = n / (n + (fy_c/2) / (0.45*fc_c)) # k approx
        j = 1 - k/3
        As_req = Mu_calc / (0.5 * fy_c * j * d) # Simple WSD
        result.update({'As_req': As_req, 'msg_flex': "✅ OK (WSD)"})

    # คำนวณจำนวนเส้น
    if result['As_req'] == 9999:
        result['nb'] = 0
    else:
        nb = max(2, int(np.ceil(result['As_req'] / main_bar_area)))
        result['nb'] = nb

    # --- Shear Design (V) ---
    if method == "SDM": 
        Vc = 0.53 * np.sqrt(fc_c) * b * d # ACI simplified
        phi_v = 0.85
        Vc_allow = phi_v * Vc
        stress_s = fy_c
    else:
        Vc = 0.29 * np.sqrt(fc_c) * b * d
        Vc_allow = Vc
        stress_s = 0.5 * fy_c

    if Vu_calc > Vc_allow:
        Vs = (Vu_calc - Vc_allow)/0.85 if method=="SDM" else (Vu_calc - Vc_allow)
        Av = 2 * stirrup_area 
        s_req = (Av * stress_s * d) / Vs
        
        limit_s = d/2
        if manual_s > 0: s_final = manual_s
        else: s_final = min(s_req, limit_s, 30.0) # Cap at 30cm
    else:
        # Min Stirrup
        s_final = manual_s if manual_s > 0 else d/2

    # Round spacing to nearest 2.5cm
    s_use = int(2.5 * np.floor(s_final / 2.5))
    if s_use < 5: s_use = 5
    
    result['stirrup_text'] = f"@{s_use*10} mm"
    result['logs'] = logs
    
    return result
