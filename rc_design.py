import numpy as np

def calculate_rc_design(max_M, max_V, fc, fy, b, h, cov, method, unit_sys, main_bar_area, stirrup_area, manual_s=0):
    logs = []
    
    # 1. เช็คหน่วยแสดงผล (Display Unit)
    stress_unit = "MPa" if "kN" in unit_sys else "ksc"
    
    logs.append("### 1. Design Parameters")
    logs.append(f"- **Material:** $f'_c = {fc}$ {stress_unit}, $f_y = {fy}$ {stress_unit}")
    # รับค่า b, h มาเป็น cm แต่โชว์ใน Log เป็น mm
    logs.append(f"- **Section:** $b={b*10:.0f}$ mm, $h={h*10:.0f}$ mm") 

    # 2. แปลงหน่วยเพื่อคำนวณ (Internal Calculation uses kg, cm)
    if "kN" in unit_sys:
        Mu_calc = max_M * 1000 * 100 / 9.80665 # kN-m -> kg-cm
        Vu_calc = max_V * 1000 / 9.80665       # kN -> kg
        fc_c = fc * 10.197 
        fy_c = fy * 10.197
    else:
        Mu_calc = max_M * 100 # kg-m -> kg-cm
        Vu_calc = max_V
        fc_c, fy_c = fc, fy
    
    d = h - cov - 0.9 # d (cm)
    logs.append(f"- **Effective Depth ($d$):** {d*10:.0f} mm")
    
    result = {}

    # --- A. Flexural Design (Moment) ---
    logs.append("### 2. Flexural Design")
    if method == "SDM":
        phi_b = 0.9
        Rn = Mu_calc / (phi_b * b * d**2)
        term = 1 - (2*Rn)/(0.85*fc_c)
        
        if term < 0:
            result.update({'As_req': 9999, 'nb': 0, 'msg_flex': "❌ Fail"})
            logs.append(f"❌ **Fail:** Section too small ($R_n$ too high).")
        else:
            rho = (0.85*fc_c/fy_c) * (1 - np.sqrt(term))
            min_rho = 14/fy_c
            rho_design = max(rho, min_rho)
            As_req = rho_design * b * d
            result.update({'As_req': As_req, 'msg_flex': "✅ OK"})
            logs.append(f"- $A_{{s,req}} = {As_req:.2f}$ cm²")
    else: # WSD
        j = 0.875 
        As_req = Mu_calc / (0.5 * fy_c * j * d)
        result.update({'As_req': As_req, 'msg_flex': "✅ OK (WSD)"})
        logs.append(f"- $A_{{s,req}} = {As_req:.2f}$ cm²")

    # Bar Selection
    if result.get('As_req', 0) == 9999:
        result['nb'] = 0
    else:
        nb_calc = result['As_req'] / main_bar_area
        nb_use = max(2, int(np.ceil(nb_calc))) # ขั้นต่ำ 2 เส้น
        result['nb'] = nb_use
        logs.append(f"- **Use Main Bars:** {nb_use} bars ($A_s={nb_use*main_bar_area:.2f}$ cm²)")

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
        
        # Logic เลือกระยะแอด
        if manual_s > 0:
            s_use_cm = manual_s
        else:
            # Rounding logic (ปัดเศษให้สวยงาม)
            limit_s = min(s_req_cm, d/2, 60)
            if limit_s < 2.5: limit_s = 2.5
            s_use_cm = int(5 * round(limit_s/5)) # ปัดเข้าหา 5 ซม.
            if s_use_cm == 0: s_use_cm = 5
            if s_use_cm > limit_s: s_use_cm = int(limit_s) # ห้ามเกิน limit
            
    else:
        logs.append(f"- $V_u < V_c$: Use Min Stirrups")
        s_use_cm = manual_s if manual_s > 0 else int(d/2)

    # --- FINAL CONVERSION TO MM ---
    s_use_mm = int(s_use_cm * 10)
    
    result['stirrup_text'] = f"@{s_use_mm} mm"  # ส่งค่า text เป็น mm กลับไป
    result['s_value_mm'] = s_use_mm           # ส่งค่าตัวเลข mm กลับไปวาดกราฟ
    result['logs'] = logs
    
    return result
