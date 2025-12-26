import numpy as np

def calculate_rc_design(max_M, max_V, fc, fy, b, h, cov, method, unit_sys, main_bar_area, stirrup_area, manual_s=0):
    logs = []
    
    # --- 1. Fix IndexError: Check Unit String Safely ---
    # แทนที่จะ split string เราเช็คว่ามีคำว่า kN หรือไม่ เพื่อกำหนดหน่วยแสดงผล
    stress_unit = "MPa" if "kN" in unit_sys else "ksc"
    
    # Header
    logs.append("### 1. Design Parameters")
    logs.append(f"- **Code:** {unit_sys} | **Method:** {method}")
    # ใช้ตัวแปร stress_unit ที่เตรียมไว้แทนการ split
    logs.append(f"- **Material:** $f'_c = {fc}$ {stress_unit}, $f_y = {fy}$ {stress_unit}")
    logs.append(f"- **Section:** $b={b:.1f}$ cm, $h={h:.1f}$ cm, $cov={cov:.1f}$ cm")

    # --- 2. Unit Conversion ---
    if "kN" in unit_sys:
        Mu_calc = max_M * 1000 * 100 / 9.80665
        Vu_calc = max_V * 1000 / 9.80665
        fc_c = fc * 10.197 
        fy_c = fy * 10.197
        logs.append(f"> **Load Conversion:** $M_u \\approx {Mu_calc:.0f}$ kg-cm, $V_u \\approx {Vu_calc:.0f}$ kg")
    else:
        Mu_calc = max_M * 100
        Vu_calc = max_V
        fc_c, fy_c = fc, fy
        logs.append(f"> **Design Load:** $M_u = {Mu_calc:.0f}$ kg-cm, $V_u = {Vu_calc:.0f}$ kg")
    
    d = h - cov - 0.9 
    logs.append(f"- **Effective Depth ($d$):** ${h} - {cov} - 0.9 = {d:.2f}$ cm")
    
    result = {}

    # --- A. Flexural Design ---
    logs.append("### 2. Flexural Design (Moment)")
    if method == "SDM":
        phi_b = 0.9
        Rn = Mu_calc / (phi_b * b * d**2)
        logs.append(f"1. **Calculate $R_n$:**")
        logs.append(f"   $$R_n = \\frac{{M_u}}{{\\phi b d^2}} = \\frac{{{Mu_calc:.0f}}}{{0.9 \\cdot {b} \\cdot {d:.2f}^2}} = {Rn:.2f} \\text{{ ksc}}$$")
        
        term = 1 - (2*Rn)/(0.85*fc_c)
        if term < 0:
            result.update({'As_req': 9999, 'nb': 0, 'msg_flex': "❌ Section Too Small"})
            logs.append(f"❌ **Fail:** $R_n$ is too high. Section is too small.")
        else:
            rho = (0.85*fc_c/fy_c) * (1 - np.sqrt(term))
            min_rho = 14/fy_c
            rho_design = max(rho, min_rho)
            As_req = rho_design * b * d
            logs.append(f"2. **Calculate Steel Ratio ($\\rho$):**")
            logs.append(f"   - $\\rho_{{req}} = {rho_design:.5f}$")
            logs.append(f"3. **Required Area ($A_{{s,req}}$):**")
            logs.append(f"   $$A_s = {As_req:.2f} \\text{{ cm}}^2$$")
            result.update({'As_req': As_req, 'msg_flex': "✅ OK"})
            
    else: # WSD
        n = 135/np.sqrt(fc_c) if "WSD" in method else 10 
        j = 0.875 
        As_req = Mu_calc / (0.5 * fy_c * j * d)
        logs.append(f"1. **Working Stress Formula:**")
        logs.append(f"   $$A_s = \\frac{{M}}{{f_s j d}} = {As_req:.2f} \\text{{ cm}}^2$$")
        result.update({'As_req': As_req, 'msg_flex': "✅ OK (WSD)"})

    # Bar Selection
    if result.get('As_req', 0) == 9999:
        result['nb'] = 0
    else:
        nb_calc = result['As_req'] / main_bar_area
        nb_use = max(2, int(np.ceil(nb_calc)))
        result['nb'] = nb_use
        logs.append(f"4. **Select Bars:**")
        logs.append(f"   - Use: **{nb_use} Bars** ($A_{{s,prov}} = {nb_use*main_bar_area:.2f}$ cm²)")

    # --- B. Shear Design ---
    logs.append("### 3. Shear Design")
    if method == "SDM": 
        Vc = 0.85 * 0.53 * np.sqrt(fc_c) * b * d 
        txt_vc = r"0.85 \cdot 0.53 \sqrt{f'_c} b d"
    else:
        Vc = 0.29 * np.sqrt(fc_c) * b * d
        txt_vc = r"0.29 \sqrt{f'_c} b d"

    logs.append(f"1. **Concrete Capacity ($V_c$):**")
    logs.append(f"   $$V_c = {txt_vc} = \\mathbf{{{Vc:.2f}}} \\text{{ kg}}$$")
    
    if Vu_calc > Vc:
        Vs_req = Vu_calc - Vc
        Av = 2 * stirrup_area 
        stress = fy_c if method == "SDM" else 0.5*fy_c
        if Vs_req <= 0: s_req_calc = 999
        else: s_req_calc = (Av * stress * d) / Vs_req
        
        logs.append(f"2. **Stirrup Requirement:**")
        logs.append(f"   - $V_u > V_c$ -> Need Stirrups")
        logs.append(f"   - $s_{{req}} = {s_req_calc:.2f} \\text{{ cm}}$")
        
        if manual_s > 0:
            s_use = manual_s
            result['msg_shear'] = "✅ User" if s_use <= s_req_calc else "⚠️ Unsafe"
        else:
            step = 5 if s_req_calc > 15 else 2.5
            limit_s = min(s_req_calc, d/2, 60)
            if limit_s < 2.5: limit_s = 2.5
            s_use = int(step * round(limit_s/step)) 
            if s_use == 0: s_use = 5
            result['msg_shear'] = "⚠️ Stirrups"
            logs.append(f"3. **Select Spacing:** Use **@{s_use} cm**")
    else:
        logs.append(f"2. **Stirrup Requirement:** $V_u < V_c$ (Use Min)")
        s_use = manual_s if manual_s > 0 else int(d/2)
        result['msg_shear'] = "✅ Min Shear"
        logs.append(f"   - Use: **@{s_use} cm**")

    result['stirrup_text'] = f"@{s_use} cm"
    result['s_value'] = s_use
    result['logs'] = logs
    return result
