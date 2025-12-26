import numpy as np

def calculate_rc_design(max_M, max_V, fc, fy, b, h, cov, method, unit_sys, main_bar_area, stirrup_area, manual_s=0):
    # เปลี่ยนการเก็บ Log เป็น List ของ Dictionary เพื่อระบุประเภทการแสดงผลได้ (ถ้าต้องการ) 
    # หรือใช้ List ของ String ที่เป็น Markdown
    logs = []
    
    # Header
    logs.append("### 1. Design Parameters")
    logs.append(f"- **Code:** {unit_sys} | **Method:** {method}")
    logs.append(f"- **Material:** $f'_c = {fc}$ {unit_sys.split(',')[2].strip()}, $f_y = {fy}$ {unit_sys.split(',')[2].strip()}")
    logs.append(f"- **Section:** $b={b}$ cm, $h={h}$ cm, $cov={cov}$ cm")

    # 1. Unit Conversion
    if "kN" in unit_sys:
        Mu_calc = max_M * 1000 * 100 / 9.80665
        Vu_calc = max_V * 1000 / 9.80665
        fc_c, fy_c = fc * 10.197, fy * 10.197 # MPa -> ksc approx
        logs.append(f"> **Conversion:** $M_u \\approx {Mu_calc:.0f}$ kg-cm, $V_u \\approx {Vu_calc:.0f}$ kg")
    else:
        Mu_calc, Vu_calc = max_M * 100, max_V
        fc_c, fy_c = fc, fy
        logs.append(f"> **Design Load:** $M_u = {Mu_calc:.0f}$ kg-cm, $V_u = {Vu_calc:.0f}$ kg")
    
    d = h - cov - 0.9 # Effective depth estimate
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
            result.update({'As_req': 9999, 'msg_flex': "❌ Section Too Small"})
            logs.append(f"❌ **Fail:** $R_n$ exceeds max limit. Increase Section.")
        else:
            rho = (0.85*fc_c/fy_c) * (1 - np.sqrt(term))
            min_rho = 14/fy_c
            rho_design = max(rho, min_rho)
            As_req = rho_design * b * d
            
            logs.append(f"2. **Calculate Steel Ratio ($\\rho$):**")
            logs.append(f"   - $\\rho_{{calc}} = {rho:.5f}$")
            logs.append(f"   - $\\rho_{{min}} = 14/f_y = {min_rho:.5f}$")
            logs.append(f"   - Use $\\rho = {rho_design:.5f}$")
            
            logs.append(f"3. **Required Area ($A_{{s,req}}$):**")
            logs.append(f"   $$A_s = \\rho b d = {rho_design:.5f} \\cdot {b} \\cdot {d:.2f} = \\mathbf{{{As_req:.2f}}} \\text{{ cm}}^2$$")
            result.update({'As_req': As_req, 'msg_flex': "✅ OK"})
            
    else: # WSD
        n = 135/np.sqrt(fc_c) if "WSD" in method else 10 # approximate n
        k = 1 / (1 + (fy_c/2)/(0.45*fc_c*n)) # approx k
        j = 0.875 # simplify j
        As_req = Mu_calc / (0.5 * fy_c * j * d)
        
        logs.append(f"1. **Working Stress Formula:**")
        logs.append(f"   $$A_s = \\frac{{M}}{{f_s j d}} \\approx \\frac{{{Mu_calc:.0f}}}{{0.5 \\cdot {fy_c:.0f} \\cdot {j} \\cdot {d:.2f}}}$$")
        logs.append(f"   $$A_s = \\mathbf{{{As_req:.2f}}} \\text{{ cm}}^2$$")
        result.update({'As_req': As_req, 'msg_flex': "✅ OK (WSD)"})

    # Bar Selection
    if result['As_req'] == 9999:
        result['nb'] = 0
    else:
        nb_calc = result['As_req'] / main_bar_area
        nb_use = max(2, int(np.ceil(nb_calc)))
        result['nb'] = nb_use
        logs.append(f"4. **Select Bars:**")
        logs.append(f"   - Try **{main_bar_area} cm²** (per bar)")
        logs.append(f"   - Need: ${result['As_req']:.2f} / {main_bar_area} = {nb_calc:.2f}$ bars")
        logs.append(f"   - **Use: {nb_use} Bars** ($A_{{s,prov}} = {nb_use*main_bar_area:.2f}$ cm²)")

    # --- B. Shear Design ---
    logs.append("### 3. Shear Design")
    # Vc Calculation
    if method == "SDM": 
        Vc = 0.85 * 0.53 * np.sqrt(fc_c) * b * d
        txt_vc = r"0.85 \cdot 0.53 \sqrt{f'_c} b d"
    else:
        Vc = 0.29 * np.sqrt(fc_c) * b * d
        txt_vc = r"0.29 \sqrt{f'_c} b d"

    logs.append(f"1. **Concrete Capacity ($V_c$):**")
    logs.append(f"   $$V_c = {txt_vc} = \\mathbf{{{Vc:.2f}}} \\text{{ kg}}$$")
    
    # Check Vu vs Vc
    if Vu_calc > Vc:
        Vs_req = Vu_calc - Vc
        Av = 2 * stirrup_area # 2 legs
        stress = fy_c if method == "SDM" else 0.5*fy_c
        s_req_calc = (Av * stress * d) / Vs_req
        
        logs.append(f"2. **Stirrup Requirement:**")
        logs.append(f"   - $V_u ({Vu_calc:.0f}) > V_c ({Vc:.0f})$ $\\rightarrow$ **Need Stirrups**")
        logs.append(f"   - $V_s = V_u - V_c = {Vs_req:.0f}$ kg")
        logs.append(f"   - $s_{{req}} = \\frac{{A_v f_y d}}{{V_s}} = \\frac{{2 \\cdot {stirrup_area} \\cdot {stress:.0f} \\cdot {d:.2f}}}{{{Vs_req:.0f}}} = \\mathbf{{{s_req_calc:.2f}}} \\text{{ cm}}$")
        
        if manual_s > 0:
            s_use = manual_s
            logs.append(f"3. **Check Manual Spacing:**")
            status = "✅ Safe" if s_use <= s_req_calc else "❌ Unsafe"
            logs.append(f"   - User Input: @{s_use} cm $\\rightarrow$ {status}")
            result['msg_shear'] = "✅ User Defined" if s_use <= s_req_calc else "⚠️ Spacing > Required"
        else:
            step = 5 if s_req_calc > 15 else 2.5
            s_use = int(step * round(min(s_req_calc, d/2, 60)/step)) or 5
            logs.append(f"3. **Select Spacing:**")
            logs.append(f"   - Round down to: **@{s_use} cm**")
            result['msg_shear'] = "⚠️ Shear Reinf."
    
    else:
        logs.append(f"2. **Stirrup Requirement:**")
        logs.append(f"   - $V_u < V_c$ $\\rightarrow$ Theory: Not required, but use Min.")
        if manual_s > 0:
            s_use = manual_s
            logs.append(f"   - User Input: **@{s_use} cm**")
        else:
            s_use = int(d/2)
            logs.append(f"   - Use Min Spacing ($d/2$): **@{s_use} cm**")
        result['msg_shear'] = "✅ Min Shear"

    result['stirrup_text'] = f"@{s_use} cm"
    result['s_value'] = s_use
    result['logs'] = logs
    return result
