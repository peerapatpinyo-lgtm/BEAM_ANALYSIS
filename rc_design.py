import numpy as np

def design_beam_flexure(Mu, b, d, fc, fy, phi=0.9):
    """
    Calculates required steel Area (As) with detailed calculation steps (LaTeX).
    Returns: As_req, rho, status_dict, calc_steps (list of latex strings)
    """
    Mu_Nmm = Mu * 1e6
    b_mm = b * 1000
    d_mm = d * 1000
    
    steps = []
    steps.append(r"\textbf{1. Design Parameters}")
    steps.append(f"M_u = {Mu:.2f} \\text{{ kNm}}, \\quad f'_c = {fc} \\text{{ MPa}}, \\quad f_y = {fy} \\text{{ MPa}}")
    steps.append(f"b = {b_mm:.0f} \\text{{ mm}}, \\quad d = {d_mm:.0f} \\text{{ mm}}, \\quad \\phi = {phi}")

    # 1. Beta1
    if fc <= 30: beta1 = 0.85
    elif fc >= 55: beta1 = 0.65
    else: beta1 = 0.85 - 0.05 * (fc - 30) / 7
    
    steps.append(r"\textbf{2. Determine } \beta_1")
    steps.append(rf"\beta_1 = {beta1:.3f} \quad (\text{{for }} f'_c = {fc} \text{{ MPa}})")

    # 2. Rho Limits
    steps.append(r"\textbf{3. Reinforcement Ratio Limits}")
    
    # Rho Min
    rho_min_1 = 0.25 * np.sqrt(fc) / fy
    rho_min_2 = 1.4 / fy
    rho_min = max(rho_min_1, rho_min_2)
    
    steps.append(r"\rho_{min} = \max \left( \frac{0.25\sqrt{f'_c}}{f_y}, \frac{1.4}{f_y} \right)")
    steps.append(rf"\rho_{{min}} = \max \left( \frac{{0.25\sqrt{{{fc}}}}}{{{fy}}}, \frac{{1.4}}{{{fy}}} \right) = \max({rho_min_1:.5f}, {rho_min_2:.5f}) = {rho_min:.5f}")

    # Rho Bal & Max
    rho_b = 0.85 * beta1 * (fc / fy) * (600 / (600 + fy))
    rho_max = 0.75 * rho_b
    
    steps.append(r"\rho_{b} = 0.85 \beta_1 \frac{f'_c}{f_y} \left( \frac{600}{600 + f_y} \right)")
    steps.append(rf"\rho_{{b}} = 0.85 ({beta1:.3f}) \frac{{{fc}}}{{{fy}}} \left( \frac{{600}}{{600 + {fy}}} \right) = {rho_b:.5f}")
    steps.append(rf"\rho_{{max}} = 0.75 \rho_b = 0.75 \times {rho_b:.5f} = {rho_max:.5f}")

    # 3. Calculate Rn
    steps.append(r"\textbf{4. Required Reinforcement}")
    Rn = Mu_Nmm / (phi * b_mm * d_mm**2)
    steps.append(r"R_n = \frac{M_u}{\phi b d^2}")
    steps.append(rf"R_n = \frac{{{Mu_Nmm:.0f}}}{{{phi} \cdot {b_mm} \cdot {d_mm}^2}} = {Rn:.4f} \text{{ MPa}}")

    # 4. Calculate Rho Required
    try:
        term_in_sqrt = 1 - (2 * Rn) / (0.85 * fc)
        if term_in_sqrt < 0:
            steps.append(r"\textbf{Error: Section too small!}")
            steps.append(rf"1 - \frac{{2 R_n}}{{0.85 f'_c}} < 0 \rightarrow \text{{Fail}}")
            return 0, 0, {"status": "Fail", "msg": "Compression Fail"}, steps
            
        rho_req = (0.85 * fc / fy) * (1 - np.sqrt(term_in_sqrt))
        
        steps.append(r"\rho_{req} = \frac{0.85 f'_c}{f_y} \left( 1 - \sqrt{1 - \frac{2 R_n}{0.85 f'_c}} \right)")
        steps.append(rf"\rho_{{req}} = \frac{{0.85 ({fc})}}{{{fy}}} \left( 1 - \sqrt{{1 - \frac{{2 ({Rn:.4f})}}{{0.85 ({fc})}}}} \right) = {rho_req:.5f}")

    except:
        return 0, 0, {"status": "Fail", "msg": "Calc Error"}, steps
        
    # 5. Check Logic
    final_rho = rho_req
    status = "OK"
    msg = "Design Pass"
    
    steps.append(r"\textbf{5. Check \& Final Area}")
    
    if rho_req < rho_min:
        steps.append(rf"\rho_{{req}} ({rho_req:.5f}) < \rho_{{min}} ({rho_min:.5f}) \rightarrow \text{{Use }} \rho_{{min}}")
        final_rho = rho_min
        msg = "Used Min Steel"
    elif rho_req > rho_max:
        steps.append(rf"\rho_{{req}} ({rho_req:.5f}) > \rho_{{max}} ({rho_max:.5f}) \rightarrow \text{{Warning: Section Over-Reinforced}}")
        status = "Warning"
        msg = "Exceeds rho_max"
    else:
        steps.append(rf"\rho_{{min}} < \rho_{{req}} < \rho_{{max}} \rightarrow \text{{OK}}")

    As_req = final_rho * b_mm * d_mm
    steps.append(r"A_{s,req} = \rho \cdot b \cdot d")
    steps.append(rf"A_{{s,req}} = {final_rho:.5f} \cdot {b_mm} \cdot {d_mm} = \mathbf{{{As_req:.2f} \text{{ mm}}^2}}")

    return As_req, final_rho, {"status": status, "msg": msg}, steps

def check_shear(Vu, b, d, fc, fy, phi=0.85):
    """
    Returns required stirrup spacing with calculation steps.
    """
    Vu_N = Vu * 1000
    b_mm = b * 1000
    d_mm = d * 1000
    
    steps = []
    steps.append(r"\textbf{Shear Design}")
    steps.append(rf"V_u = {Vu:.2f} \text{{ kN}}, \quad \phi = {phi}")
    
    Vc = 0.17 * np.sqrt(fc) * b_mm * d_mm
    phi_Vc = phi * Vc
    
    steps.append(r"V_c = 0.17 \sqrt{f'_c} b d")
    steps.append(rf"V_c = 0.17 \sqrt{{{fc}}} ({b_mm}) ({d_mm}) = {Vc/1000:.2f} \text{{ kN}}")
    steps.append(rf"\phi V_c = {phi} \times {Vc/1000:.2f} = {phi_Vc/1000:.2f} \text{{ kN}}")
    
    req_s = None
    status = "OK"
    
    if Vu_N <= phi_Vc / 2:
        status = "No Shear Reinf. Needed"
        req_s = 600
        steps.append(r"V_u \le 0.5 \phi V_c \rightarrow \text{Theoreticaly no stirrups needed (Use max spacing)}")
    elif Vu_N <= phi_Vc:
        status = "Min Shear Reinf."
        req_s = 300 
        steps.append(r"0.5 \phi V_c < V_u \le \phi V_c \rightarrow \text{Use Minimum Stirrups}")
    else:
        Vs = (Vu_N - phi_Vc) / phi
        steps.append(r"V_u > \phi V_c \rightarrow \text{Stirrups Required}")
        steps.append(r"V_s = \frac{V_u - \phi V_c}{\phi}")
        steps.append(rf"V_s = \frac{{{Vu_N:.0f} - {phi_Vc:.0f}}}{{{phi}}} = {Vs/1000:.2f} \text{{ kN}}")
        
        # Try RB6 (2 legs) -> Av = 2 * 28 = 56 mm2
        Av = 56.5
        steps.append(r"\text{Try RB6 (2 legs), } A_v \approx 56.5 \text{ mm}^2")
        
        s_req = (Av * fy * d_mm) / Vs
        steps.append(r"s_{req} = \frac{A_v f_y d}{V_s}")
        steps.append(rf"s_{{req}} = \frac{{56.5 \cdot {fy} \cdot {d_mm}}}{{{Vs:.0f}}} = {s_req:.0f} \text{{ mm}}")
        
        req_s = s_req
        
        if Vs > 0.66 * np.sqrt(fc) * b_mm * d_mm:
            status = "Fail (Section too small)"
            req_s = 0
            steps.append(r"\textbf{Fail: } V_s \text{ exceeds max limit}")
            
    return req_s, status, steps
