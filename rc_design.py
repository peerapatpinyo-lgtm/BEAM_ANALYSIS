import numpy as np

def design_beam_flexure(Mu, b_m, h_m, cover_mm, db_main_mm, db_stir_mm, fc, fy, phi=0.9):
    """
    Calculates required steel Area (As) and Number of Bars with detailed LaTeX steps.
    Now calculates 'd' accurately based on cover and bar sizes.
    """
    # 1. Unit Conversion & Geometry
    Mu_Nmm = abs(Mu) * 1e6
    b = b_m * 1000
    h = h_m * 1000
    
    # Calculate Effective Depth (d)
    # d = h - cover - stirrup - main/2
    d = h - cover_mm - db_stir_mm - (db_main_mm / 2)
    
    steps = []
    steps.append(r"\textbf{1. Design Parameters}")
    steps.append(rf"M_u = {abs(Mu):.2f} \text{{ kNm}}, \quad f'_c = {fc} \text{{ MPa}}, \quad f_y = {fy} \text{{ MPa}}")
    steps.append(rf"b = {b:.0f} \text{{ mm}}, \quad h = {h:.0f} \text{{ mm}}, \quad \text{{Cover}} = {cover_mm} \text{{ mm}}")
    steps.append(rf"\text{{Main DB}}{db_main_mm}, \quad \text{{Stirrup RB/DB}}{db_stir_mm}")
    steps.append(rf"d = {h:.0f} - {cover_mm} - {db_stir_mm} - {db_main_mm}/2 = \mathbf{{{d:.1f} \text{{ mm}}}}")

    # 2. Beta1
    if fc <= 30: beta1 = 0.85
    elif fc >= 55: beta1 = 0.65
    else: beta1 = 0.85 - 0.05 * (fc - 30) / 7
    
    # 3. Rho Limits
    rho_min_1 = 0.25 * np.sqrt(fc) / fy
    rho_min_2 = 1.4 / fy
    rho_min = max(rho_min_1, rho_min_2)
    
    rho_b = 0.85 * beta1 * (fc / fy) * (600 / (600 + fy))
    rho_max = 0.75 * rho_b # Maximum allowed rho
    
    # 4. Calculate Rn & Rho Required
    steps.append(r"\textbf{2. Flexural Calculation}")
    Rn = Mu_Nmm / (phi * b * d**2)
    steps.append(rf"R_n = \frac{{{Mu_Nmm:.0f}}}{{0.9 \cdot {b:.0f} \cdot {d:.1f}^2}} = {Rn:.3f} \text{{ MPa}}")

    try:
        term_in_sqrt = 1 - (2 * Rn) / (0.85 * fc)
        if term_in_sqrt < 0:
            steps.append(r"\color{red}{\textbf{Fail: Section too small! Increase Depth.}}")
            return {'status': 'Fail', 'msg': 'Section Small'}, steps
            
        rho_req = (0.85 * fc / fy) * (1 - np.sqrt(term_in_sqrt))
    except:
        return {'status': 'Error', 'msg': 'Calc Error'}, steps
        
    # 5. Check Logic
    final_rho = rho_req
    status = "OK"
    
    if rho_req < rho_min:
        final_rho = rho_min
        steps.append(rf"\rho_{{req}} ({rho_req:.5f}) < \rho_{{min}} \rightarrow \text{{Use }} \rho_{{min}} = {rho_min:.5f}")
    elif rho_req > rho_max:
        status = "Warning"
        steps.append(rf"\rho_{{req}} ({rho_req:.5f}) > \rho_{{max}} \rightarrow \text{{Warning: Over-Reinforced}}")
    else:
        steps.append(rf"\rho = {rho_req:.5f} \quad (\text{{OK}})")

    # 6. Calculate Area & Number of Bars
    As_req = final_rho * b * d
    
    # Area of one bar
    A_bar = 3.14159 * (db_main_mm / 2)**2
    num_bars = np.ceil(As_req / A_bar)
    if num_bars < 2: num_bars = 2
    
    As_prov = num_bars * A_bar
    
    steps.append(r"\textbf{3. Reinforcement}")
    steps.append(rf"A_{{s,req}} = {final_rho:.5f} \cdot {b:.0f} \cdot {d:.1f} = {As_req:.2f} \text{{ mm}}^2")
    steps.append(rf"\text{{Use }} \mathbf{{{int(num_bars)} \text{{ - DB}} {db_main_mm}}} \quad (A_{{s,prov}} = {As_prov:.2f} \text{{ mm}}^2)")

    return {
        'As_req': As_req,
        'As_prov': As_prov,
        'n_bars': int(num_bars),
        'rho': final_rho,
        'd_used': d,
        'status': status
    }, steps

def check_shear(Vu, b_m, d_mm, fc, fy, db_stir_mm, phi=0.85):
    """
    Calculates stirrup spacing based on user selected stirrup size.
    """
    Vu_N = abs(Vu) * 1000
    b = b_m * 1000
    d = d_mm # d passed from flexure calculation for consistency
    
    steps = []
    steps.append(r"\textbf{Shear Design (Stirrups)}")
    
    # Vc Calculation
    Vc = 0.17 * np.sqrt(fc) * b * d
    phi_Vc = phi * Vc
    
    steps.append(rf"V_u = {abs(Vu):.2f} \text{{ kN}}, \quad \phi V_c = {phi_Vc/1000:.2f} \text{{ kN}}")
    
    req_s = 0
    status_msg = ""
    
    # Av Calculation (2 legs)
    Av = 2 * (3.14159 * (db_stir_mm/2)**2)
    
    if Vu_N <= phi_Vc / 2:
        status_msg = "Not Req."
        req_s = d / 2
        steps.append(r"V_u \le 0.5 \phi V_c \rightarrow \text{Theoretically None (Use Min)}")
    
    elif Vu_N <= phi_Vc:
        status_msg = "Min Stirrups"
        # Min spacing (simplified)
        s1 = (Av * fy) / (0.062 * np.sqrt(fc) * b)
        s2 = (Av * fy) / (0.35 * b)
        req_s = min(s1, s2, d/2, 600)
        steps.append(r"0.5 \phi V_c < V_u \le \phi V_c \rightarrow \text{Min Stirrups}")
        
    else:
        # Design Stirrups
        Vs = (Vu_N - phi_Vc) / phi
        steps.append(rf"V_s = \frac{{{Vu_N:.0f} - {phi_Vc:.0f}}}{{{phi}}} = {Vs/1000:.2f} \text{{ kN}}")
        
        # Check Max Capacity
        if Vs > (0.66 * np.sqrt(fc) * b * d):
            steps.append(r"\color{red}{\textbf{Fail: V_s exceeds limit. Increase Size.}}")
            return 0, "Fail", steps

        s_calc = (Av * fy * d) / Vs
        
        # Max Spacing
        if Vs <= (0.33 * np.sqrt(fc) * b * d):
            s_max = min(d/2, 600)
        else:
            s_max = min(d/4, 300)
            
        req_s = min(s_calc, s_max)
        steps.append(rf"\text{{Try }} \text{{RB/DB}}{db_stir_mm} (A_v={Av:.1f}), \quad s_{{req}} = {req_s:.0f} \text{{ mm}}")
        status_msg = "Req. Calc"

    # Practical Rounding (10mm or 25mm steps)
    if req_s > 300: req_s = 300
    if req_s < 50: req_s = 50
    s_final = int(req_s // 10) * 10
    
    steps.append(rf"\textbf{{Use }} \mathbf{{\text{{RB/DB}}{db_stir_mm} @ {s_final} \text{{ mm}} c/c}}")
    
    return s_final, status_msg, steps
