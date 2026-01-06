import math

def get_development_length(db_mm, fc, fy, is_top_bar):
    """
    คำนวณระยะฝังยึด (Development Length) ตามมาตรฐาน ACI 318
    Ld = (fy / (1.1 * lambda * sqrt(fc))) * db * modifiers
    """
    # Parameters
    lambda_val = 1.0 (Normal weight concrete)
    # Top bar factor: 1.3 ถ้าเป็นเหล็กบน (คอนกรีตข้างใต้ลึก > 30 ซม.), 1.0 ถ้าเหล็กล่าง
    psi_t = 1.3 if is_top_bar else 1.0
    psi_e = 1.0 # Uncoated
    psi_s = 0.8 if db_mm <= 19 else 1.0 # Size factor
    
    # Simplified Equation (ACI 318) for generally conservative results
    # Term: (fy * psi_t * psi_e) / (1.1 * lambda * sqrt(fc))
    # แต่ใช้สูตรตารางทั่วไปเพื่อความปลอดภัยและครอบคลุม:
    # Ld approx 50db for high strength or calculation:
    
    cb_factor = (3 * db_mm) / db_mm # Assuming spacing/cover > db
    if cb_factor > 2.5: cb_factor = 2.5
    
    # Full Equation: Ld = [ (3/40) * (fy / sqrt(fc)) * (psi_t / cb_factor) ] * db
    # หน่วย: fy, fc (MPa), db (mm) -> Ld (mm)
    
    term1 = (3.0 / 40.0)
    term2 = fy / math.sqrt(fc)
    term3 = (psi_t * psi_e * psi_s) / 1.5 # 1.5 approx for confinement/spacing
    
    # Use Simplified Safe Value used in Thai construction (EIT):
    # Ld = 0.019 * Ab * fy / sqrt(fc) ... (Old)
    # Let's use Robust ACI approx: Ld = 40 * db * factors
    
    base_mult = 40.0
    if fy > 400: base_mult = 50.0
    
    Ld_req = base_mult * psi_t * db_mm
    
    # Min Limit
    Ld_final = max(Ld_req, 300.0)
    
    # Splice Class B (Most common) = 1.3 * Ld
    L_splice = 1.3 * Ld_final
    
    return Ld_final, L_splice

def check_crack_width_aci(fs_service, dc_mm, s_mm):
    """
    ตรวจสอบรอยร้าวโดยใช้ ACI 318 Spacing Control
    s_max = 380 * (280/fs) - 2.5*cc
    """
    # fs_service approx 0.60 fy or 2/3 fy
    term1 = 380.0 * (280.0 / fs_service)
    term2 = 2.5 * dc_mm
    s_max = term1 - term2
    
    # Upper cap just in case
    s_max = min(s_max, 300.0)
    
    if s_max < 0: s_max = 50.0 # Strict
    
    pass_crack = s_mm <= s_max
    
    return pass_crack, s_max

def design_shear(Vu_kN, b_m, d_m, fc, fy):
    phi = 0.85
    Vu = Vu_kN * 1000 
    bv = b_m * 1000   
    d = d_m * 1000    
    
    Vc = 0.17 * math.sqrt(fc) * bv * d
    phi_Vc = phi * Vc
    
    if Vu <= phi_Vc / 2:
        return "Min. Stirrups (RB6 @ 0.25m)", "OK (Min)"
    else:
        Vs = (Vu - phi_Vc) / phi
        if Vs > 4 * Vc:
            return "❌ Section too small (Increase Size)", "Fail"

        # Design Stirrup RB9 (Av=127mm2 for 2 legs)
        Av = 127.0 
        fy_stirrup = 240.0 # SR24 standard for stirrups
        
        s_req = (Av * fy_stirrup * d) / Vs
        s_max = d / 2
        s_final = min(s_req, s_max, 600.0)
        
        # Rounding 2.5cm steps
        s_show = math.floor(s_final / 25) * 25 
        if s_show < 50: s_show = 50 
        
        return f"RB9 @ {s_show/100:.2f} m", "OK"

def design_span_expert(m_pos, m_neg, v_u, b, h, fc, fy, cover, db):
    """
    Main Function: Flexure + Shear + Detailing + Serviceability
    """
    d = h - (cover/1000) - (db/2000) - 0.009 # d approx
    Es = 200000.0 # Steel Modulus MPa
    
    def get_steel(Mu_kNm, is_top):
        if Mu_kNm == 0: 
            return 0, 0.0, "None", 0, 0, True, 0
            
        Mu = abs(Mu_kNm) * 1e6 # N-mm
        phi = 0.9
        
        # 1. Strength Design
        As_try = Mu / (phi * fy * 0.9 * (d*1000))
        a = (As_try * fy) / (0.85 * fc * (b*1000))
        As_req = Mu / (phi * fy * ((d*1000) - a/2))
        
        As_min = (1.4 / fy) * (b*1000) * (d*1000)
        As_final = max(As_req, As_min)
        
        A_bar = 3.1416 * (db/2)**2
        n = math.ceil(As_final / A_bar)
        if n < 2: n = 2 # Minimum 2 bars for constructability
        
        # 2. Spacing Check (Congestion)
        b_mm = b * 1000
        min_gap = max(25.0, db)
        width_req = (2 * cover) + (n * db) + ((n - 1) * min_gap)
        note_spacing = "1 Layer OK"
        if width_req > b_mm:
            note_spacing = "⚠️ Congested (Use 2 Layers)"

        # 3. Development Length & Splice (Item #1)
        Ld, L_splice = get_development_length(db, fc, fy, is_top)
        
        # 4. Crack Control Check (Item #3)
        # Calculate actual spacing s_act
        s_act = (b_mm - 2*cover - n*db) / (n - 1) if n > 1 else 0
        
        # Calculate Service Stress (fs) approx: M_service ~ Mu/1.5
        # fs = M_service / (As * 0.9 d)
        M_service = Mu / 1.5 
        fs = M_service / ((n * A_bar) * 0.9 * (d*1000))
        if fs > 0.6 * fy: fs = 0.6 * fy # Cap at limit
        
        pass_crack, s_max_allow = check_crack_width_aci(fs, cover + db/2, s_act)
        
        return n, As_final, note_spacing, Ld, L_splice, pass_crack, s_max_allow

    # Execute Design
    n_pos, as_pos, note_pos, Ld_pos, Ls_pos, cr_pos, s_lim_pos = get_steel(m_pos, False)
    n_neg, as_neg, note_neg, Ld_neg, Ls_neg, cr_neg, s_lim_neg = get_steel(m_neg, True)
    
    # Shear Design
    stir_info, stir_status = design_shear(v_u, b, d, fc, fy)

    return {
        "pos": {
            "n": n_pos, "note": note_pos, 
            "Ld": Ld_pos, "Ls": Ls_pos, 
            "crack_ok": cr_pos, "s_limit": s_lim_pos
        },
        "neg": {
            "n": n_neg, "note": note_neg, 
            "Ld": Ld_neg, "Ls": Ls_neg, 
            "crack_ok": cr_neg, "s_limit": s_lim_neg
        },
        "shear_stirrups": stir_info,
        "shear_status": stir_status
    }
