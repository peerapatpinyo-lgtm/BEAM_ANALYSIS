import math

def design_shear(Vu_kN, b_m, d_m, fc, fy):
    """
    ออกแบบเหล็กปลอก (Stirrups) ตามมาตรฐาน ACI/EIT
    Return: ข้อความสรุประยะห่างเหล็กปลอก
    """
    phi = 0.85
    Vu = Vu_kN * 1000 # Convert to N
    bv = b_m * 1000   # mm
    d = d_m * 1000    # mm
    
    # 1. Concrete Capacity (Vc) = 0.17 * sqrt(fc) * b * d
    Vc = 0.17 * math.sqrt(fc) * bv * d
    phi_Vc = phi * Vc
    
    status = ""
    stirrup_info = ""
    
    # 2. Check Requirement
    if Vu <= phi_Vc / 2:
        status = "OK (No Shear Reinf. Req)"
        stirrup_info = "Min. Stirrups (RB6 or RB9 @ 0.30m)"
    else:
        # Need Stirrups
        # Vs = (Vu - phiVc) / phi
        Vs = (Vu - phi_Vc) / phi
        
        # Check Max Capacity (Vc + Vs limit)
        if Vs > 4 * Vc:
            return "❌ Section too small (Change Size)", "Fail"

        # 3. Calculate Spacing (s) using RB6 (Av = 2 legs * 28mm2 = 56) or RB9
        # Let's assume RB6 (2 legs) -> Av = 56 mm2 (approx) or RB9 (2 legs) = 126 mm2
        # Use RB9 for beam standard -> Av = 2 * 63.6 = 127 mm2
        Av = 127.0 
        fy_stirrup = 240.0 # SR24
        
        # s = (Av * fy * d) / Vs
        s_req = (Av * fy_stirrup * d) / Vs
        
        # Max Spacing Limits
        s_max = d / 2
        s_final = min(s_req, s_max, 600.0) # mm
        
        # Round down to nearest 5cm or 2.5cm
        s_show = math.floor(s_final / 25) * 25 # step 25mm
        if s_show < 50: s_show = 50 # min spacing 5cm
        
        status = "OK"
        stirrup_info = f"RB9 @ {s_show/100:.2f} m" # Convert back to m
        
        if Vu > phi_Vc:
            status += " (Stirrups Required)"
        else:
            status += " (Min Reinforcement)"
            
    return stirrup_info, status

def check_bar_spacing(n_bars, db_mm, b_m, cover_mm):
    """
    ตรวจสอบว่าเหล็กเรียงในชั้นเดียวพอไหม
    """
    b_mm = b_m * 1000
    # Space required = 2*cover + n*db + (n-1)*spacing
    # Min spacing = max(25mm, db)
    min_gap = max(25.0, db_mm)
    
    width_req = (2 * cover_mm) + (n_bars * db_mm) + ((n_bars - 1) * min_gap)
    
    if width_req > b_mm:
        return False, f"⚠️ Too tight! (Req {width_req:.0f}mm > {b_mm:.0f}mm) -> Use 2 Layers"
    return True, "1 Layer OK"

def design_span_expert(m_pos, m_neg, v_u, b, h, fc, fy, cover, db):
    """
    Main Function: ออกแบบครบวงจร (Flexure + Shear + Detailing)
    """
    d = h - (cover/1000) - (db/2000) - 0.009 # approx d (subtract stirrup RB9)
    
    def get_steel(Mu_kNm):
        if Mu_kNm == 0: return 0, 0.0
        Mu = abs(Mu_kNm) * 1e6 # N-mm
        phi = 0.9
        
        # As approx = Mu / (phi * fy * 0.9d)
        As_try = Mu / (phi * fy * 0.9 * (d*1000))
        
        # Iterate for a (depth of stress block)
        # a = (As * fy) / (0.85 * fc * b)
        a = (As_try * fy) / (0.85 * fc * (b*1000))
        As_req = Mu / (phi * fy * ((d*1000) - a/2))
        
        # Min Steel
        As_min = (1.4 / fy) * (b*1000) * (d*1000)
        As_final = max(As_req, As_min)
        
        # Convert to number of bars
        A_bar = 3.1416 * (db/2)**2
        n = math.ceil(As_final / A_bar)
        return n, As_final

    # 1. Flexural Design
    n_pos, as_pos = get_steel(m_pos)
    n_neg, as_neg = get_steel(m_neg)
    
    # 2. Detailing Check (Spacing)
    spacing_ok_pos, note_pos = check_bar_spacing(n_pos, db, b, cover)
    spacing_ok_neg, note_neg = check_bar_spacing(n_neg, db, b, cover)
    
    # 3. Shear Design
    stir_info, stir_status = design_shear(v_u, b, d, fc, fy)

    return {
        "pos": {"n": n_pos, "note": note_pos},
        "neg": {"n": n_neg, "note": note_neg},
        "shear_stirrups": stir_info,
        "shear_status": stir_status
    }
