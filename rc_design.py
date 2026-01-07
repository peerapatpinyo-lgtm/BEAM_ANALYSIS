import math

def get_development_length(db_mm, fc, fy, is_top_bar):
    lambda_val = 1.0 
    psi_t = 1.3 if is_top_bar else 1.0
    psi_e = 1.0
    psi_s = 0.8 if db_mm <= 19 else 1.0
    
    cb_factor = 2.5 # Simplified conservative
    
    base_mult = 40.0 if fy <= 400 else 50.0
    Ld_req = base_mult * psi_t * db_mm
    Ld_final = max(Ld_req, 300.0)
    L_splice = 1.3 * Ld_final
    return Ld_final, L_splice

def check_crack_width_aci(fs_service, dc_mm, s_mm):
    if fs_service <= 0: return True, 300.0
    term1 = 380.0 * (280.0 / fs_service)
    term2 = 2.5 * dc_mm
    s_max = min(term1 - term2, 300.0)
    if s_max < 50: s_max = 50.0
    return s_mm <= s_max, s_max

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
        if Vs < 0: Vs = 0 
        
        if Vs > 4 * Vc:
            return "❌ Section too small (Increase Size)", "Fail"

        # Design Stirrup RB9 (Av=127mm2 for 2 legs)
        Av = 127.0 
        fy_stirrup = 240.0
        
        if Vs == 0: 
             s_req = 600
        else:
             s_req = (Av * fy_stirrup * d) / Vs
             
        s_max = d / 2
        s_final = min(s_req, s_max, 600.0)
        s_show = math.floor(s_final / 25) * 25 
        if s_show < 50: s_show = 50 
        
        return f"RB9 @ {s_show/100:.2f} m", "OK"

def design_span_expert(m_pos, m_neg, v_u, b, h, fc, fy, cover, db):
    d = h - (cover/1000) - (db/2000) - 0.009
    A_bar = 3.1416 * (db/2)**2
    
    def get_steel(Mu_kNm, is_top):
        if Mu_kNm == 0: 
            return 0, 0.0, "None", 0, 0, True, 0, 0.0
            
        Mu = abs(Mu_kNm) * 1e6
        phi = 0.9
        
        # Strength
        As_min = (1.4 / fy) * (b*1000) * (d*1000)
        denom = (phi * fy * 0.9 * (d*1000))
        if denom == 0: return 0, 0, "Error", 0, 0, False, 0, 0.0
        
        As_try = Mu / denom
        a = (As_try * fy) / (0.85 * fc * (b*1000))
        As_req = Mu / (phi * fy * ((d*1000) - a/2))
        As_final = max(As_req, As_min)
        
        n = math.ceil(As_final / A_bar)
        if n < 2: n = 2
        
        # Capacity
        As_provided = n * A_bar
        a_prov = (As_provided * fy) / (0.85 * fc * (b*1000))
        Mn_prov = As_provided * fy * ((d*1000) - a_prov/2)
        Phi_Mn_prov_kNm = (phi * Mn_prov) / 1e6
        
        # Spacing
        b_mm = b * 1000
        min_gap = max(25.0, db)
        width_req = (2 * cover) + (n * db) + ((n - 1) * min_gap)
        note_spacing = "1 Layer OK" if width_req <= b_mm else "⚠️ Congested"

        # Development
        Ld, L_splice = get_development_length(db, fc, fy, is_top)
        
        # Crack Control
        s_act = 0
        if n > 1: s_act = (b_mm - 2*cover - n*db) / (n - 1)
        M_service = Mu / 1.5 
        fs = M_service / (As_provided * 0.9 * (d*1000))
        if fs > 0.6 * fy: fs = 0.6 * fy 
        pass_crack, s_max_allow = check_crack_width_aci(fs, cover + db/2, s_act)
        
        return n, As_final, note_spacing, Ld, L_splice, pass_crack, s_max_allow, Phi_Mn_prov_kNm

    n_pos, as_pos, note_pos, Ld_pos, Ls_pos, cr_pos, s_lim_pos, cap_pos = get_steel(m_pos, False)
    n_neg, as_neg, note_neg, Ld_neg, Ls_neg, cr_neg, s_lim_neg, cap_neg = get_steel(m_neg, True)
    
    stir_info, stir_status = design_shear(v_u, b, d, fc, fy)

    return {
        "pos": {"n": n_pos, "note": note_pos, "Ld": Ld_pos, "Ls": Ls_pos, "crack_ok": cr_pos, "s_limit": s_lim_pos, "capacity": cap_pos},
        "neg": {"n": n_neg, "note": note_neg, "Ld": Ld_neg, "Ls": Ls_neg, "crack_ok": cr_neg, "s_limit": s_lim_neg, "capacity": cap_neg},
        "shear_stirrups": stir_info,
        "shear_status": stir_status
    }
