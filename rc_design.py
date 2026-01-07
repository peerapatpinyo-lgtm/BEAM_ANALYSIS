import math

def get_development_length(db_mm, fc, fy, is_top_bar):
    psi_t = 1.3 if is_top_bar else 1.0
    base_mult = 40.0 if fy <= 400 else 50.0
    Ld_req = base_mult * psi_t * db_mm
    Ld_final = max(Ld_req, 300.0)
    return Ld_final, 1.3 * Ld_final

def check_crack_width_aci(fs_service, dc_mm, s_mm):
    if fs_service <= 0: return True, 300.0
    s_max = 380.0 * (280.0 / fs_service) - 2.5 * dc_mm
    s_max = min(s_max, 300.0)
    if s_max < 50: s_max = 50.0 
    return s_mm <= s_max, s_max

def design_shear(Vu_kN, b_m, d_m, fc, fy):
    phi, Vu = 0.85, Vu_kN * 1000 
    bv, d = b_m * 1000, d_m * 1000      
    Vc = 0.17 * math.sqrt(fc) * bv * d
    phi_Vc = phi * Vc
    
    if Vu <= phi_Vc / 2: return "Min. Stirrups (RB6 @ 0.25m)", "OK (Min)"
    else:
        Vs = (Vu - phi_Vc) / phi
        if Vs < 0: Vs = 0 
        if Vs > 4 * Vc: return "❌ Section too small", "Fail"
        
        Av, fy_s = 127.0, 240.0 # RB9 (2 legs)
        s_req = 600 if Vs == 0 else (Av * fy_s * d) / Vs
        s_final = min(s_req, d/2, 600.0)
        s_show = math.floor(s_final / 25) * 25 
        return f"RB9 @ {max(s_show, 50)/100:.2f} m", "OK"

def design_span_expert(m_pos, m_neg, v_u, b, h, fc, fy, cover, db):
    d = h - (cover/1000) - (db/2000) - 0.009
    A_bar = 3.1416 * (db/2)**2
    
    def get_steel(Mu_kNm, is_top):
        if Mu_kNm == 0: return 0, 0.0, "None", 0, 0, True, 0, 0.0
        Mu = abs(Mu_kNm) * 1e6
        phi = 0.9
        
        denom = (phi * fy * 0.9 * (d*1000))
        if denom == 0: return 0, 0, "Error", 0, 0, False, 0, 0.0
        
        As_try = Mu / denom
        a = (As_try * fy) / (0.85 * fc * (b*1000))
        As_req = Mu / (phi * fy * ((d*1000) - a/2))
        As_min = (1.4 / fy) * (b*1000) * (d*1000)
        As_final = max(As_req, As_min)
        
        n = max(2, math.ceil(As_final / A_bar))
        
        As_provided = n * A_bar
        a_prov = (As_provided * fy) / (0.85 * fc * (b*1000))
        Mn_prov = As_provided * fy * ((d*1000) - a_prov/2)
        
        b_mm, min_gap = b * 1000, max(25.0, db)
        width_req = (2 * cover) + (n * db) + ((n - 1) * min_gap)
        note = "1 Layer OK" if width_req <= b_mm else "⚠️ Congested"
        
        Ld, Ls = get_development_length(db, fc, fy, is_top)
        
        s_act = (b_mm - 2*cover - n*db) / (n - 1) if n > 1 else 0
        fs = (Mu / 1.5) / (As_provided * 0.9 * (d*1000)) if As_provided > 0 else 0
        pass_crack, s_max = check_crack_width_aci(min(fs, 0.6*fy), cover + db/2, s_act)
        
        return n, As_final, note, Ld, Ls, pass_crack, s_max, (phi * Mn_prov) / 1e6

    n_pos, as_pos, note_pos, Ld_pos, Ls_pos, cr_pos, s_lim_pos, cap_pos = get_steel(m_pos, False)
    n_neg, as_neg, note_neg, Ld_neg, Ls_neg, cr_neg, s_lim_neg, cap_neg = get_steel(m_neg, True)
    stir_info, stir_status = design_shear(v_u, b, d, fc, fy)

    return {
        "pos": {"n": n_pos, "note": note_pos, "Ld": Ld_pos, "Ls": Ls_pos, "crack_ok": cr_pos, "s_limit": s_lim_pos, "capacity": cap_pos},
        "neg": {"n": n_neg, "note": note_neg, "Ld": Ld_neg, "Ls": Ls_neg, "crack_ok": cr_neg, "s_limit": s_lim_neg, "capacity": cap_neg},
        "shear_stirrups": stir_info, "shear_status": stir_status
    }

# --- NEW: BBS & BOQ Functions ---
def get_steel_weight(diameter_mm):
    return (diameter_mm ** 2) / 162.0

def generate_bbs(design_res, spans, b, h, cover_mm):
    bbs_data = []
    for i, res in enumerate(design_res):
        span_len = spans[i]
        # Bottom
        n_bot = res['pos']['n']
        if n_bot > 0:
            len_bar = span_len + 0.6 
            bbs_data.append({"Span": f"Span {i+1}", "Position": "Bot (Main)", "Bar": "DB16", "No. of Bars": n_bot, "Length (m)": round(len_bar,2), "Total Wt (kg)": round(n_bot*len_bar*get_steel_weight(16),2)})
        # Top
        n_top = res['neg']['n']
        if n_top > 0:
            len_bar = (span_len / 3.0) * 2 
            bbs_data.append({"Span": f"Span {i+1}", "Position": "Top (Sup)", "Bar": "DB16", "No. of Bars": n_top, "Length (m)": round(len_bar,2), "Total Wt (kg)": round(n_top*len_bar*get_steel_weight(16),2)})
    
    # Stirrups
    cover = cover_mm / 1000.0
    len_stir = (2 * (b - 2*cover)) + (2 * (h - 2*cover)) + 0.15 
    for i, res in enumerate(design_res):
        try:
            spacing = float(res['shear_stirrups'].split('@')[1].replace('m','').strip())
            n_stir = int(spans[i] / spacing) + 1
            bbs_data.append({"Span": f"Span {i+1}", "Position": "Stirrup", "Bar": "RB9", "No. of Bars": n_stir, "Length (m)": round(len_stir,2), "Total Wt (kg)": round(n_stir*len_stir*get_steel_weight(9),2)})
        except: pass
    return bbs_data

def get_boq(spans, b, h, bbs_data):
    vol_conc = sum(spans) * b * h
    w_steel = sum([item['Total Wt (kg)'] for item in bbs_data])
    return vol_conc, w_steel
