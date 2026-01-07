import math

def get_beta1(fc):
    """Calculate Beta1 factor per ACI 318"""
    if fc <= 28:
        return 0.85
    elif fc >= 55:
        return 0.65
    else:
        return 0.85 - 0.05 * (fc - 28) / 7

def solve_steel_section(Mu_kNm, b_m, d_m, fc, fy, is_top=False):
    """
    Calculate required steel area with detailed steps and size recommendation
    """
    logs = [] 
    
    # 1. Init
    Mu = abs(Mu_kNm) * 1e6 # N-mm
    b = b_m * 1000 # mm
    d = d_m * 1000 # mm
    phi = 0.9 
    beta1 = get_beta1(fc)
    
    header = "Top Reinforcement (Negative Moment)" if is_top else "Bottom Reinforcement (Positive Moment)"
    logs.append(f"**{header}**")
    logs.append(f"- Design Moment, Mu = {abs(Mu_kNm):.2f} kNm")

    if Mu == 0:
        logs.append("- Mu = 0, Use Minimum Reinforcement.")
        return 0, 0, 0, logs

    # 2. Limit State Check (Max Reinforcement / Section Size)
    # Calculate rho_max (Tension controlled limit, strain=0.005)
    # rho_bal = (0.85 * beta1 * fc / fy) * (600 / (600 + fy))
    # Using 0.005 strain limit directly (approx 0.375 beta1 ...) usually safer
    # But let's use the standard ACI max ratio:
    rho_bal = (0.85 * beta1 * fc / fy) * (600 / (600 + fy))
    rho_max = 0.75 * rho_bal # Common practice limit
    
    # 3. Calculate Rn required
    Rn = Mu / (phi * b * d**2)
    logs.append(f"- Rn (Required) = {Rn:.3f} MPa")
    
    # Check if Section is adequate
    # Condition: 1 - (2Rn / 0.85fc) must be >= 0
    check_val = 1 - (2 * Rn) / (0.85 * fc)
    
    if check_val < 0:
        logs.append(f"❌ **Error: Section too small!** (Concrete Crushing Risk)")
        
        # --- RECOMMENDED SIZE CALCULATION ---
        # Back-calculate required d from rho_max
        # Rn_max corresponds to rho_max
        m = fy / (0.85 * fc)
        Rn_max = rho_max * fy * (1 - 0.5 * rho_max * m)
        
        # d_req = sqrt( Mu / (phi * b * Rn_max) )
        d_req = math.sqrt(Mu / (phi * b * Rn_max))
        
        # Estimate total h (d + cover + stirrup + half_bar)
        # approx cover 40 + stirrup 9 + half_bar 8 = 57mm -> say 60mm
        h_req = d_req + 60 
        
        # Round up to nearest 5 cm
        h_suggest = math.ceil(h_req / 50) * 50 / 1000.0 # convert to m
        
        logs.append(f"💡 **Suggestion:**")
        logs.append(f"   For width b = {b_m:.2f} m:")
        logs.append(f"   Minimum effective depth (d) should be **{d_req:.0f} mm**")
        logs.append(f"   Try increasing Height (h) to **{h_suggest:.2f} m**")
        
        return 0, 0, 0, logs

    # 4. If OK, Calculate Steel Area
    rho_req = (0.85 * fc / fy) * (1 - math.sqrt(check_val))
    As_req = rho_req * b * d
    
    # Check Min Steel
    As_min1 = (0.25 * math.sqrt(fc) / fy) * b * d
    As_min2 = (1.4 / fy) * b * d
    As_min = max(As_min1, As_min2)
    
    As_final = max(As_req, As_min)
    logs.append(f"- As,req = {As_req:.1f} mm² (As,min = {As_min:.1f})")

    # 5. Bar Selection
    db = 16 
    A_bar = 3.1416 * (db/2)**2
    n_bars = max(2, math.ceil(As_final / A_bar))
    As_prov = n_bars * A_bar
    
    logs.append(f"👉 **Select {n_bars}-DB{db}** (As_prov = {As_prov:.1f} mm²)")
    
    # 6. Capacity Check
    a = (As_prov * fy) / (0.85 * fc * b)
    Mn = As_prov * fy * (d - a/2)
    phi_Mn = 0.9 * Mn / 1e6
    
    logs.append(f"- Capacity $\phi M_n$ = {phi_Mn:.2f} kNm")
    
    return n_bars, As_final, phi_Mn, logs

def design_shear_detailed(Vu_kN, b_m, d_m, fc, fy):
    """
    Detailed Shear Design Calculation
    """
    logs = []
    logs.append("**Shear Reinforcement Design**")
    
    Vu = abs(Vu_kN) * 1000 # N
    b = b_m * 1000
    d = d_m * 1000
    phi = 0.85 # Shear
    
    logs.append(f"- Design Shear, Vu = {abs(Vu_kN):.2f} kN")
    
    # Concrete Capacity
    Vc = 0.17 * math.sqrt(fc) * b * d
    phi_Vc = phi * Vc
    
    logs.append(f"- Concrete Capacity, $\phi V_c$ = {phi_Vc/1000:.2f} kN")
    
    stirrup_info = ""
    status = ""
    
    if Vu <= phi_Vc / 2:
        logs.append("- $V_u < \phi V_c / 2$ -> Theoretically no stirrups required.")
        logs.append("👉 Provide Minimum Stirrups for stability.")
        stirrup_info = "RB6 @ 0.25 m"
        status = "OK (Min)"
        
    elif Vu <= phi_Vc:
        logs.append("- $\phi V_c / 2 < V_u \leq \phi V_c$ -> Provide Minimum Reinforcement.")
        # Av min check
        s_max = d / 2
        Av = 2 * (3.1416 * 4.5**2) # RB9 2 legs (using RB9 as standard min)
        # Simplified min selection
        stirrup_info = "RB9 @ 0.20 m" 
        status = "OK (Min)"
        
    else:
        # Stirrups required
        Vs = (Vu - phi_Vc) / phi
        logs.append(f"- Stirrups Required, $V_s$ = {Vs/1000:.2f} kN")
        
        # Max check
        if Vs > 4 * Vc:
            logs.append("❌ **Error:** $V_s$ too high (> $4V_c$). Increase Section Size.")
            return "Fail", "Fail", logs
            
        # Spacing Calculation
        Av = 2 * (3.1416 * (9/2)**2) # RB9 (2 legs) = 127 mm2
        fy_v = 240 # RB9 usually SR24
        
        s_req = (Av * fy_v * d) / Vs
        logs.append(f"- Required Spacing, s = {s_req:.1f} mm (using RB9)")
        
        # Max spacing limits
        if Vs <= 2 * Vc:
            s_max = min(600, d/2)
        else:
            s_max = min(300, d/4)
            
        logs.append(f"- Max Spacing, $s_{{max}}$ = {s_max:.1f} mm")
        
        s_final = min(s_req, s_max)
        s_show = math.floor(s_final / 25) * 25 # Round down to nearest 25mm
        if s_show < 50: s_show = 50
        
        stirrup_info = f"RB9 @ {s_show/1000:.2f} m"
        logs.append(f"👉 **Select {stirrup_info}**")
        status = "OK"

    return stirrup_info, status, logs

def design_span_expert(m_pos, m_neg, v_u, b, h, fc, fy, cover, db):
    # Effective Depth (approx)
    d = h - (cover/1000) - (db/2000) - 0.009 # stirrup 9mm
    
    # 1. Design Positive Moment
    n_pos, as_pos, cap_pos, log_pos = solve_steel_section(m_pos, b, d, fc, fy, is_top=False)
    
    # 2. Design Negative Moment
    n_neg, as_neg, cap_neg, log_neg = solve_steel_section(m_neg, b, d, fc, fy, is_top=True)
    
    # 3. Design Shear
    stir_info, stir_status, log_shear = design_shear_detailed(v_u, b, d, fc, fy)
    
    return {
        "pos": {"n": n_pos, "capacity": cap_pos, "logs": log_pos},
        "neg": {"n": n_neg, "capacity": cap_neg, "logs": log_neg},
        "shear_stirrups": stir_info, 
        "shear_status": stir_status,
        "shear_logs": log_shear,
        "db": db
    }

# --- BBS & BOQ Functions (คงเดิมไว้ หรือใส่รวมในนี้) ---
def get_steel_weight(diameter_mm):
    return (diameter_mm ** 2) / 162.0

def generate_bbs(design_res, spans, b, h, cover_mm):
    # (ใช้ Code BBS เดิมจากคำตอบก่อนหน้าได้เลยครับ)
    bbs_data = []
    # ... Paste BBS logic here if needed ...
    return bbs_data

def get_boq(spans, b, h, bbs_data):
    # (ใช้ Code BOQ เดิม)
    return 0, 0
