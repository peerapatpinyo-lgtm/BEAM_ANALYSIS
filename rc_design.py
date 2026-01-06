import numpy as np

def calculate_advanced_rc(mu_pos, mu_neg, vu, b_m, h_m, fc, fy=400):
    """
    mu_pos: Max positive moment (kNm) - สำหรับเหล็กล่าง
    mu_neg: Max negative moment (kNm) - สำหรับเหล็กบน
    vu: Max shear force (kN)
    b_m, h_m: width, height in meters
    """
    phi_m, phi_v = 0.90, 0.75
    b, h = b_m * 1000, h_m * 1000  # แปลงเป็น mm
    d = h - 50  # Covering 50mm
    report = []

    def calc_as(mu_knm):
        if mu_knm <= 0: return 0, 0, "No reinforcement needed"
        mu_n_mm = (mu_knm * 1e6) / phi_m
        rn = mu_n_mm / (b * d**2)
        m = fy / (0.85 * fc)
        
        # Check if section can handle moment
        check_val = 1 - (2 * m * rn / fy)
        if check_val < 0: return 0, 0, "⚠️ Section too small! (Over-reinforced)"
        
        rho = (1/m) * (1 - np.sqrt(check_val))
        rho_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy)
        rho_final = max(rho, rho_min)
        as_req = rho_final * b * d
        return as_req, rho_final, "OK"

    # --- 1. Flexure Design ---
    as_bot, rho_bot, msg_bot = calc_as(abs(mu_pos))
    as_top, rho_top, msg_top = calc_as(abs(mu_neg))

    # --- 2. Shear Design (ACI 318) ---
    vc = (1/6) * np.sqrt(fc) * b * d / 1000  # kN
    vs_req = (vu / phi_v) - vc if vu > (phi_v * vc * 0.5) else 0
    
    spacing = 0
    if vs_req > 0:
        # ใช้เหล็กปลอก 2 ขา RB9 (Asv = 127 mm2)
        asv = 127 
        spacing = (asv * fy * d) / (vs_req * 1000)
        s_max = min(d/2, 600)
        spacing = min(spacing, s_max)
    
    # --- 3. Build Calculation Note ---
    report.append(f"**Section:** {b}x{h} mm, **d:** {d} mm")
    report.append(f"**Material:** fc'={fc} MPa, fy={fy} MPa")
    report.append(f"---")
    report.append(f"**Flexure Design (Mu_max = {max(abs(mu_pos), abs(mu_neg)):.2f} kNm):**")
    report.append(f"- Required As (Bottom): {as_bot:.0f} mm² (ρ={rho_bot:.5f}) - {msg_bot}")
    report.append(f"- Required As (Top): {as_top:.0f} mm² (ρ={rho_top:.5f}) - {msg_top}")
    report.append(f"---")
    report.append(f"**Shear Design (Vu = {vu:.2f} kN):**")
    report.append(f"- Concrete Capacity (φVc): {phi_v*vc:.2f} kN")
    if vs_req > 0:
        report.append(f"- Required Vs: {vs_req:.2f} kN")
        report.append(f"- Stirrup Spacing (RB9): @ {spacing:.0f} mm o.c.")
    else:
        report.append(f"- Shear reinforcement: Provide Minimum Stirrups (RB9 @ 200mm)")

    return {
        'as_top': as_top, 'as_bot': as_bot, 'spacing': spacing,
        'report': report, 'msg_bot': msg_bot, 'msg_top': msg_top
    }
