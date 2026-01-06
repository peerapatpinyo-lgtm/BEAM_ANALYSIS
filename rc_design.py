import numpy as np

def calculate_advanced_rc(mu_pos, mu_neg, vu, b_m, h_m, fc, fy=400):
    phi_m, phi_v = 0.90, 0.75
    b, h = b_m * 1000, h_m * 1000  # mm
    d = h - 50  # mm (Effective depth)
    report = []

    def calc_as(mu_knm, label):
        if abs(mu_knm) < 0.1: 
            return 0, 0, f"{label}: Moment is negligible."
        
        mu_n_mm = (abs(mu_knm) * 1e6) / phi_m
        rn = mu_n_mm / (b * d**2)
        m = fy / (0.85 * fc)
        
        check_val = 1 - (2 * m * rn / fy)
        if check_val < 0:
            return 0, 0, f"⚠️ {label}: Section too small! Over-reinforced."
        
        rho = (1/m) * (1 - np.sqrt(check_val))
        rho_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy)
        rho_final = max(rho, rho_min)
        as_req = rho_final * b * d
        return as_req, rho_final, f"{label}: Design OK"

    # Flexure
    as_bot, rho_bot, msg_bot = calc_as(mu_pos, "Bottom Steel")
    as_top, rho_top, msg_top = calc_as(mu_neg, "Top Steel")

    # Shear (ACI 318)
    vc = (1/6) * np.sqrt(fc) * b * d / 1000 # kN
    vs_req = (vu / phi_v) - vc if vu > (phi_v * vc * 0.5) else 0
    
    spacing = 200 # Default/Min spacing mm
    if vs_req > 0:
        asv = 127 # RB9 @ 2 legs
        spacing = min((asv * fy * d) / (vs_req * 1000), d/2, 600)

    # Compile Detailed Report
    report.append(f"**Material:** fc' = {fc} MPa, fy = {fy} MPa")
    report.append(f"**Section:** b = {b:.0f} mm, h = {h:.0f} mm (d = {d:.0f} mm)")
    report.append(f"---")
    report.append(f"**Flexure Analysis:**")
    report.append(f"- Max Mu(+) = {mu_pos:.2f} kNm → As_req = {as_bot:.0f} mm² (ρ={rho_bot:.5f})")
    report.append(f"- Max Mu(-) = {mu_neg:.2f} kNm → As_req = {as_top:.0f} mm² (ρ={rho_top:.5f})")
    report.append(f"---")
    report.append(f"**Shear Analysis:**")
    report.append(f"- Vu_max = {vu:.2f} kN, φVc = {phi_v*vc:.2f} kN")
    if vs_req > 0:
        report.append(f"- Required Vs = {vs_req:.2f} kN → RB9 Spacing = {spacing:.0f} mm")
    else:
        report.append(f"- Shear handled by concrete. Provide min stirrups RB9 @ 200 mm")

    return {
        'as_top': as_top, 'as_bot': as_bot, 'spacing': spacing,
        'report': report, 'msg_bot': msg_bot, 'msg_top': msg_top
    }
