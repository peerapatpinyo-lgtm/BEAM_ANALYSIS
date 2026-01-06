import numpy as np

def calculate_advanced_rc(mu_pos, mu_neg, vu, b_m, h_m, fc, fy):
    phi_m, phi_v = 0.90, 0.75
    b, h = b_m * 1000, h_m * 1000  # mm
    d = h - 50  # mm
    report = []

    def calc_as(mu_knm, label):
        if abs(mu_knm) < 0.1: return 0, 0, f"{label}: No tension reinforcement needed."
        
        mu_n_mm = (abs(mu_knm) * 1e6) / phi_m
        rn = mu_n_mm / (b * d**2)
        m = fy / (0.85 * fc)
        
        # Senior Engineer Logic: Ductility Check (ACI 318)
        beta1 = max(0.65, 0.85 - 0.05 * (fc - 28) / 7) if fc > 28 else 0.85
        rho_b = (0.85 * fc * beta1 / fy) * (611 / (611 + fy))
        rho_max = 0.75 * rho_b # Tension-controlled limit
        
        check_val = 1 - (2 * m * rn / fy)
        if check_val < 0:
            return 0, 0, f"❌ {label}: Section too small! (Compression steel required or resize section)"
        
        rho = (1/m) * (1 - np.sqrt(check_val))
        rho_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy)
        
        if rho > rho_max: status = f"⚠️ {label}: Over-reinforced! Efficiency low."
        else: status = f"✅ {label}: Design meets ductility standards."
            
        rho_final = max(rho, rho_min)
        return rho_final * b * d, rho_final, status

    as_bot, rho_bot, msg_bot = calc_as(mu_pos, "Bottom Reinforcement")
    as_top, rho_top, msg_top = calc_as(mu_neg, "Top Reinforcement")

    # Shear Design
    vc = (1/6) * np.sqrt(fc) * b * d / 1000
    vs_req = (vu / phi_v) - vc if vu > (phi_v * vc * 0.5) else 0
    spacing = 200 # Default
    if vs_req > 0:
        asv = 127 # 2-RB9
        spacing = min((asv * fy * d) / (vs_req * 1000), d/2, 600)

    report.append(f"**Design Summary (fc'={fc}, fy={fy})**")
    report.append(f"- {msg_bot} (As={as_bot:.0f} mm²)")
    report.append(f"- {msg_top} (As={as_top:.0f} mm²)")
    report.append(f"- Concrete Shear (φVc): {phi_v*vc:.2f} kN")
    report.append(f"- Stirrup Spacing (RB9): @ {spacing:.0f} mm")

    return {
        'as_top': as_top, 'as_bot': as_bot, 'spacing': spacing,
        'report': report, 'rho_bot': rho_bot, 'rho_top': rho_top
    }
