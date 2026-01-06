import numpy as np

def calculate_advanced_rc(mu_pos, mu_neg, vu, b_m, h_m, fc, fy):
    phi_m, phi_v = 0.90, 0.75
    b, h = b_m * 1000, h_m * 1000  # mm
    d = h - 50  # mm
    Es = 200000 # MPa
    report = []

    def calc_as(mu_knm, label):
        if abs(mu_knm) < 0.1: return 0, 0, 0, f"{label}: Minimal"
        mu_n_mm = (abs(mu_knm) * 1e6) / phi_m
        rn = mu_n_mm / (b * d**2)
        m = fy / (0.85 * fc)
        
        beta1 = max(0.65, 0.85 - 0.05 * (fc - 28) / 7) if fc > 28 else 0.85
        rho_b = (0.85 * fc * beta1 / fy) * (611 / (611 + fy))
        rho_max = 0.75 * rho_b
        
        check_val = 1 - (2 * m * rn / fy)
        if check_val < 0: return 0, 0, 0, f"❌ {label}: Section too small!"
        
        rho = (1/m) * (1 - np.sqrt(check_val))
        rho_final = max(rho, max(0.25 * np.sqrt(fc) / fy, 1.4 / fy))
        
        # Serviceability: Crack Width Estimation (z-factor method)
        fs = 0.6 * fy
        dc = 50 # Cover
        A = (2 * dc * b) / 1 # Effective area per bar (simplified)
        z = fs * (dc * A)**(1/3)
        crack_status = "✅ Crack Control: OK" if z < 30000 else "⚠️ Crack Risk: High"
        
        return rho_final * b * d, rho_final, z, f"✅ {label}: OK | {crack_status}"

    as_bot, rho_bot, z_bot, msg_bot = calc_as(mu_pos, "Bottom")
    as_top, rho_top, z_top, msg_top = calc_as(mu_neg, "Top")

    # Shear & Stirrups
    vc = (1/6) * np.sqrt(fc) * b * d / 1000
    vs_req = (vu / phi_v) - vc if vu > (phi_v * vc * 0.5) else 0
    spacing = min(200, d/2)
    if vs_req > 0:
        spacing = min((127 * fy * d) / (vs_req * 1000), d/2, 600)

    # World Class Addition: Long-term Multiplier (λ)
    # 5 years or more (Time-dependent factor ξ = 2.0)
    rho_prime = 0 # No compression steel in simple model
    lambda_inf = 2.0 / (1 + 50 * rho_prime)

    report.append(f"**Structural Integrity Report**")
    report.append(f"- {msg_bot}")
    report.append(f"- {msg_top}")
    report.append(f"- **Durability:** Estimated z-factor = {max(z_bot, z_top):.0f} (Limit < 30,000 N/mm)")
    report.append(f"- **Serviceability:** Long-term Deflection Multiplier (λ) = {lambda_inf:.2f}")
    
    return {'as_top': as_top, 'as_bot': as_bot, 'spacing': spacing, 'report': report}
