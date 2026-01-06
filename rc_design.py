import numpy as np

def calculate_advanced_rc(mu_pos, mu_neg, vu, b_m, h_m, fc, fy, fyt, cover_mm, db_main, db_stirrup):
    phi_m, phi_v = 0.90, 0.75
    b, h = b_m * 1000, h_m * 1000
    # Effective depth based on user-defined cover and bar sizes
    d = h - cover_mm - db_stirrup - (db_main / 2)
    report = []
    
    # Material Constants
    Es = 200000
    n = Es / (4700 * np.sqrt(fc)) # Modular ratio
    
    def design_flexure(mu_knm, label):
        if abs(mu_knm) < 0.1: return 0, 0, 0, "Negligible"
        mu_n = (abs(mu_knm) * 1e6) / phi_m
        k = mu_n / (b * d**2)
        m = fy / (0.85 * fc)
        
        # Check Capacity
        check = 1 - (2 * m * k / fy)
        if check < 0: return 0, k, 0, "SECTION_TOO_SMALL"
        
        rho = (1/m) * (1 - np.sqrt(check))
        rho_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy)
        rho_final = max(rho, rho_min)
        as_req = rho_final * b * d
        n_bars = max(2, int(np.ceil(as_req / (np.pi * (db_main**2) / 4))))
        return as_req, k, n_bars, "OK"

    as_bot, k_bot, n_bot, st_bot = design_flexure(mu_pos, "Bottom")
    as_top, k_top, n_top, st_top = design_flexure(mu_neg, "Top")

    # Optimization Report Logic
    k_limit = 0.18 * fc # Recommended limit for economical design
    if st_bot == "SECTION_TOO_SMALL" or st_top == "SECTION_TOO_SMALL":
        opt_msg = "CRITICAL: Section size is insufficient for the applied moment."
        opt_status = "Error"
    elif max(k_bot, k_top) > k_limit:
        opt_msg = "UNECONOMICAL: High steel ratio. Increasing height (h) would save cost."
        opt_status = "Warning"
    else:
        opt_msg = "OPTIMIZED: Section dimensions and reinforcement are well-balanced."
        opt_status = "Success"

    # Shear Design
    vc = (1/6) * np.sqrt(fc) * b * d / 1000
    vs_req = (vu / phi_v) - vc if vu > (phi_v * vc * 0.5) else 0
    asv = 2 * (np.pi * (db_stirrup**2) / 4) # 2-legs
    spacing = min(d/2, 300)
    if vs_req > 0:
        spacing = min((asv * fyt * d) / (vs_req * 1000), spacing)

    return {
        'as_bot': as_bot, 'n_bot': n_bot, 'as_top': as_top, 'n_top': n_top,
        'spacing': spacing, 'opt_msg': opt_msg, 'opt_status': opt_status,
        'b': b, 'h': h, 'd': d, 'cover': cover_mm, 'db_m': db_main, 'db_s': db_stirrup,
        'k_max': max(k_bot, k_top)
    }
