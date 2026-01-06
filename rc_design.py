import numpy as np

def design_section(mu_pos, mu_neg, vu, b_m, h_m, fc, fy, fyt, cover_mm, db_main, db_stirrup):
    phi_m, phi_v = 0.90, 0.75
    b, h = b_m * 1000, h_m * 1000
    d = h - cover_mm - db_stirrup - (db_main / 2)
    
    def calc_bars(mu_knm):
        if abs(mu_knm) < 0.5: return 2 # Minimum 2 bars
        mu_n = (abs(mu_knm) * 1e6) / phi_m
        m = fy / (0.85 * fc)
        k = mu_n / (b * d**2)
        check = 1 - (2 * m * k / fy)
        if check < 0: return 0 # Failure
        rho = (1/m) * (1 - np.sqrt(check))
        rho_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy)
        as_req = max(rho, rho_min) * b * d
        return max(2, int(np.ceil(as_req / (np.pi * (db_main**2) / 4))))

    n_top = calc_bars(mu_neg)
    n_bot = calc_bars(mu_pos)
    
    # Stirrup spacing
    vc = (1/6) * np.sqrt(fc) * b * d / 1000
    vs_req = (vu / phi_v) - vc if vu > (phi_v * vc * 0.5) else 0
    asv = 2 * (np.pi * (db_stirrup**2) / 4)
    s = min(d/2, 300)
    if vs_req > 0:
        s = min((asv * fyt * d) / (vs_req * 1000), s)
        
    return {"n_top": n_top, "n_bot": n_bot, "spacing": int(s)}
