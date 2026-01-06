import numpy as np

def design_section_detailed(mu_pos, mu_neg, vu, b_m, h_m, fc, fy, fyt, cover_mm, db_main, db_stirrup):
    phi_m, phi_v = 0.90, 0.75
    b, h = b_m * 1000, h_m * 1000
    d = h - cover_mm - db_stirrup - (db_main / 2)
    
    def calc_step(mu_knm):
        if abs(mu_knm) < 0.1: return {"n": 2, "as_req": 0, "k": 0, "rho": 0}
        mu_n = (abs(mu_knm) * 1e6) / phi_m
        k = mu_n / (b * d**2)
        m = fy / (0.85 * fc)
        check = 1 - (2 * m * k / fy)
        
        if check < 0: rho = 0.03 # Failure Case
        else: rho = (1/m) * (1 - np.sqrt(max(0, check)))
        
        rho_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy)
        rho_final = max(rho, rho_min)
        as_req = rho_final * b * d
        n_bars = max(2, int(np.ceil(as_req / (np.pi * (db_main**2) / 4))))
        
        return {"n": n_bars, "as_req": as_req, "k": k, "rho": rho_final}

    pos_res = calc_step(mu_pos)
    neg_res = calc_step(mu_neg)
    
    # Shear
    vc = (1/6) * np.sqrt(fc) * b * d / 1000
    vs_req = (vu / phi_v) - vc if vu > (phi_v * vc * 0.5) else 0
    asv = 2 * (np.pi * (db_stirrup**2) / 4)
    s = min(d/2, 300)
    if vs_req > 0:
        s = min((asv * fyt * d) / (vs_req * 1000), s)
        
    return {
        "pos": pos_res, "neg": neg_res, "spacing": int(s), 
        "d": d, "b": b, "h": h, "mu_pos": mu_pos, "mu_neg": mu_neg, "vu": vu
    }
