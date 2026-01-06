import numpy as np

def design_span_expert(mu_pos, mu_neg, vu, b_m, h_m, fc, fy, cover_mm, db_main):
    phi_m, phi_v = 0.90, 0.75
    b, h = b_m * 1000, h_m * 1000
    db_stirrup = 9 
    d = h - cover_mm - db_stirrup - (db_main / 2)
    
    def calc_flexure(mu_knm):
        abs_mu = abs(mu_knm)
        if abs_mu < 0.1:
            as_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy) * b * d
            return {"n": 2, "as_req": as_min, "status": "Min Steel"}
        
        mu_n = (abs_mu * 1e6) / phi_m
        rn = mu_n / (b * d**2)
        m = fy / (0.85 * fc)
        
        # Check for Over-reinforced
        if rn > (0.85 * fc * 0.3): 
            return {"n": 0, "status": "FAIL: SECTION TOO SMALL"}
            
        rho = (1/m) * (1 - np.sqrt(max(0, 1 - (2 * m * rn / fy))))
        rho_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy)
        as_req = max(rho, rho_min) * b * d
        n_bars = max(2, int(np.ceil(as_req / (np.pi * (db_main**2) / 4))))
        return {"n": n_bars, "as_req": as_req, "status": "OK"}

    res_pos = calc_flexure(mu_pos)
    res_neg = calc_flexure(mu_neg)
    phi_vc = (phi_v * 0.17 * np.sqrt(fc) * b * d) / 1000
    
    return {
        "pos": res_pos, "neg": res_neg, "vu": vu, "phi_vc": phi_vc,
        "mu_pos": mu_pos, "mu_neg": mu_neg
    }
