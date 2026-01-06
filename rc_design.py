import numpy as np

def calculate_rc_details(mu_max_abs, vu_max, b, h, fc, fy=400):
    phi_m, phi_v = 0.90, 0.75
    d = h - 0.05 
    
    as_req, rho = 0, 0
    if mu_max_abs > 0:
        rn = (mu_max_abs * 1e6) / (phi_m * (b * 1000) * (d * 1000)**2)
        m = fy / (0.85 * fc)
        rho = (1/m) * (1 - np.sqrt(max(0, 1 - (2 * m * rn / fy))))
        rho_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy)
        rho = max(rho, rho_min)
        as_req = rho * (b * 1000) * (d * 1000)

    vc = (1/6) * np.sqrt(fc) * (b * 1000) * (d * 1000) / 1000
    vs_req = max(0, (vu_max / phi_v) - vc) if vu_max > (0.5 * phi_v * vc) else 0
        
    return {'as_mm2': as_req, 'rho': rho, 'vc_kn': vc, 'vs_kn': vs_req, 'd_mm': d * 1000}
