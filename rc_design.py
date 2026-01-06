import numpy as np

def design_section(mu_knm, b_m, h_m, fc, fy, cover_mm, db_main):
    # พารามิเตอร์พื้นฐาน
    phi = 0.9
    b = b_m * 1000  # mm
    h = h_m * 1000  # mm
    d = h - cover_mm - 9 - (db_main/2) # d = h - cover - stirrup - db/2
    
    if abs(mu_knm) < 1.0: # กรณีโมเมนต์น้อยมาก ให้ใส่เหล็กขั้นต่ำ
        as_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy) * b * d
        n_bars = max(2, int(np.ceil(as_min / (np.pi * (db_main**2) / 4))))
        return n_bars, "Min Steel"

    # คำนวณเนื้อที่เหล็กเสริม
    rn = (abs(mu_knm) * 1e6) / (phi * b * d**2)
    m = fy / (0.85 * fc)
    rho = (1/m) * (1 - np.sqrt(max(0, 1 - (2 * m * rn / fy))))
    
    # ตรวจสอบ rho min/max
    rho_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy)
    rho_final = max(rho, rho_min)
    
    as_req = rho_final * b * d
    n_bars = max(2, int(np.ceil(as_req / (np.pi * (db_main**2) / 4))))
    
    status = "OK" if rho <= 0.75 * (0.85 * fc * 0.85 / fy * (600/(600+fy))) else "Section too small"
    return n_bars, status
