import numpy as np

def design_span_expert(mu_pos, mu_neg, vu, b_m, h_m, fc, fy, cover_mm, db_main):
    """
    คำนวณเหล็กเสริมทั้งบน (Negative Moment) และล่าง (Positive Moment) พร้อมกัน
    """
    phi_m = 0.90
    phi_v = 0.75
    b = b_m * 1000  # แปลง m -> mm
    h = h_m * 1000  # แปลง m -> mm
    db_stirrup = 9 
    d = h - cover_mm - db_stirrup - (db_main / 2)
    
    def calc_flexure(mu_knm):
        # ใช้ absolute value เพราะเราสนใจแค่ปริมาณเหล็ก ไม่สนเครื่องหมาย
        abs_mu = abs(mu_knm)
        
        # กรณีโมเมนต์น้อยมาก (เช่น น้อยกว่า 0.1 kNm) ให้ใส่เหล็กขั้นต่ำ
        if abs_mu < 0.1:
            as_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy) * b * d
            n_bars = max(2, int(np.ceil(as_min / (np.pi * (db_main**2) / 4))))
            return {"n": n_bars, "as_req": as_min, "status": "Min Steel"}
        
        # คำนวณ Rn, rho
        mu_n = (abs_mu * 1e6) / phi_m
        rn = mu_n / (b * d**2)
        m = fy / (0.85 * fc)
        
        # ตรวจสอบหน้าตัดระเบิด (Compression Failure)
        rho_max = 0.85 * (fc / fy) * (600 / (600 + fy)) # ประมาณการคร่าวๆ
        if rn > (0.85 * fc * 0.35): # Check Limit
             return {"n": 0, "status": "FAIL: SECTION TOO SMALL"}
            
        # คำนวณ Rho ที่ต้องการ
        try:
            rho = (1/m) * (1 - np.sqrt(max(0, 1 - (2 * m * rn / fy))))
        except:
            return {"n": 0, "status": "CALC ERROR"}

        rho_min = max(0.25 * np.sqrt(fc) / fy, 1.4 / fy)
        as_req = max(rho, rho_min) * b * d
        
        # แปลงเป็นจำนวนเส้น
        area_bar = np.pi * (db_main**2) / 4
        n_bars = max(2, int(np.ceil(as_req / area_bar)))
        
        return {"n": n_bars, "as_req": as_req, "status": "OK"}

    # คำนวณแยก บน-ล่าง
    res_pos = calc_flexure(mu_pos)
    res_neg = calc_flexure(mu_neg)
    
    # คำนวณแรงเฉือนรับได้
    phi_vc = (phi_v * 0.17 * np.sqrt(fc) * b * d) / 1000 # kN
    
    return {
        "pos": res_pos, 
        "neg": res_neg, 
        "vu": vu, 
        "phi_vc": phi_vc,
        "mu_pos": mu_pos, 
        "mu_neg": mu_neg
    }
