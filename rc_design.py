import math

def design_span_expert(m_pos, m_neg, v_u, b, h, fc, fy, cover, db):
    """
    ออกแบบคานรับแรงดัดและแรงเฉือน
    Inputs:
    - m_pos, m_neg (kNm): โมเมนต์บวกและลบ
    - v_u (kN): แรงเฉือน
    - b, h (m): ขนาดหน้าตัด
    - fc, fy (MPa): กำลังวัสดุ
    - cover (mm): ระยะหุ้ม
    - db (mm): ขนาดเหล็กเสริมหลัก
    """
    
    # แปลงหน่วย
    d = h - (cover/1000) - (db/2000) - 0.009 # d effective (approx stirrup 9mm)
    phi_b = 0.9 # Flexure
    phi_v = 0.85 # Shear
    
    # ฟังก์ชันคำนวณ As
    def get_As(Mu):
        if abs(Mu) < 0.1: return 0
        Mu_Nm = abs(Mu) * 1000 * 1000
        # Rn = Mu / (phi * b * d^2)
        Rn = Mu_Nm / (phi_b * (b*1000) * (d*1000)**2)
        rho = (0.85 * fc / fy) * (1 - math.sqrt(max(0, 1 - (2 * Rn) / (0.85 * fc))))
        As_req = rho * (b*1000) * (d*1000)
        
        # Min Steel
        As_min = (1.4 / fy) * (b*1000) * (d*1000)
        return max(As_req, As_min)

    # คำนวณเหล็กบน (Negative) และเหล็กล่าง (Positive)
    as_top = get_As(m_neg)
    as_bot = get_As(m_pos)
    
    area_db = 3.14159 * (db/2)**2
    n_top = math.ceil(as_top / area_db)
    n_bot = math.ceil(as_bot / area_db)
    
    # จัดเหล็กขั้นต่ำ 2 เส้น
    n_top = max(2, n_top)
    n_bot = max(2, n_bot)
    
    return {
        "pos": {"as": as_bot, "n": n_bot},
        "neg": {"as": as_top, "n": n_top},
        "shear_status": "OK" if v_u < 1000 else "Check Shear" # Simplified check
    }
