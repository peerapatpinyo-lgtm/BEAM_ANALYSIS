import streamlit as st
import pandas as pd
import numpy as np

def calculate_as_req(Mu, b, d, fc, fy):
    """
    ฟังก์ชันคำนวณหน้าตัดเหล็กเสริมรับโมเมนต์ดัด (USD Method)
    Mu: kNm
    b, d: cm
    fc, fy: ksc
    Return: As_req (cm^2)
    """
    if Mu == 0:
        return 0.0
    
    # แปลงหน่วย
    Mu_kgcm = Mu * 1000 * 100 # kNm -> kg.cm
    phi = 0.9
    
    # คำนวณ Rn
    Rn = Mu_kgcm / (phi * b * d**2)
    
    # อัตราส่วนเหล็กเสริม (rho)
    m = fy / (0.85 * fc)
    try:
        rho = (1/m) * (1 - np.sqrt(1 - (2 * m * Rn / fy)))
    except ValueError:
        return 999.99 # Section too small (Error)

    As = rho * b * d
    
    # เหล็กขั้นต่ำ (As min) ตาม ACI/EIT
    as_min1 = (14 / fy) * b * d
    as_min2 = (0.8 * np.sqrt(fc) / fy) * b * d
    as_min = max(as_min1, as_min2)
    
    return max(As, as_min)

def render(n_spans, spans, params, x_ult, M_ult, V_ult, x_svc, M_svc, D_svc, is_service, sup_df):
    """
    ฟังก์ชันหลักสำหรับแสดงผล Tab Design
    รับ sup_df เข้ามาเพื่อแก้ปัญหา TypeError
    """
    st.header("2. Concrete Beam Design (USD)")

    # --- 1. ดึงค่าพารามิเตอร์ ---
    # ใช้ .get() เพื่อป้องกัน Error ถ้า key ไม่มี
    fc = params.get('fc', 240)
    fy = params.get('fy', 4000)
    b = params.get('b', 25)
    h = params.get('h', 50)
    cover = params.get('cover', 4.0) # Covering to centroid
    d = h - cover
    
    # แสดงค่า Design Parameters
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("f'c (ksc)", f"{fc}")
    c2.metric("fy (ksc)", f"{fy}")
    c3.metric("Size b x h (cm)", f"{b} x {h}")
    c4.metric("d (cm)", f"{d}")

    # --- 2. วนลูปคำนวณแต่ละช่วงคาน (Span) ---
    design_data = []
    
    current_x = 0.0
    
    for i in range(n_spans):
        span_length = spans[i]
        end_x = current_x + span_length
        
        # กรองข้อมูลเฉพาะ Span นี้ (โดยใช้ index array เทียบกับ x)
        # หมายเหตุ: x_ult อาจมีจุดทศนิยม ต้องกรองช่วง [current_x, end_x]
        mask = (x_ult >= current_x) & (x_ult <= end_x)
        
        # ตัดข้อมูล Moment และ Shear ในช่วงนี้
        m_span = M_ult[mask]
        v_span = V_ult[mask]
        
        # หาค่า Max Positive Moment (กลางช่วง) และ Max Negative (แถวหัวท้าย)
        # Note: การหาตำแหน่ง Support ที่แม่นยำอาจซับซ้อน ในที่นี้ใช้ Min/Max ของช่วง
        mu_pos_max = np.max(m_span) if len(m_span) > 0 else 0
        mu_neg_max = np.min(m_span) if len(m_span) > 0 else 0 # เป็นลบ
        vu_max = np.max(np.abs(v_span)) if len(v_span) > 0 else 0

        # คำนวณเหล็กเสริม (เหล็กบน และ เหล็กล่าง)
        as_bot = calculate_as_req(max(0, mu_pos_max), b, d, fc, fy)
        as_top = calculate_as_req(abs(mu_neg_max), b, d, fc, fy)
        
        # ตรวจสอบ Deflection (Service Load)
        if is_service and len(D_svc) > 0:
            d_span = D_svc[(x_svc >= current_x) & (x_svc <= end_x)]
            d_max = np.max(np.abs(d_span)) if len(d_span) > 0 else 0
            d_allow = (span_length * 100) / 240 # L/240 convert m to cm
            d_status = "OK" if d_max <= d_allow else "Fail"
        else:
            d_max = 0.0
            d_allow = (span_length * 100) / 240
            d_status = "N/A"

        # เก็บข้อมูลลง List
        design_data.append({
            "Span No.": i + 1,
            "Length (m)": span_length,
            "Mu+ (kNm)": round(mu_pos_max, 2),
            "As Bot (cm2)": round(as_bot, 2),
            "Mu- (kNm)": round(mu_neg_max, 2),
            "As Top (cm2)": round(as_top, 2),
            "Vu Max (kN)": round(vu_max, 2),
            "Deflect (cm)": round(d_max * 100, 3), # แปลง m เป็น cm
            "Allow (cm)": round(d_allow, 3),
            "Check": d_status
        })
        
        current_x += span_length

    # --- 3. สร้าง DataFrame ผลลัพธ์ ---
    df_results = pd.DataFrame(design_data)
    
    st.subheader("Design Results Table")
    st.dataframe(df_results, use_container_width=True)

    # --- 4. แสดงข้อมูล Support (ที่รับมาจาก sup_df) ---
    st.subheader("Support Information")
    st.info("Support data received for design detailing check.")
    st.dataframe(sup_df, use_container_width=True)

    # --- 5. Return ผลลัพธ์กลับไปให้ app.py (เพื่อส่งต่อให้ tab_report) ---
    return df_results
