# tab_design.py
import streamlit as st
import numpy as np
import pandas as pd
import calcs
import section_plotter # สมมติว่ามีไฟล์วาดรูปตัด

def render(n_spans, spans, params, x_ult, M_ult, V_ult, x_svc, M_svc, D_svc, is_service):
    st.header(f"🏗️ Interactive RC Design")
    if is_service: 
        st.warning("⚠️ Warning: Viewing Service Loads, but Design logic uses Ultimate Loads.")

    # ดึงค่าคงที่และแปลงหน่วย
    b_mm, h_mm = calcs.normalize_section_units(params['b'], params['h'])
    fc, fy = params['fc'], params['fy']
    
    final_design_res = [] # ตัวแปรที่จะเก็บผลลัพธ์ส่งกลับ
    offsets = [0] + list(np.cumsum(spans))
    
    for i in range(n_spans):
        s_len = spans[i]
        s_start, s_end = offsets[i], offsets[i+1]
        
        # กรองหาค่า Max ในแต่ละช่วงคาน (Span)
        mask_ult = (x_ult >= s_start - 1e-6) & (x_ult <= s_end + 1e-6)
        if any(mask_ult):
            mu_pos = max(0.0, (M_ult[mask_ult] / 1000.0).max())
            mu_neg = abs(min(0.0, (M_ult[mask_ult] / 1000.0).min()))
            vu_max = abs((V_ult[mask_ult] / 1000.0)).max()
        else: mu_pos, mu_neg, vu_max = 0, 0, 0
        
        # Service Load Data (สำหรับดู Deflection ประกอบ)
        mask_svc = (x_svc >= s_start - 1e-6) & (x_svc <= s_end + 1e-6)
        if any(mask_svc):
            ma_pos_svc = max(0.0, (M_svc[mask_svc] / 1000.0).max())
            delta_svc_mm = abs((D_svc[mask_svc] * 1000.0)).max()
        else: ma_pos_svc, delta_svc_mm = 0, 0

        # --- UI ส่วนเลือกเหล็ก ---
        with st.expander(f"📍 **Span {i+1}** (L={s_len} m) | Forces: Mu+={mu_pos:.1f}, Mu-={mu_neg:.1f}", expanded=True):
            c_const, c_cov = st.columns([3, 1])
            with c_const: st.caption(f"Size {b_mm:.0f}x{h_mm:.0f} mm | fc'={fc} | fy={fy}")
            with c_cov: cover_mm = st.number_input(f"Cover (mm)", 20.0, 50.0, 25.0, key=f"cov_{i}")

            d_est = h_mm - cover_mm - 20 

            # 1. เหล็กล่าง (Bottom)
            st.markdown("##### 1. Bottom Reinforcement")
            as_req_bot, _, _ = calcs.get_as_req(mu_pos, d_est, fc, fy, b_mm)
            
            c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
            with c1: st.info(f"Req As: {as_req_bot:.0f}")
            with c2: bot_db = st.selectbox("DB", [12, 16, 20, 25], index=1, key=f"bdb_{i}")
            with c3: bot_n = st.number_input("Qty", 2, 10, 2, key=f"bn_{i}")
            
            d_real = h_mm - cover_mm - 9 - (bot_db / 2)
            phi_Mn_bot, as_prov_bot, _, _, _, _ = calcs.get_phi_Mn_details(bot_n, bot_db, d_real, b_mm, fc, fy)
            pass_b = (phi_Mn_bot >= mu_pos)
            with c4: 
                st.metric("Capacity", f"{phi_Mn_bot:.2f} kNm", delta="OK" if pass_b else "FAIL")

            # 2. เหล็กบน (Top)
            st.markdown("##### 2. Top Reinforcement")
            as_req_top, _, _ = calcs.get_as_req(mu_neg, d_est, fc, fy, b_mm)
            
            c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
            with c1: st.info(f"Req As: {as_req_top:.0f}")
            with c2: top_db = st.selectbox("DB", [12, 16, 20, 25], index=1, key=f"tdb_{i}")
            with c3: top_n = st.number_input("Qty", 2, 10, 2, key=f"tn_{i}")
            
            d_top_real = h_mm - cover_mm - 9 - (top_db / 2)
            phi_Mn_top, as_prov_top, _, _, _, _ = calcs.get_phi_Mn_details(top_n, top_db, d_top_real, b_mm, fc, fy)
            pass_t = (phi_Mn_top >= mu_neg)
            with c4:
                st.metric("Capacity", f"{phi_Mn_top:.2f} kNm", delta="OK" if pass_t else "FAIL")

            # 3. เหล็กปลอก (Shear)
            st.markdown("##### 3. Shear Stirrup")
            c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
            with c1: st.markdown(f"Vu: **{vu_max:.1f}** kN")
            with c2: stir_db = st.selectbox("RB", [6, 9], index=0, key=f"sdb_{i}")
            with c3: stir_s = st.number_input("@Spacing", 50, 300, 150, 10, key=f"ss_{i}")
            
            status_v, phi_Vn, _, _, _, _ = calcs.check_shear_details(vu_max, b_mm, d_real, fc, fy, stir_db, stir_s)
            with c4: st.write(f"Status: **{status_v}**")

            # เก็บข้อมูลเข้า List
           
# tab_design.py

# ... (โค้ดส่วนบนเหมือนเดิม) ...

        # เก็บข้อมูลเข้า List
        final_design_res.append({
            'span_id': i, 'L': s_len, 'b': b_mm, 'h': h_mm, 'fc': fc, 'fy': fy,
            'Mu_pos': mu_pos, 'Mu_neg': mu_neg, 'Vu_max': vu_max,
            'cover': cover_mm, 
            
            # --- ส่วนที่ต้องเพิ่ม (Update) ---
            'Ma_pos_svc': ma_pos_svc,      # <--- เพิ่มบรรทัดนี้
            'delta_svc_mm': delta_svc_mm,  # <--- เพิ่มบรรทัดนี้
            # ---------------------------
            
            'pos': {'n': bot_n, 'db': bot_db, 'status': pass_b},
            'neg': {'n': top_n, 'db': top_db, 'status': pass_t},
            'shear': {'s': stir_s, 'db': stir_db, 'status': status_v},
            
            # เผื่อไว้: หาก reporter.py ของคุณต้องการ key ชื่อ 'bot' หรือ 'top' แยกต่างหาก (ตามเวอร์ชั่นแรก)
            'bot': {'n': bot_n, 'db': bot_db}, 
            'top': {'n': top_n, 'db': top_db},
        })

    # ... (โค้ดส่วนล่างเหมือนเดิม) ...
    # ปุ่ม Generate Drawing (Optional)
    if st.button("Generat Section Drawing", type="primary"):
         try:
            fig = section_plotter.plot_longitudinal_section_detailed(spans, final_design_res, h_mm)
            st.pyplot(fig)
         except: st.error("Drawing module not ready")

    return final_design_res
