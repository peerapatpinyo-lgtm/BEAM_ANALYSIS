import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import beam_analysis
import rc_design
import input_handler

# --- Page Config ---
st.set_page_config(page_title="RC Beam Master", page_icon="🏗️", layout="wide")

# --- 1. SIDEBAR & INPUTS ---
# เรียกใช้ UI จาก input_handler
design_code, method, f_dl, f_ll, unit_sys = input_handler.render_sidebar()

st.title("🏗️ RC Beam Analysis & Design")
st.markdown("---")

# 1.1 Geometry Input
n_span, spans, supports = input_handler.render_geometry_input()

# 1.2 Loads Input
loads = input_handler.render_loads_input(n_span, spans, f_dl, f_ll, unit_sys)

# 1.3 Design Parameters Input
fc, fy, b, h, cov, m_bar, s_bar, manual_s = input_handler.render_design_input(unit_sys)

# 1.4 Database พื้นที่หน้าตัดเหล็ก (cm2)
bar_db = {
    'RB6': 0.28, 'RB9': 0.64, 
    'DB10': 0.79, 'DB12': 1.13, 'DB16': 2.01, 
    'DB20': 3.14, 'DB25': 4.91, 'DB28': 6.16
}
m_area = bar_db.get(m_bar, 1.13) # Default DB12
s_area = bar_db.get(s_bar, 0.28) # Default RB6

# --- 2. ACTION BUTTON ---
st.markdown("---")
if st.button("🚀 Run Analysis & Design", type="primary"):
    
    # ==========================
    # A. ANALYSIS ENGINE
    # ==========================
    try:
        # ส่งค่าไปคำนวณที่ beam_analysis.py
        df_res, df_sup = beam_analysis.run_beam_analysis(spans, supports, loads)
    except Exception as e:
        st.error(f"Analysis Error: {e}")
        st.stop()
        
   # ==========================
    # B. VISUALIZATION (SFD & BMD)
    # ==========================
    st.header("📊 Analysis Results")
    
    # Checkbox สำหรับกลับด้านกราฟ Moment (เผื่อวิศวกรถนัดดูแบบ Positive Down)
    invert_moment = st.checkbox("Invert Moment Diagram (กลับด้านโมเมนต์)", value=False)
    
    # ตั้งค่ากราฟ
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # กำหนดตัวแปรสำหรับขยับแกน X (Cumulative Distance)
    current_x_offset = 0.0
    
    # วนลูปพล็อตทีละช่วงคาน (แก้ปัญหากราฟวิ่งย้อน)
    for i in range(n_span):
        # ดึงข้อมูลเฉพาะ Span นั้น
        span_data = df_res[df_res['span_id'] == i].copy()
        
        if span_data.empty:
            continue

        # ตรวจสอบว่า x ใน data เป็น Local (เริ่มที่ 0) หรือ Global
        # ถ้าค่า x ตัวแรกของ span นี้ น้อยกว่า x ตัวสุดท้ายของ span ก่อนหน้า -> แสดงว่าเป็น Local -> ต้องบวก Offset
        local_x = span_data['x']
        
        # ป้องกัน Error ถ้าข้อมูลไม่เรียง: ให้พล็อตตามค่า x ที่แท้จริงบวก offset สะสม
        # (เราสมมติว่าถ้า x มัน reset เป็น 0 แสดงว่าเป็น Local coordinate)
        if i > 0 and local_x.min() < 0.1: 
             plot_x = local_x + current_x_offset
        else:
             # ถ้า x มันต่อเนื่องอยู่แล้ว (Global) ก็ใช้ค่าเดิม
             plot_x = local_x

        # SFD Data
        v = span_data['shear']
        
        # BMD Data (จัดการ Invert ตาม user เลือก)
        m = span_data['moment']
        if invert_moment:
            m = -m

        # 1. Plot SFD (Shear)
        ax1.plot(plot_x, v, color='#1f77b4', linewidth=2)
        ax1.fill_between(plot_x, v, 0, alpha=0.3, color='#1f77b4')
        
        # 2. Plot BMD (Moment)
        ax2.plot(plot_x, m, color='#d62728', linewidth=2)
        ax2.fill_between(plot_x, m, 0, alpha=0.3, color='#d62728')
        
        # อัปเดตระยะสะสม (สำหรับ Span ถัดไป)
        current_x_offset += spans[i]

    # ตกแต่งกราฟ (Shear)
    ax1.set_ylabel(f"Shear ({'kN' if 'kN' in unit_sys else 'kg'})")
    ax1.set_title("Shear Force Diagram (SFD)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    

[Image of shear force diagram]

    
    # ตกแต่งกราฟ (Moment)
    ax2.set_ylabel(f"Moment ({'kN-m' if 'kN' in unit_sys else 'kg-m'})")
    ax2.set_xlabel("Distance (m)")
    ax2.set_title("Bending Moment Diagram (BMD)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    if invert_moment:
        ax2.invert_yaxis() # กลับแกน Y ให้เหมือน convention ไทยบางที่
    

[Image of bending moment diagram]


    # วาดเส้นตำแหน่ง Support (Vertical Lines)
    # ใช้ current_x_offset ไม่ได้ ต้องคำนวณตำแหน่ง Support ใหม่ให้ชัวร์
    sup_x_accum = 0
    for i in range(n_span + 1):
        ax1.axvline(sup_x_accum, color='black', linestyle=':', alpha=0.5)
        ax2.axvline(sup_x_accum, color='black', linestyle=':', alpha=0.5)
        if i < n_span:
            sup_x_accum += spans[i]

    st.pyplot(fig)

    # ==========================
    # C. DESIGN RESULTS (RC)
    # ==========================
    st.header("🧱 Design Results")
    
    # สร้างคอลัมน์ตามจำนวนช่วงคาน (Span)
    cols = st.columns(n_span)
    
    for i in range(n_span):
        with cols[i]:
            st.markdown(f"### 🔹 Span {i+1}")
            
            # ดึงผลลัพธ์เฉพาะ Span นี้
            span_res = df_res[df_res['span_id'] == i]
            
            if span_res.empty:
                st.warning("No data for this span")
                continue
            
            # --- 1. หาค่าแรงวิกฤต (Critical Forces) ---
            m_max_pos = span_res['moment'].max()      # โมเมนต์บวกสูงสุด (Mid-span) -> เหล็กล่าง
            m_max_neg = span_res['moment'].min()      # โมเมนต์ลบต่ำสุด (Supports) -> เหล็กบน
            v_max = span_res['shear'].abs().max()     # แรงเฉือนสูงสุด (Supports) -> เหล็กปลอก
            
            # --- 2. ออกแบบเหล็กล่าง (Bottom Steel) รับ +M ---
            st.markdown("**👇 Bottom Steel (+M):**")
            if m_max_pos > 0.01: # ถ้ามีโมเมนต์บวก
                res_bot = rc_design.calculate_rc_design(
                    m_max_pos, v_max, fc, fy, b, h, cov, 
                    method, unit_sys, m_area, s_area, manual_s
                )
                status_icon = "✅" if "OK" in res_bot.get('msg_flex', '') else "❌"
                st.info(f"{status_icon} **{res_bot['nb']} - {m_bar}**\n\n($M_u$={m_max_pos:.2f})")
            else:
                st.caption("Min. Reinf (No +M)")
                # กรณีโมเมนต์เป็นลบทั้งช่วง (เช่น Cantilever) ก็คำนวณ Min Steel ได้
                # แต่ในที่นี้ละไว้เพื่อให้ UI สะอาด
                
            # --- 3. ออกแบบเหล็กบน (Top Steel) รับ -M ---
            st.markdown("**👆 Top Steel (-M):**")
            if m_max_neg < -0.01: # ถ้ามีโมเมนต์ลบ
                res_top = rc_design.calculate_rc_design(
                    abs(m_max_neg), v_max, fc, fy, b, h, cov, 
                    method, unit_sys, m_area, s_area, manual_s
                )
                status_icon = "✅" if "OK" in res_top.get('msg_flex', '') else "❌"
                st.warning(f"{status_icon} **{res_top['nb']} - {m_bar}**\n\n($M_u$={m_max_neg:.2f})")
            else:
                st.caption("Min. Reinf (No -M)")

            # --- 4. ออกแบบเหล็กปลอก (Stirrups) รับ V ---
            st.markdown("**⛓️ Stirrups (Shear):**")
            # ใช้ V_max คำนวณ (Mu ใส่ค่าไหนก็ได้ เพราะ Shear ไม่ขึ้นกับ M โดยตรงในสูตรนี้)
            res_shear = rc_design.calculate_rc_design(
                max(abs(m_max_pos), abs(m_max_neg)), v_max, 
                fc, fy, b, h, cov, method, unit_sys, m_area, s_area, manual_s
            )
            st.success(f"**{s_bar} {res_shear.get('stirrup_text', '@Err')}**")
            st.caption(f"$V_u$ max = {v_max:.2f}")

            # --- 5. Logs (ซ่อนไว้) ---
            with st.expander("📝 Calc Logs"):
                if m_max_pos > 0.01:
                    st.markdown("**Bottom Design:**")
                    for log in locals().get('res_bot', {}).get('logs', []): st.write(log)
                if m_max_neg < -0.01:
                    st.markdown("---")
                    st.markdown("**Top Design:**")
                    for log in locals().get('res_top', {}).get('logs', []): st.write(log)
                    
            st.markdown("---")

    # แสดง Section Details ภาพรวม
    st.info(f"ℹ️ **Section Used:** {b*10:.0f}x{h*10:.0f} cm | **Cover:** {cov*10:.0f} mm")

