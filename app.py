import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import beam_analysis
import rc_design
import input_handler

# --- Page Config ---
st.set_page_config(page_title="RC Beam Master", page_icon="🏗️", layout="wide")

# ==========================
# 1. SIDEBAR & INPUTS
# ==========================
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
m_area = bar_db.get(m_bar, 1.13)
s_area = bar_db.get(s_bar, 0.28)

# ==========================
# 2. ACTION BUTTON
# ==========================
st.markdown("---")
if st.button("🚀 Run Analysis & Design", type="primary"):
    
    # ----------------------------------------
    # A. ANALYSIS ENGINE (ห้ามแตะต้อง Logic นี้)
    # ----------------------------------------
    try:
        # ส่งค่าไปคำนวณที่ beam_analysis.py
        df_res, df_sup = beam_analysis.run_beam_analysis(spans, supports, loads)
    except Exception as e:
        st.error(f"Analysis Error: {e}")
        st.stop()
        
    # ----------------------------------------
    # B. VISUALIZATION (แก้ไขกราฟให้ไม่เพี้ยน)
    # ----------------------------------------
    st.header("📊 Analysis Results")
    
    # Checkbox กลับด้านโมเมนต์
    invert_moment = st.checkbox("Invert Moment Diagram (กลับด้านโมเมนต์)", value=False)
    
    # เตรียมพื้นที่กราฟ
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # ตัวแปรช่วยขยับแกน X ให้ต่อเนื่อง (Offset)
    current_x_offset = 0.0
    
    # วนลูปพล็อตทีละช่วงคาน (แก้ปัญหากราฟลากเส้นมั่ว/Zig-Zag)
    for i in range(n_span):
        # ดึงข้อมูลเฉพาะ Span นี้
        span_data = df_res[df_res['span_id'] == i].copy()
        
        if span_data.empty:
            continue

        local_x = span_data['x']
        
        # Logic: ถ้า x เริ่มที่ 0 ใหม่ทุก Span (Local) ให้บวก Offset
        # แต่ถ้า x เป็น Global (ต่อเนื่องมาแล้ว) ก็ใช้ค่าเดิม
        if i > 0 and local_x.min() < 0.1: 
             plot_x = local_x + current_x_offset
        else:
             plot_x = local_x

        # ข้อมูล Shear และ Moment
        v = span_data['shear']
        m = span_data['moment']
        if invert_moment:
            m = -m

        # Plot SFD (Shear) - สีน้ำเงิน
        ax1.plot(plot_x, v, color='#1f77b4', linewidth=2)
        ax1.fill_between(plot_x, v, 0, alpha=0.3, color='#1f77b4')
        
        # Plot BMD (Moment) - สีแดง
        ax2.plot(plot_x, m, color='#d62728', linewidth=2)
        ax2.fill_between(plot_x, m, 0, alpha=0.3, color='#d62728')
        
        # อัปเดตระยะสะสม
        current_x_offset += spans[i]

    # ตกแต่งกราฟ Shear
    ax1.set_ylabel(f"Shear ({'kN' if 'kN' in unit_sys else 'kg'})")
    ax1.set_title("Shear Force Diagram (SFD)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # ตกแต่งกราฟ Moment
    ax2.set_ylabel(f"Moment ({'kN-m' if 'kN' in unit_sys else 'kg-m'})")
    ax2.set_xlabel("Distance (m)")
    ax2.set_title("Bending Moment Diagram (BMD)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    if invert_moment:
        ax2.invert_yaxis()

    # วาดเส้นตำแหน่ง Support
    sup_x_accum = 0
    for i in range(n_span + 1):
        ax1.axvline(sup_x_accum, color='black', linestyle=':', alpha=0.5)
        ax2.axvline(sup_x_accum, color='black', linestyle=':', alpha=0.5)
        if i < n_span:
            sup_x_accum += spans[i]

    st.pyplot(fig)

    # ----------------------------------------
    # C. DESIGN RESULTS (RC) - แยกบน/ล่าง
    # ----------------------------------------
    st.header("🧱 Design Results")
    
    cols = st.columns(n_span)
    
    for i in range(n_span):
        with cols[i]:
            st.markdown(f"### 🔹 Span {i+1}")
            
            # ดึงผลลัพธ์เฉพาะ Span นี้
            span_res = df_res[df_res['span_id'] == i]
            
            if span_res.empty:
                st.warning("No data")
                continue
            
            # หาค่าแรงวิกฤต
            m_max_pos = span_res['moment'].max()  # +M (เหล็กล่าง)
            m_max_neg = span_res['moment'].min()  # -M (เหล็กบน)
            v_max = span_res['shear'].abs().max() # V (เหล็กปลอก)
            
            # --- 1. เหล็กล่าง (Bottom Steel) ---
            st.markdown("**👇 Bottom Steel (+M):**")
            if m_max_pos > 0.01:
                res_bot = rc_design.calculate_rc_design(
                    m_max_pos, v_max, fc, fy, b, h, cov, 
                    method, unit_sys, m_area, s_area, manual_s
                )
                icon = "✅" if "OK" in res_bot.get('msg_flex', '') else "❌"
                st.info(f"{icon} **{res_bot['nb']} - {m_bar}**\n\n(Mu={m_max_pos:.2f})")
            else:
                st.caption("Min. Reinf (No +M)")
                
            # --- 2. เหล็กบน (Top Steel) ---
            st.markdown("**👆 Top Steel (-M):**")
            if m_max_neg < -0.01:
                res_top = rc_design.calculate_rc_design(
                    abs(m_max_neg), v_max, fc, fy, b, h, cov, 
                    method, unit_sys, m_area, s_area, manual_s
                )
                icon = "✅" if "OK" in res_top.get('msg_flex', '') else "❌"
                st.warning(f"{icon} **{res_top['nb']} - {m_bar}**\n\n(Mu={m_max_neg:.2f})")
            else:
                st.caption("Min. Reinf (No -M)")

            # --- 3. เหล็กปลอก (Stirrups) ---
            st.markdown("**⛓️ Stirrups:**")
            # ใช้ Vmax ออกแบบ
            res_shear = rc_design.calculate_rc_design(
                max(abs(m_max_pos), abs(m_max_neg)), v_max, 
                fc, fy, b, h, cov, method, unit_sys, m_area, s_area, manual_s
            )
            st.success(f"**{s_bar} {res_shear.get('stirrup_text', 'Err')}**")
            st.caption(f"Vu max = {v_max:.2f}")

            # Logs
            with st.expander("📝 Calc Logs"):
                if m_max_pos > 0.01:
                    st.markdown("**Bottom:**")
                    for l in locals().get('res_bot', {}).get('logs', []): st.write(l)
                if m_max_neg < -0.01:
                    st.markdown("**Top:**")
                    for l in locals().get('res_top', {}).get('logs', []): st.write(l)
                    
            st.markdown("---")

    st.info(f"ℹ️ **Section Used:** {b*10:.0f}x{h*10:.0f} cm | **Cover:** {cov*10:.0f} mm")
