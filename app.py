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
    # A. ANALYSIS ENGINE
    # ----------------------------------------
    try:
        df_res, df_sup = beam_analysis.run_beam_analysis(spans, supports, loads)
    except Exception as e:
        st.error(f"Analysis Error: {e}")
        st.stop()
        
    # ----------------------------------------
    # B. VISUALIZATION (SFD & BMD) - กู้คืนส่วนนี้
    # ----------------------------------------
    st.header("📊 Analysis Results")
    
    invert_moment = st.checkbox("Invert Moment Diagram", value=False)
    
    # เตรียมข้อมูลสำหรับ Plot (จัดการเรื่องแกน X ให้ต่อเนื่อง)
    # เราจะสร้าง Global X ขึ้นมาใหม่ เพื่อให้กราฟวาดต่อกันสวยงามไม่ ZigZag
    plot_data = []
    current_offset = 0
    for i in range(n_span):
        # ดึงข้อมูลของ Span นั้น
        span_df = df_res[df_res['span_id'] == i].copy()
        
        # ถ้า x เป็น local (เริ่ม 0 ใหม่) ให้บวก offset
        # ถ้า x เป็น global อยู่แล้ว ก็ใช้ค่าเดิม
        if i > 0 and span_df['x'].min() < 0.1:
            span_df['plot_x'] = span_df['x'] + current_offset
        else:
            span_df['plot_x'] = span_df['x']
            
        plot_data.append(span_df)
        current_offset += spans[i]
    
    # รวมข้อมูลกลับเป็นตารางเดียวเพื่อพล็อตทีเดียว (ชัวร์กว่า Loop พล็อต)
    if plot_data:
        df_plot = pd.concat(plot_data)
    else:
        df_plot = df_res.copy()
        df_plot['plot_x'] = df_plot['x']

    # เริ่มวาดกราฟ
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    x = df_plot['plot_x']
    v = df_plot['shear']
    m = df_plot['moment']
    if invert_moment: m = -m

    # 1. Shear Force Diagram
    ax1.plot(x, v, color='#1f77b4', linewidth=2)
    ax1.fill_between(x, v, 0, alpha=0.3, color='#1f77b4')
    ax1.set_ylabel(f"Shear ({'kN' if 'kN' in unit_sys else 'kg'})")
    ax1.set_title("Shear Force Diagram (SFD)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Bending Moment Diagram
    ax2.plot(x, m, color='#d62728', linewidth=2)
    ax2.fill_between(x, m, 0, alpha=0.3, color='#d62728')
    ax2.set_ylabel(f"Moment ({'kN-m' if 'kN' in unit_sys else 'kg-m'})")
    ax2.set_xlabel("Distance (m)")
    ax2.set_title("Bending Moment Diagram (BMD)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    if invert_moment: ax2.invert_yaxis()
    
    # วาดเส้น Support
    sup_accum = 0
    for i in range(n_span + 1):
        ax1.axvline(sup_accum, color='black', linestyle=':', alpha=0.5)
        ax2.axvline(sup_accum, color='black', linestyle=':', alpha=0.5)
        if i < n_span: sup_accum += spans[i]

    st.pyplot(fig)

    # ----------------------------------------
    # C. DESIGN RESULTS (RC) - ส่วนที่ขอใหม่ (แยกบน/ล่าง)
    # ----------------------------------------
    st.header("🧱 Design Results")
    
    cols = st.columns(n_span)
    
    for i in range(n_span):
        with cols[i]:
            st.markdown(f"### 🔹 Span {i+1}")
            
            span_res = df_res[df_res['span_id'] == i]
            if span_res.empty: continue
            
            # Critical Forces
            m_max_pos = span_res['moment'].max()
            m_max_neg = span_res['moment'].min()
            v_max = span_res['shear'].abs().max()
            
            # 1. Bottom Steel (+M)
            st.markdown("**👇 Bottom Steel (+M):**")
            if m_max_pos > 0.01:
                res = rc_design.calculate_rc_design(
                    m_max_pos, v_max, fc, fy, b, h, cov, 
                    method, unit_sys, m_area, s_area, manual_s
                )
                icon = "✅" if "OK" in res.get('msg_flex', '') else "❌"
                st.info(f"{icon} **{res['nb']} - {m_bar}**")
            else:
                st.caption("-")
                
            # 2. Top Steel (-M)
            st.markdown("**👆 Top Steel (-M):**")
            if m_max_neg < -0.01:
                res = rc_design.calculate_rc_design(
                    abs(m_max_neg), v_max, fc, fy, b, h, cov, 
                    method, unit_sys, m_area, s_area, manual_s
                )
                icon = "✅" if "OK" in res.get('msg_flex', '') else "❌"
                st.warning(f"{icon} **{res['nb']} - {m_bar}**")
            else:
                st.caption("-")

            # 3. Stirrups
            st.markdown("**⛓️ Stirrups:**")
            res_s = rc_design.calculate_rc_design(
                max(abs(m_max_pos), abs(m_max_neg)), v_max, 
                fc, fy, b, h, cov, method, unit_sys, m_area, s_area, manual_s
            )
            st.success(f"**{s_bar} {res_s.get('stirrup_text','')}**")
            
            st.markdown("---")
