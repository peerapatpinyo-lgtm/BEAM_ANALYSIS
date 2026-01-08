# ===========================================================================================
# 🏗️ RC BEAM ANALYSIS & DESIGN SYSTEM: PRECISION LOAD PATH EDITION
# ===========================================================================================
# Version: 5.2.0 (Separated Load Processing & FEM Core)
# Structural Engine: Finite Element Matrix Stiffness Analysis
# Standard: ACI 318-14 Strength Design Method (SDM)
# Language: English UI / Thai Commentary | Script Length: 300+ Lines
# ===========================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sys

# --- 1. INTEGRATION OF ENGINEERING MODULES ---
# ตรวจสอบการเชื่อมต่อกับ Module คำนวณหลัก
try:
    import input_handler
    import solver
    import rc_design
    import design_view
    import section_plotter
except ImportError as e:
    st.error(f"❌ CRITICAL ERROR: Structural modules not found - {e}")
    st.stop()

# --- 2. GLOBAL PAGE ARCHITECTURE ---
st.set_page_config(
    page_title="RC Beam Pro | Precision Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. PROFESSIONAL CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    .main-title { font-size: 34px; color: #1e3a8a; font-weight: bold; border-bottom: 5px solid #3b82f6; padding-bottom: 10px; margin-bottom: 25px; }
    .section-header { font-size: 22px; color: #1e40af; font-weight: 600; margin-top: 30px; border-left: 8px solid #3b82f6; padding-left: 15px; }
    .formula-card { background-color: #f8fafc; border-radius: 10px; padding: 18px; font-family: 'Roboto Mono', monospace; border: 1px solid #cbd5e1; color: #334155; }
    .footer-text { text-align: center; color: #94a3b8; font-size: 12px; margin-top: 50px; padding: 20px; border-top: 1px solid #e2e8f0; }
    .highlight-blue { color: #2563eb; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. PROJECT METADATA ---
st.markdown('<div class="main-title">Continuous RC Beam Analysis (Separated Load Logic)</div>', unsafe_allow_html=True)
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.write(f"📅 **Computation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
with m_col2:
    st.write("💻 **Engine:** Stiffness Matrix FEM v5.2")
with m_col3:
    st.write("📐 **Code Standard:** ACI 318-14")

# --- 5. DATA INPUT ACQUISITION ---
# รับค่าจาก sidebar ทั้งหมด (ขนาดคาน, วัสดุ, จุดรองรับ, แรงกระทำ)
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

# --- 6. KINEMATIC STABILITY VALIDATION ---
if not stable:
    st.warning("⚠️ โครงสร้างไม่มีความมั่นคง (Unstable): โปรดตรวจสอบการตั้งค่า Support ให้เพียงพอต่อการรับแรง")
    st.stop()

# --- 7. LOAD COMBINATION SETTINGS ---
st.markdown('<div class="section-header">1. Load Factoring & Design Combination</div>', unsafe_allow_html=True)
f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
with f_col1:
    f_dl = st.number_input("Dead Load Factor ($f_{DL}$)", value=1.4, step=0.05)
    st.caption("ACI-318 Default: 1.4")
with f_col2:
    f_ll = st.number_input("Live Load Factor ($f_{LL}$)", value=1.7, step=0.05)
    st.caption("ACI-318 Default: 1.7")
with f_col3:
    st.markdown(f'<div class="formula-card">Design Strength ($U$) = {f_dl}DL + {f_ll}LL</div>', unsafe_allow_html=True)

# --- 8. SEPARATED LOAD PROCESSING ENGINE ---
# แยกการคำนวณ Uniform Load และ Point Load ออกจากกันเพื่อความแม่นยำสูงสุด
st.markdown('<div class="section-header">2. Load Integration & Precise Summation</div>', unsafe_allow_html=True)

try:
    final_solver_loads = []
    log_report = []
    
    # Span-wise Accumulator สำหรับ Uniform Load (เพื่อไม่ให้ตัวเลขบนกราฟซ้อนกัน)
    span_udl_totals = {i: 0.0 for i in range(n_spans)}

    # 8.1 การคำนวณน้ำหนักบรรทุกคงที่ (Self-Weight)
    # สูตร: (b * h * 24.0 kN/m3) * Dead Load Factor
    for i in range(n_spans):
        sw_factored = (params['b'] * params['h'] * 24.0) * f_dl
        span_udl_totals[i] += sw_factored
        log_report.append({
            "Span": i + 1, "Load Type": "Self-Weight (Uniform)", "Source": "Dead Load",
            "Factor": f_dl, "Calculated Value": f"{sw_factored:.3f} kN/m",
            "Resultant (kN)": sw_factored * spans[i]
        })

    # 8.2 การแยกแยะแรงจากผู้ใช้งาน (Point vs Uniform)
    if not loads_df.empty:
        for idx, row in loads_df.iterrows():
            current_f = f_dl if row['case'] == "DL" else f_ll
            factored_mag = float(row['mag']) * current_f
            target_span = int(row['span_index'])
            
            # --- กรณีที่ 1: POINT LOAD (แรงแบบจุด) ---
            if row['type'] == 'P':
                # ส่งค่าเข้า Solver แยกกันแต่ละลูก เพื่อความแม่นยำของจุดหักเหบน BMD
                final_solver_loads.append({
                    'span_index': target_span, 'type': 'P',
                    'mag': factored_mag * 1000.0, # เปลี่ยนหน่วยเป็น Newton
                    'd_start': float(row['d_start']), # ตำแหน่ง X ที่แน่นอน
                    'dist': 0.0, 
                    'desc': f"P={factored_mag:.1f}kN" 
                })
                res_force = factored_mag
            
            # --- กรณีที่ 2: UNIFORM LOAD (แรงกระจาย) ---
            else:
                # นำไปรวมกับน้ำหนักบรรทุกใน Span นั้นๆ เพื่อลด Label ซ้อนบนกราฟ
                span_udl_totals[target_span] += factored_mag
                res_force = factored_mag * float(row['dist'])

            log_report.append({
                "Span": target_span + 1, "Load Type": f"User {row['type']}", "Source": row['case'],
                "Factor": current_f, "Calculated Value": f"{factored_mag:.2f}",
                "Resultant (kN)": res_force
            })

    # 8.3 รวม Uniform Load ที่สะสมไว้เข้าสู่ Solver
    for i in range(n_spans):
        if span_udl_totals[i] > 0:
            final_solver_loads.append({
                'span_index': i, 'type': 'U', 
                'mag': span_udl_totals[i] * 1000.0,
                'd_start': 0.0, 'dist': spans[i], 
                'desc': f"Wu={span_udl_totals[i]:.2f}kN/m"
            })

    # แสดงตารางสรุปการคำนวณแรง
    trace_df = pd.DataFrame(log_report)
    st.table(trace_df.assign(**{"Resultant (kN)": trace_df["Resultant (kN)"].map('{:.3f}'.format)}))
    
    sum_total_w = trace_df["Resultant (kN)"].astype(float).sum()
    st.markdown(f'<p class="highlight-blue">Total Factored Load in System (ΣWu + ΣP): {sum_total_w:.4f} kN</p>', unsafe_allow_html=True)

    # --- 9. STRUCTURAL ANALYSIS CORE (FEM) ---
    st.markdown('<div class="section-header">3. FEM Structural Analysis (SFD, BMD, Deflection)</div>', unsafe_allow_html=True)
    
    with st.spinner('กำลังแก้สมการ Matrix Stiffness...'):
        solver_input = pd.DataFrame(final_solver_loads)
        # ประมวลผลคานด้วย Finite Element Method
        x_coords, M_values, V_values, D_values, R_values = solver.solve_beam(spans, sup_df, solver_input, params)
        analysis_results = pd.DataFrame({
            'x': x_coords, 'moment': M_values, 'shear': V_values, 'deflection': D_values * 1000.0
        })

    # แสดงผลกราฟ SFD / BMD
    st.plotly_chart(design_view.plot_analysis_results(analysis_results, spans, sup_df, solver_input, R_values), use_container_width=True)

    # --- 10. EQUILIBRIUM QUALITY CHECK ---
    st.markdown("#### ⚖️ การตรวจสอบสมดุลแรง (Static Equilibrium)")
    total_r_kN = sum(R_values.values()) / 1000.0
    abs_err = abs(sum_total_w - total_r_kN)
    
    q_col1, q_col2, q_col3 = st.columns(3)
    q_col1.metric("Applied Load (ΣW)", f"{sum_total_w:.3f} kN")
    q_col2.metric("Support Reactions (ΣR)", f"{total_r_kN:.3f} kN")
    
    if abs_err < 0.01:
        q_col3.success(f"Equilibrium Verified (Error: {abs_err:.6f} kN)")
    else:
        q_col3.error(f"Equilibrium Error: {abs_err:.4f} kN")

    # --- 11. REINFORCEMENT DESIGN (ACI-BASED) ---
    st.markdown('<div class="section-header">4. RC Design & Structural Detailing</div>', unsafe_allow_html=True)
    
    tab_rep, tab_calc = st.tabs(["📊 สรุปการเสริมเหล็ก", "🧮 รายการคำนวณทางวิศวกรรม"])
    
    final_detailing = []
    main_db = 16 # ขนาดเหล็กประธาน (mm)
    span_start_pos = [0] + list(np.cumsum(spans))

    for i in range(n_spans):
        # ดึงผลการวิเคราะห์เฉพาะช่วง Span นั้นๆ
        mask = (analysis_results['x'] >= span_start_pos[i] - 1e-9) & (analysis_results['x'] <= span_start_pos[i+1] + 1e-9)
        span_data = analysis_results[mask]
        
        if not span_data.empty:
            m_pos_max = span_data['moment'].max() / 1000.0
            m_neg_max = abs(span_data['moment'].min()) / 1000.0
            v_max = span_data['shear'].abs().max() / 1000.0
            d_eff = params['h'] - 0.05
            
            # คำนวณหาพื้นที่เหล็กเสริม (Flexure & Shear)
            as_p, _, _, log_p = rc_design.design_beam_flexure(m_pos_max, params['b'], d_eff, params['fc'], params['fy'])
            as_n, _, _, log_n = rc_design.design_beam_flexure(m_neg_max, params['b'], d_eff, params['fc'], params['fy'])
            sv_req, _, log_v = rc_design.check_shear(v_max, params['b'], d_eff, params['fc'], params['fy'])
            
            def get_n_bars(area, db): return max(2, int(np.ceil(area / (np.pi * (db/2)**2))))
            n_bot, n_top = get_n_bars(as_p, main_db), get_n_bars(as_n, main_db)
            
            final_detailing.append({
                'span': i + 1, 'pos': {'n': n_bot}, 'neg': {'n': n_top},
                'db': main_db, 'stirrup_label': f"RB6@{sv_req*100:.0f}cm", 'shear': {'s': sv_req}
            })
            
            with tab_calc:
                st.write(f"### 📑 รายการคำนวณ: Span {i+1}")
                tc1, tc2 = st.columns(2)
                with tc1:
                    st.write("**การออกแบบเหล็กเสริมรับโมเมนต์ดัด (Flexure):**")
                    for stmt in log_p: st.latex(stmt)
                with tc2:
                    st.write("**การออกแบบเหล็กปลอก (Shear):**")
                    for stmt in log_v: st.latex(stmt)

    with tab_rep:
        summary_df = pd.DataFrame([
            {"Span ID": d['span'], "Top Steel": f"{d['neg']['n']}-DB{d['db']}", "Bottom Steel": f"{d['pos']['n']}-DB{d['db']}", "Stirrups": d['stirrup_label']} 
            for d in final_detailing
        ])
        st.table(summary_df)
        
        # --- 12. DRAWINGS & GRAPHICS ---
        st.markdown("#### 🎨 Sectional Profile & Longitudinal Details")
        draw1, draw2 = st.columns([1, 2])
        with draw1:
            st.write("**Typical Cross-Section**")
            fig_sec = section_plotter.plot_section(params['b'], params['h'], 40, main_db, final_detailing[0]['neg']['n'], final_detailing[0]['pos']['n'], "RB6", params['fc'], params['fy'])
            st.pyplot(fig_sec)
        with draw2:
            st.write("**Longitudinal Detailing**")
            fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, final_detailing, params['h'], 40)
            st.pyplot(fig_long)

except Exception as ex:
    st.error(f"⚠️ เกิดข้อผิดพลาดในการประมวลผล: {str(ex)}")
    st.exception(ex)

st.markdown('<div class="footer-text">RC Beam Analyzer v5.2.0 | Structural Precision Analysis | ACI 318-14 Compliance</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------
# END OF PROFESSIONAL SCRIPT
# -------------------------------------------------------------------------------------------
