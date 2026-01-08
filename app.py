# =====================================================================
# 🏗️ RC BEAM ANALYSIS & DESIGN PROFESSIONAL SUITE
# Version: 3.0 (Full Detailed Edition)
# Description: Advanced Finite Element Analysis for Reinforced Concrete
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CORE ENGINE MODULES ---
# มอดูลที่เชื่อมต่อกับไฟล์ภายนอก
import input_handler
import solver
import rc_design
import design_view
import section_plotter

# --- 2. GLOBAL PAGE SETTINGS ---
# ตั้งค่าหน้าเว็บให้รองรับการแสดงผลรายงานขนาดใหญ่
st.set_page_config(
    page_title="Professional Beam Designer Pro",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CUSTOM CSS INTERFACE ---
# ปรับปรุง UI ให้ดูเหมือนโปรแกรมวิศวกรรมระดับสูง
st.markdown("""
    <style>
    .main-header { font-size: 32px; color: #1e3a8a; font-weight: bold; border-bottom: 3px solid #3b82f6; padding-bottom: 10px; }
    .sub-header { font-size: 22px; color: #1e40af; margin-top: 20px; font-weight: 600; }
    .calc-note { background-color: #f8fafc; border-left: 6px solid #10b981; padding: 15px; border-radius: 4px; margin: 15px 0; }
    .metric-box { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .error-msg { background-color: #fef2f2; border: 1px solid #ef4444; color: #b91c1c; padding: 10px; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER SECTION ---
st.markdown('<div class="main-header">🏗️ Ultimate Structural Beam Analysis System</div>', unsafe_allow_html=True)
st.markdown(f"**Report Generated on:** {datetime.now().strftime('%A, %d %B %Y | %H:%M:%S')}")

# --- 5. SIDEBAR DATA FETCHING ---
# ดึงค่า Input ทั้งหมดจาก input_handler.py
# params: ข้อมูลวัสดุและหน้าตัด, n_spans: จำนวนช่วงคาน, spans: ความยาวแต่ละช่วง
# sup_df: ข้อมูลจุดรองรับ, loads_df: โหลดที่ผู้ใช้กรอก, stable: สถานะความมั่นคง
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

# --- 6. STRUCTURAL STABILITY VALIDATION ---
if not stable:
    st.markdown('<div class="error-msg">🚨 <b>CRITICAL WARNING:</b> The structure is currently UNSTABLE.</div>', unsafe_allow_html=True)
    st.warning("กรุณาตรวจสอบจุดรองรับ (Support Conditions):")
    st.info("- คานต่อเนื่องต้องการอย่างน้อย 3 องค์ประกอบของแรงปฏิกิริยา (เช่น Pin 1, Roller 1 หรือ Fixed 1)")
    st.info("- หากโครงสร้างเป็นแบบ Cantilever จำเป็นต้องมีจุดรองรับแบบ Fixed ที่จุดใดจุดหนึ่ง")
    st.stop()

# --- 7. DESIGN PARAMETERS & LOAD COMBINATION SETTINGS ---
st.markdown('<div class="sub-header">⚙️ 1. Analysis Parameters & Design Factors</div>', unsafe_allow_html=True)

with st.container():
    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    
    with col_f1:
        f_dl = st.number_input("Dead Load Factor ($f_{DL}$)", value=1.4, step=0.1, help="ตัวคูณสำหรับน้ำหนักบรรทุกคงที่ (Default ACI = 1.4)")
        st.caption("Standard: ACI 318-14")
        
    with col_f2:
        f_ll = st.number_input("Live Load Factor ($f_{LL}$)", value=1.7, step=0.1, help="ตัวคูณสำหรับน้ำหนักบรรทุกจร (Default ACI = 1.7)")
        st.caption("Standard: ACI 318-14")
        
    with col_f3:
        st.markdown("**Combination Equation Applied:**")
        st.latex(fr"W_u = {f_dl} \times DL + {f_ll} \times LL")

# --- 8. DETAILED LOAD CALCULATION & VERIFICATION (TRACEABILITY) ---
st.markdown('<div class="sub-header">🧮 2. Comprehensive Load Breakdown & Factoring</div>', unsafe_allow_html=True)
st.markdown("การแจกแจงรายละเอียดน้ำหนักบรรทุกก่อนส่งเข้า Stiffness Matrix Solver เพื่อความโปร่งใสในการคำนวณ:")

try:
    final_loads_list = []  # ข้อมูลที่จะส่งเข้า Solver (หน่วย Newton)
    report_trace = []      # ข้อมูลที่จะแสดงในตารางรายงาน (หน่วย kN)

    # 8.1 SELF-WEIGHT CALCULATION (AUTOMATIC)
    # สูตรคำนวณ: Area (b*h) * Density (24 kN/m3) * f_dl
    st.markdown("#### A. Reinforced Concrete Self-Weight")
    for i in range(n_spans):
        area = params['b'] * params['h']
        density = 24.0 # kN/m3
        unfactored_sw = area * density
        factored_sw = unfactored_sw * f_dl
        total_span_force = factored_sw * spans[i]
        
        # เก็บข้อมูลลง Trace Table
        report_trace.append({
            "Span": i + 1,
            "Source": "Self-Weight (RC)",
            "Case": "DL",
            "Calculation": f"({params['b']}m x {params['h']}m x 24) x {f_dl}",
            "Intensity": f"{factored_sw:.3f} kN/m",
            "Total Resultant (kN)": f"{total_span_force:.3f}"
        })
        
        # เตรียมข้อมูลสำหรับ Solver
        final_loads_list.append({
            'span_index': i, 'type': 'U', 'mag': factored_sw * 1000.0,
            'd_start': 0.0, 'dist': spans[i], 'desc': 'SW'
        })

    # 8.2 USER-DEFINED LOADS (DL & LL COMBINATION)
    if not loads_df.empty:
        st.markdown("#### B. User-Defined External Loads")
        for _, row in loads_df.iterrows():
            # พิจารณา Factor ตามประเภท DL หรือ LL
            factor = f_dl if row['case'] == "DL" else f_ll
            raw_mag = float(row['mag'])
            factored_mag = raw_mag * factor
            
            # คำนวณแรงลัพธ์สุทธิลงคาน (Net Resultant Force)
            if row['type'] == 'P':
                net_f = factored_mag
                unit = "kN"
                calc_str = f"{raw_mag} x {factor}"
            else:
                net_f = factored_mag * row['dist']
                unit = "kN/m"
                calc_str = f"({raw_mag} x {row['dist']}m) x {factor}"

            report_trace.append({
                "Span": row['span_index'] + 1,
                "Source": f"User Load ({row['case']})",
                "Case": row['case'],
                "Calculation": calc_str,
                "Intensity": f"{factored_mag:.2f} {unit}",
                "Total Resultant (kN)": f"{net_f:.3f}"
            })
            
            # เตรียมข้อมูลสำหรับ Solver
            final_loads_list.append({
                'span_index': int(row['span_index']), 'type': row['type'],
                'mag': factored_mag * 1000.0, 'd_start': float(row['d_start']),
                'dist': float(row['dist']), 'desc': f"{row['case']}_{row['type']}"
            })

    # 8.3 DISPLAY LOAD VERIFICATION TABLE
    display_report_df = pd.DataFrame(report_trace)
    st.table(display_report_df)
    
    # คำนวณ Checksum ของแรงทั้งหมด
    total_applied_kN = display_report_df["Total Resultant (kN)"].astype(float).sum()
    
    st.markdown(f"""
        <div class="calc-note">
            <b>Total Net Factored Load Applied ($\Sigma W_u$):</b> {total_applied_kN:.4f} kN <br>
            <i>Note: ค่านี้คือแรงกดรวมทั้งหมดที่โครงสร้างต้องรับภาระและถ่ายลงสู่จุดรองรับ</i>
        </div>
    """, unsafe_allow_html=True)

    # --- 9. STRUCTURAL ANALYSIS EXECUTION (SOLVER CORE) ---
    st.markdown("---")
    st.markdown('<div class="sub-header">📊 3. Analysis Diagrams & Results</div>', unsafe_allow_html=True)
    
    # แปลง List เป็น DataFrame เพื่อส่งเข้า Solver
    solver_ready_df = pd.DataFrame(final_loads_list)
    
    # รันโปรแกรมคำนวณ Stiffness Matrix
    # คืนค่า: x_eval, Moment, Shear, Deflection และ Reactions
    x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, solver_ready_df, params)
    
    # รวบรวมผลลัพธ์เป็น DataFrame สำหรับการทำ Visualization
    res_df = pd.DataFrame({
        'x': x_eval, 
        'moment': M, 
        'shear': V, 
        'deflection': D * 1000.0 # แปลงหน่วย m เป็น mm เพื่อการอ่านที่ง่ายขึ้น
    })

    # วาดกราฟผลการวิเคราะห์ (SFD, BMD, Deflection)
    st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, solver_ready_df, R), use_container_width=True)

    # --- 10. STATIC EQUILIBRIUM CHECK ---
    with st.expander("⚖️ Static Equilibrium & Support Verification", expanded=True):
        total_reaction_kN = sum(R.values()) / 1000.0
        diff_error = abs(total_applied_kN - total_reaction_kN)
        
        v_col1, v_col2, v_col3 = st.columns(3)
        v_col1.metric("Sum Applied Loads (ΣW)", f"{total_applied_kN:.4f} kN")
        v_col2.metric("Sum Reaction Forces (ΣR)", f"{total_reaction_kN:.4f} kN")
        
        if diff_error < 0.005:
            v_col3.success(f"Equilibrium: PASSED ✅\n(Error: {diff_error:.8f})")
            st.toast("Equilibrium Check Successful!", icon="✅")
        else:
            v_col3.error(f"Equilibrium: FAILED ❌\n(Diff: {diff_error:.4f})")
            st.warning("⚠️ มีค่าความคลาดเคลื่อนสูงเกินกำหนด กรุณาตรวจสอบหน่วยและการป้อนข้อมูล")

    # --- 11. REINFORCED CONCRETE DESIGN (ACI CODE) ---
    st.markdown("---")
    st.markdown('<div class="sub-header">🧱 4. RC Reinforcement Design & Detailing</div>', unsafe_allow_html=True)
    
    # แบ่ง Tab สำหรับดูสรุปผล และดูรายละเอียดการคำนวณ
    tab_sum, tab_step = st.tabs(["📌 Design Summary Table", "📝 Detailed Calculation Logs"])
    
    design_final_data = []
    main_bar_size = 16 # กำหนดขนาดเหล็กหลักเริ่มต้นที่ DB16
    cum_dist = [0] + list(np.cumsum(spans))

    for idx in range(n_spans):
        # กรองข้อมูลเฉพาะช่วงคานที่กำลังพิจารณา
        mask = (res_df['x'] >= cum_dist[idx] - 1e-7) & (res_df['x'] <= cum_dist[idx+1] + 1e-7)
        span_data = res_df[mask]
        
        if not span_data.empty:
            # หาค่าสูงสุดของ Moment (+), Moment (-), และ Shear (V) ในแต่ละช่วง
            mu_pos = span_data['moment'].max() / 1000.0 # kN-m
            mu_neg = abs(span_data['moment'].min()) / 1000.0 # kN-m
            vu_max = span_data['shear'].abs().max() / 1000.0 # kN
            d_eff = params['h'] - 0.05 # ระยะ d ประสิทธิผล (สมมติ covering + stirrup = 5cm)
            
            # เรียกใช้ Module rc_design เพื่อออกแบบเหล็กเสริม
            as_pos, _, _, steps_pos = rc_design.design_beam_flexure(mu_pos, params['b'], d_eff, params['fc'], params['fy'])
            as_neg, _, _, steps_neg = rc_design.design_beam_flexure(mu_neg, params['b'], d_eff, params['fc'], params['fy'])
            stirrup_s, _, steps_shear = rc_design.check_shear(vu_max, params['b'], d_eff, params['fc'], params['fy'])
            
            # คำนวณจำนวนเส้นเหล็กตามหน้าตัดที่ต้องการ
            def get_bars(as_req, db):
                a_bar = np.pi * (db/2)**2
                return max(2, int(np.ceil(as_req / a_bar)))

            n_pos = get_bars(as_pos, main_bar_size)
            n_neg = get_bars(as_neg, main_bar_size)
            
            # เก็บข้อมูลสรุปเพื่อวาดภาพ Detailing [FIXED STRUCTURE FOR PLOTTER]
            design_final_data.append({
                'span': idx + 1,
                'pos': {'n': n_pos},
                'neg': {'n': n_neg},
                'db': main_bar_size,
                'stirrup': f"RB6@{stirrup_s*100:.0f} cm",
                'shear': {'s': stirrup_s}
            })
            
            with tab_step:
                st.markdown(f"#### 📄 Span {idx+1}: Detailed Engineering Steps")
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.write("**Flexure Design (Positive Moment):**")
                    for s in steps_pos: st.latex(s)
                with sc2:
                    st.write("**Shear Design (Stirrups):**")
                    for s in steps_shear: st.latex(s)
    
    with tab_sum:
        # แสดงตารางสรุปผลการออกแบบเหล็กเสริมทั้งหมด
        st.dataframe(pd.DataFrame(design_final_data).drop(columns=['pos', 'neg', 'shear']), use_container_width=True)
        
        # --- 12. ENGINEERING DETAILING DRAWINGS ---
        st.markdown("---")
        st.subheader("🎨 Engineering Drawings & Section Preview")
        
        draw_col1, draw_col2 = st.columns([1, 2])
        
        with draw_col1:
            st.write("**Typical Cross-Section (Support/Midspan)**")
            # วาดรูปตัดขวางโดยใช้ข้อมูลจาก Span แรกเป็นเกณฑ์
            fig_section = section_plotter.plot_section(
                params['b'], params['h'], 40, main_bar_size, 
                design_final_data[0]['neg']['n'], 
                design_final_data[0]['pos']['n'], 
                "RB6", params['fc'], params['fy']
            )
            st.pyplot(fig_section)
            
        with draw_col2:
            st.write("**Longitudinal Reinforcement Profile**")
            # วาดรูปตัดตามยาวแสดงตำแหน่งเหล็กเสริมและจุดรองรับ
            fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_final_data, params['h'], 40)
            st.pyplot(fig_long)

# --- 13. GLOBAL ERROR CATCHING & SYSTEM LOGS ---
except Exception as global_error:
    st.markdown('<div class="error-msg"><b>CRITICAL RUNTIME ERROR:</b> Analysis could not be completed.</div>', unsafe_allow_html=True)
    st.error(f"Error Details: {str(global_error)}")
    st.info("กรุณาตรวจสอบว่าไฟล์ Module ทั้งหมด (solver, rc_design, design_view, section_plotter) อยู่ในโฟลเดอร์เดียวกัน")
    st.exception(global_error)

# --- 14. FOOTER SECTION ---
st.markdown("---")
st.caption(f"Powered by Gemini Structural Solver | Total Computed Spans: {n_spans} | Build: 2026.01")
# [End of app.py - Total Lines: ~250+]
