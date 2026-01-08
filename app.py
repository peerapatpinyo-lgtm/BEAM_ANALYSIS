# ===========================================================================================
# 🏗️ PROFESSIONAL RC BEAM ANALYSIS & DESIGN SYSTEM (FULL VERSION)
# ===========================================================================================
# ระบบวิเคราะห์และออกแบบคานคอนกรีตเสริมเหล็กตามมาตรฐานวิศวกรรม
# พัฒนาโดย: Gemini Engineering Suite (2026 Edition)
# รองรับ: Load Combinations (SDM), Finite Element Analysis, และ RC Detailing Report
# ===========================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# --- 1. CORE ENGINE MODULES IMPORT ---
# นำเข้ามอดูลภายนอกที่พัฒนาแยกไว้ เพื่อรักษาโครงสร้างของโปรแกรม
try:
    import input_handler
    import solver
    import rc_design
    import design_view
    import section_plotter
except ImportError as e:
    st.error(f"❌ ไม่พบมอดูลสำคัญ: {e}")
    st.stop()

# --- 2. GLOBAL PAGE CONFIGURATION ---
# ตั้งค่า Layout หน้าเว็บให้กว้างและเหมาะสมกับรายงานวิศวกรรม
st.set_page_config(
    page_title="RC Beam Pro - Comprehensive Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. ADVANCED UI CUSTOMIZATION (CSS) ---
# ตกแต่งหน้าตาแอปพลิเคชันให้ดูเป็นมืออาชีพและอ่านง่าย
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Sarabun', sans-serif; }
    .main-title { font-size: 36px; color: #1e3a8a; font-weight: bold; border-bottom: 4px solid #3b82f6; padding-bottom: 10px; }
    .report-section { background-color: #f8fafc; border-left: 5px solid #1e40af; padding: 20px; border-radius: 5px; margin: 20px 0; }
    .metric-container { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .equilibrium-passed { color: #059669; font-weight: bold; }
    .equilibrium-failed { color: #dc2626; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. APPLICATION HEADER ---
st.markdown('<div class="main-title">🏗️ ระบบวิเคราะห์และออกแบบคาน RC (Professional Report)</div>', unsafe_allow_html=True)
st.write(f"📊 **สถานะระบบ:** พร้อมประมวลผล | **วันที่ออกรายงาน:** {datetime.now().strftime('%d/%m/%Y | %H:%M:%S')}")

# --- 5. INITIAL DATA FETCHING ---
# รับข้อมูลจาก Sidebar (Input Handler)
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

# --- 6. STRUCTURAL STABILITY VALIDATION ---
# ตรวจสอบว่าโครงสร้างมีเสถียรภาพหรือไม่ก่อนดำเนินการต่อ
if not stable:
    st.markdown("""
        <div style="background-color: #fef2f2; border: 1px solid #ef4444; padding: 20px; border-radius: 10px;">
            <h3 style="color: #b91c1c;">🚨 ตรวจพบความไม่มั่นคงของโครงสร้าง (Instability Detected)</h3>
            <p>กรุณาตรวจสอบจุดรองรับ (Support) ของท่านตามกฎทางวิศวกรรม:</p>
            <ul>
                <li>คานต้องมีแรงปฏิกิริยาอย่างน้อย 3 แนวทาง (เช่น Pin 1 + Roller 1)</li>
                <li>กรณีคานปลายยื่น (Cantilever) ต้องมีจุดยึดรั้งแบบ Fixed</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- 7. DESIGN BASIS & LOAD COMBINATION CONFIGURATION ---
# ส่วนของการตั้งค่าตัวคูณน้ำหนักบรรทุกตามมาตรฐาน Strength Design Method (SDM)
st.markdown("### ⚙️ 1. เกณฑ์การตั้งค่าและวิเคราะห์ (Design Basis)")
with st.container():
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        f_dl = st.number_input("ตัวคูณ Dead Load ($f_{DL}$)", value=1.4, step=0.1, help="มาตรฐาน ACI/EIT ใช้ 1.4")
        st.caption("กำหนดค่าตัวคูณน้ำหนักบรรทุกคงที่")
        
    with c2:
        f_ll = st.number_input("ตัวคูณ Live Load ($f_{LL}$)", value=1.7, step=0.1, help="มาตรฐาน ACI/EIT ใช้ 1.7")
        st.caption("กำหนดค่าตัวคูณน้ำหนักบรรทุกจร")
        
    with c3:
        st.markdown("**Load Combination Applied:**")
        st.latex(fr"U = {f_dl} \cdot DL + {f_ll} \cdot LL")
        st.info("ระบบจะใช้ค่าแรงที่คูณ Factor แล้ว (Factored Load) ในการวิเคราะห์ Stiffness Matrix")

# --- 8. COMPREHENSIVE LOAD FACTORING & TRACEABILITY ---
# ส่วนสำคัญ: การแจกแจงรายการโหลดทั้งหมดเพื่อให้ผู้ใช้งานตรวจสอบความถูกต้องได้
st.markdown("---")
st.markdown("### 🧮 2. รายละเอียดการรวมน้ำหนักบรรทุก (Load Traceability)")

try:
    final_analysis_loads = [] # เก็บข้อมูลหน่วย Newton สำหรับ Solver
    load_trace_report = []   # เก็บข้อมูลสำหรับแสดงผลในตารางรายงาน (หน่วย kN)

    # 8.1 การคำนวณน้ำหนักตัวเองของโครงสร้าง (Self-Weight Analysis)
    # สูตร: b(m) * h(m) * 2400 kg/m3 * 9.81 m/s2 / 1000 = kN/m
    for i in range(n_spans):
        b_m = params['b']
        h_m = params['h']
        sw_base = b_m * h_m * 24.0 # kN/m
        sw_factored = sw_base * f_dl
        span_len = spans[i]
        resultant_sw = sw_factored * span_len
        
        load_trace_report.append({
            "Span": i + 1,
            "Load Source": "Self-Weight (RC)",
            "Case": "DL",
            "Calculation Formula": f"({b_m} x {h_m} x 24.0) x {f_dl}",
            "Design Value": f"{sw_factored:.3f} kN/m",
            "Net Force (kN)": f"{resultant_sw:.3f}"
        })
        
        # ส่งค่าเข้า Solver (ใช้ชื่อ 'SW' สั้นๆ เพื่อเลี่ยงตัวเลขทับกันบนกราฟ)
        final_analysis_loads.append({
            'span_index': i, 'type': 'U', 'mag': sw_factored * 1000.0,
            'd_start': 0.0, 'dist': span_len, 'desc': 'SW'
        })

    # 8.2 การจัดการน้ำหนักบรรทุกจากผู้ใช้งาน (Applied User Loads)
    if not loads_df.empty:
        for _, row in loads_df.iterrows():
            current_f = f_dl if row['case'] == "DL" else f_ll
            raw_val = float(row['mag'])
            factored_val = raw_val * current_f
            
            # คำนวณแรงลัพธ์สุทธิ (Total Force)
            if row['type'] == 'P':
                net_res = factored_val
                unit_label = "kN"
                formula = f"{raw_val} x {current_f}"
            else:
                net_res = factored_val * row['dist']
                unit_label = "kN/m"
                formula = f"({raw_val} x {row['dist']}m) x {current_f}"

            load_trace_report.append({
                "Span": row['span_index'] + 1,
                "Load Source": f"User Added ({row['case']})",
                "Case": row['case'],
                "Calculation Formula": formula,
                "Design Value": f"{factored_val:.2f} {unit_label}",
                "Net Force (kN)": f"{net_res:.3f}"
            })
            
            # เก็บข้อมูลลง List สำหรับส่งต่อให้ Solver
            final_analysis_loads.append({
                'span_index': int(row['span_index']), 'type': row['type'],
                'mag': factored_val * 1000.0, 'd_start': float(row['d_start']),
                'dist': float(row['dist']), 'desc': f"{row['case']}"
            })

    # 8.3 การแสดงผลตารางรายงานน้ำหนัก (Verification Table)
    report_df = pd.DataFrame(load_trace_report)
    st.table(report_df)
    
    total_w_sum = report_df["Net Force (kN)"].astype(float).sum()
    st.markdown(f"""
        <div style="background-color: #f1f5f9; padding: 15px; border-radius: 8px; border: 1px solid #cbd5e1;">
            <strong>ยอดรวมน้ำหนักบรรทุกออกแบบทั้งหมด (Total Factored Load):</strong> 
            <span style="font-size: 20px; color: #1e40af;">{total_w_sum:.4f} kN</span>
        </div>
    """, unsafe_allow_html=True)

    # --- 9. FINITE ELEMENT ANALYSIS EXECUTION (SOLVER CORE) ---
    st.markdown("---")
    st.markdown("### 📊 3. ผลการวิเคราะห์ทางโครงสร้าง (Structural Results)")
    
    with st.spinner('กำลังประมวลผล Stiffness Matrix...'):
        solver_input = pd.DataFrame(final_analysis_loads)
        # รันการวิเคราะห์หลัก
        x_pts, M_vals, V_vals, D_vals, R_vals = solver.solve_beam(spans, sup_df, solver_input, params)
        
        # รวบรวมข้อมูลลง DataFrame เพื่อใช้ในการ Plot กราฟ
        analysis_db = pd.DataFrame({
            'x': x_pts, 
            'moment': M_vals, 
            'shear': V_vals, 
            'deflection': D_vals * 1000.0 # m -> mm
        })

    # --- 10. HANDLING OVERLAPPING LABELS & GRAPHING ---
    # เรียกใช้ฟังก์ชัน Plot ที่ถูกปรับปรุงให้รับค่าเพื่อลดการซ้อนทับของเลข
    st.plotly_chart(design_view.plot_analysis_results(analysis_db, spans, sup_df, solver_input, R_vals), use_container_width=True)

    # --- 11. STATIC EQUILIBRIUM VERIFICATION ---
    # ส่วนการตรวจสอบความสมดุลของแรงเพื่อให้แน่ใจว่าผลการคำนวณถูกต้อง 100%
    st.markdown("#### ⚖️ การตรวจสอบสมดุลของแรง (Equilibrium Check)")
    total_reac_sum = sum(R_vals.values()) / 1000.0
    err_val = abs(total_w_sum - total_reac_sum)
    
    eq_c1, eq_c2, eq_c3 = st.columns(3)
    with eq_c1:
        st.metric("แรงกดลงทั้งหมด (ΣW)", f"{total_w_sum:.3f} kN")
    with eq_c2:
        st.metric("แรงปฏิกิริยาทั้งหมด (ΣR)", f"{total_reac_sum:.3f} kN")
    with eq_c3:
        if err_val < 0.001:
            st.markdown('<p class="equilibrium-passed">✅ สมดุลแรงถูกต้อง (MATCHED)</p>', unsafe_allow_html=True)
            st.write(f"ความคลาดเคลื่อน: {err_val:.8f}")
        else:
            st.markdown('<p class="equilibrium-failed">❌ แรงไม่สมดุล (MISMATCHED)</p>', unsafe_allow_html=True)
            st.write(f"ส่วนต่าง: {err_val:.4f}")

    # --- 12. RC DESIGN & DETAILING (ACI 318-14 BASED) ---
    st.markdown("---")
    st.markdown("### 🧱 4. การออกแบบเหล็กเสริมและรายละเอียด (RC Design)")
    
    tab_sum, tab_calc = st.tabs(["📝 ตารางสรุปการออกแบบ", "🧮 รายละเอียดการคำนวณแยกช่วง"])
    
    final_detailing_list = []
    main_bar_db = 16 # ขนาดเหล็กหลักมาตรฐาน
    cum_offsets = [0] + list(np.cumsum(spans))

    for i in range(n_spans):
        # ดึงข้อมูลภายใน Span นั้นๆ
        mask = (analysis_db['x'] >= cum_offsets[i] - 1e-8) & (analysis_db['x'] <= cum_offsets[i+1] + 1e-8)
        span_segment = analysis_db[mask]
        
        if not span_segment.empty:
            mu_pos_max = span_segment['moment'].max() / 1000.0
            mu_neg_max = abs(span_segment['moment'].min()) / 1000.0
            vu_max_span = span_segment['shear'].abs().max() / 1000.0
            d_effective = params['h'] - 0.05
            
            # เรียกใช้ Module ออกแบบ
            as_pos, _, _, steps_p = rc_design.design_beam_flexure(mu_pos_max, params['b'], d_effective, params['fc'], params['fy'])
            as_neg, _, _, steps_n = rc_design.design_beam_flexure(mu_neg_max, params['b'], d_effective, params['fc'], params['fy'])
            s_stirrup, _, steps_v = rc_design.check_shear(vu_max_span, params['b'], d_effective, params['fc'], params['fy'])
            
            # ฟังก์ชันคำนวณจำนวนเส้นเหล็ก
            def calc_n_bars(area_req, db_size):
                area_single = np.pi * (db_size/2)**2
                return max(2, int(np.ceil(area_req / area_single)))

            n_p = calc_n_bars(as_pos, main_bar_db)
            n_n = calc_n_bars(as_neg, main_bar_db)
            
            # [FIXED KEYERROR] จัดโครงสร้างข้อมูลแบบ Nested ตามที่ section_plotter ต้องการ
            final_detailing_list.append({
                'span': i + 1,
                'pos': {'n': n_p},  # ป้องกัน KeyError: 'pos'
                'neg': {'n': n_n},  # ป้องกัน KeyError: 'neg'
                'db': main_bar_db,
                'stirrup': f"RB6@{s_stirrup*100:.0f} cm",
                'shear': {'s': s_stirrup}
            })
            
            with tab_calc:
                st.markdown(f"#### 📄 รายละเอียดช่วงที่ {i+1} (Span {i+1})")
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.write("**การออกแบบแรงดัด (Flexure):**")
                    for s in steps_p: st.latex(s)
                with sc2:
                    st.write("**การออกแบบแรงเฉือน (Shear):**")
                    for s in steps_v: st.latex(s)

    with tab_sum:
        # แสดงผลตารางสรุปแบบภาษาไทย
        summary_table = pd.DataFrame(final_detailing_list).drop(columns=['pos', 'neg', 'shear'])
        summary_table.columns = ['ช่วงที่', 'เหล็กบน', 'เหล็กล่าง', 'เหล็กปลอก']
        st.table(summary_table)
        
        # --- 13. ENGINEERING DRAWING RENDER ---
        st.markdown("#### 🎨 รูปแบบการเสริมเหล็ก (Engineering Detail)")
        plot_c1, plot_c2 = st.columns([1, 2])
        
        with plot_c1:
            st.write("**Cross Section**")
            fig_sect = section_plotter.plot_section(
                params['b'], params['h'], 40, main_bar_db, 
                final_detailing_list[0]['neg']['n'], 
                final_detailing_list[0]['pos']['n'], 
                "RB6", params['fc'], params['fy']
            )
            st.pyplot(fig_sect)
            
        with plot_c2:
            st.write("**Longitudinal Profile**")
            fig_long_sect = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, final_detailing_list, params['h'], 40)
            st.pyplot(fig_long_sect)

# --- 14. EXCEPTION HANDLING & SYSTEM LOG ---
except Exception as sys_err:
    st.error(f"❌ ระบบขัดข้อง: {str(sys_err)}")
    st.exception(sys_err)

# --- 15. REPORT FOOTER ---
st.markdown("---")
st.markdown("""
    <div class="footer-text">
        © 2026 Professional Beam Design Suite | พัฒนาตามมาตรฐาน ACI 318-14 และมาตรฐาน วสท. | 
        ประมวลผลด้วยเทคโนโลยี Finite Element Method (Matrix Stiffness)
    </div>
""", unsafe_allow_html=True)

# [End of app.py - Total Lines Approx: 280-300]
