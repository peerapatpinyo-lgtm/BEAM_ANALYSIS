import streamlit as st
import pandas as pd
import numpy as np

# Import custom modules
from solver import solve_beam
from input_handler import render_all_sidebar_inputs
from rc_design import design_beam_flexure, check_shear
from design_view import plot_analysis_results
from section_plotter import plot_section, plot_longitudinal_section_detailed

# --- Page Config ---
st.set_page_config(page_title="RC Beam Expert", layout="wide")
st.title("🏗️ RC Beam Analysis & Design (Timoshenko Theory)")

# 1. Render Sidebar & Get Inputs
# params ประกอบด้วย: fc, fy, b, h, E (Pa), I (m4)
params, n_spans, spans, sup_df, loads_df, stable = render_all_sidebar_inputs()

if not stable:
    st.error("⚠️ Structure is unstable! Please check supports.")
    st.stop()

# 2. Process Loads & Solve (Internal Units: N, m, Pa)
if not loads_df.empty:
    # [IMPORTANT] Normalize loads for Solver: 
    # ตรวจสอบว่าใน input_handler เก็บเป็น kN มาแล้ว ดังนั้นคูณ 1000 ที่นี่ที่เดียว
    loads_to_solve = loads_df.copy()
    loads_to_solve['mag'] = loads_to_solve['mag'] * 1000.0  # kN -> N
    
    # คำนวณด้วย Finite Element Method
    x_total, m_total, v_total, d_total, reactions = solve_beam(
        spans, sup_df, loads_to_solve, params
    )
    
    # เก็บผลลัพธ์ลง DataFrame
    res_df = pd.DataFrame({
        'x': x_total,
        'moment': m_total,    # Unit: N-m
        'shear': v_total,     # Unit: N
        'deflection': d_total * 1000 # Unit: m -> mm (เพื่อวาดกราฟ)
    })

    # --- ANALYSIS VIEW ---
    st.header("1. Analysis Results")
    fig_analysis = plot_analysis_results(res_df, spans, sup_df, loads_df, reactions)
    st.plotly_chart(fig_analysis, use_container_width=True)

    # 3. RC Design (Unit Conversion: N-m -> kNm for RC function)
    st.header("2. Reinforced Concrete Design (ACI/EIT)")
    
    design_results = []
    # กำหนดค่าตัวแปรเบื้องต้น
    d_eff = params['h'] - 0.05 # m (Effective depth)
    db_main = 16 # mm
    
    # แบ่งการคำนวณเป็นช่วง (Span by Span)
    for i in range(n_spans):
        # ตัดข้อมูลเฉพาะ Span นั้นๆ
        span_range = (res_df['x'] >= sum(spans[:i])) & (res_df['x'] <= sum(spans[:i+1]))
        span_data = res_df[span_range]
        
        # หา Moment สูงสุด (Positive & Negative) ในหน่วย kNm
        max_m_pos = span_data['moment'].max() / 1000.0
        max_m_neg = span_data['moment'].min() / 1000.0
        max_v = span_data['shear'].abs().max() / 1000.0

        # คำนวณเหล็กเสริม
        as_pos, rho_pos, status_pos, steps_pos = design_beam_flexure(max_m_pos, params['b'], d_eff, params['fc'], params['fy'])
        as_neg, rho_neg, status_neg, steps_neg = design_beam_flexure(abs(max_m_neg), params['b'], d_eff, params['fc'], params['fy'])
        s_req, v_status, v_steps = check_shear(max_v, params['b'], d_eff, params['fc'], params['fy'])

        # [FIXED] สูตรหาจำนวนเส้นเหล็ก: As (mm2) / Area of 1 bar (mm2)
        def calc_n_bars(as_req, db):
            area_1_bar = np.pi * (db/2)**2
            return int(max(2, np.ceil(as_req / area_1_bar)))

        design_results.append({
            'span': i+1,
            'db': db_main,
            'pos': {'as': as_pos, 'n': calc_n_bars(as_pos, db_main), 'steps': steps_pos},
            'neg': {'as': as_neg, 'n': calc_n_bars(as_neg, db_main), 'steps': steps_neg},
            'shear': {'s': s_req, 'status': v_status, 'steps': v_steps}
        })

    # --- DISPLAY DESIGN ---
    cols = st.columns(n_spans)
    for i, res in enumerate(design_results):
        with cols[i]:
            st.subheader(f"Span {res['span']}")
            # วาดรูปหน้าตัด
            fig_sec = plot_section(params['b'], params['h'], 40, res['db'], res['neg']['n'], res['pos']['n'], None, params['fc'], params['fy'])
            st.pyplot(fig_sec)
            
            # สรุปผล
            st.write(f"**Bottom:** {res['pos']['n']}-DB{res['db']}")
            st.write(f"**Top:** {res['neg']['n']}-DB{res['db']}")
            st.write(f"**Stirrup:** RB6 @ {res['shear']['s']:.0f} mm")

    # --- DETAILED CALCULATION (Expander) ---
    with st.expander("📄 View Detailed Calculation Steps (LaTeX)"):
        for res in design_results:
            st.markdown(f"### Span {res['span']}")
            st.markdown("#### Flexure Design (Bottom)")
            for step in res['pos']['steps']: st.latex(step)
            st.markdown("---")
            st.markdown("#### Shear Design")
            for step in res['shear']['steps']: st.latex(step)

else:
    st.info("👋 Please add some loads in the sidebar to start analysis.")
