import streamlit as st
import pandas as pd
import numpy as np

from solver import solve_beam
from input_handler import render_all_sidebar_inputs
from rc_design import design_beam_flexure, check_shear
from design_view import plot_analysis_results
from section_plotter import plot_section

st.set_page_config(page_title="RC Beam Pro", layout="wide")

# 1. Sidebar Inputs (รับ params, spans, sup_df, loads_df)
params, n_spans, spans, sup_df, loads_df, stable = render_all_sidebar_inputs()

if not stable:
    st.error("Structure is unstable.")
    st.stop()

if not loads_df.empty:
    # --- STEP 1: SOLVE ---
    # มั่นใจว่าโหลดเป็น N และ m (ใน input_handler ต้องแก้ให้ส่ง kN มาแล้วมาคูณ 1000 ที่นี่)
    loads_to_solve = loads_df.copy()
    loads_to_solve['mag'] = loads_to_solve['mag'] * 1000.0 
    
    x, m, v, delta, react = solve_beam(spans, sup_df, loads_to_solve, params)
    
    res_df = pd.DataFrame({'x': x, 'moment': m, 'shear': v, 'deflection': delta * 1000})

    # --- STEP 2: PLOT ANALYSIS ---
    st.header("Analysis Diagrams")
    fig_analysis = plot_analysis_results(res_df, spans, sup_df, loads_df, react)
    st.plotly_chart(fig_analysis, use_container_width=True)

    # --- STEP 3: DESIGN ---
    st.header("RC Design & Detailing")
    d_eff = params['h'] - 0.05
    db_main = 16 # mm
    design_results = []

    for i in range(n_spans):
        # Filter span data
        s_start, s_end = sum(spans[:i]), sum(spans[:i+1])
        mask = (res_df['x'] >= s_start) & (res_df['x'] <= s_end)
        
        mu_pos = res_df[mask]['moment'].max() / 1000.0
        mu_neg = abs(res_df[mask]['moment'].min()) / 1000.0
        vu_max = res_df[mask]['shear'].abs().max() / 1000.0

        # Call rc_design.py functions
        as_pos, _, _, steps_pos = design_beam_flexure(mu_pos, params['b'], d_eff, params['fc'], params['fy'])
        as_neg, _, _, steps_neg = design_beam_flexure(mu_neg, params['b'], d_eff, params['fc'], params['fy'])
        s_shear, v_status, v_steps = check_shear(vu_max, params['b'], d_eff, params['fc'], params['fy'])

        # Calculate number of bars (Corrected Units)
        area_db = np.pi * (db_main / 2)**2
        n_pos = int(max(2, np.ceil(as_pos / area_db)))
        n_neg = int(max(2, np.ceil(as_neg / area_db)))

        design_results.append({
            'span': i+1, 'db': db_main,
            'pos_n': n_pos, 'neg_n': n_neg, 'shear_s': s_shear,
            'steps_pos': steps_pos, 'steps_v': v_steps
        })

    # Display Cross-sections
    cols = st.columns(n_spans)
    for i, res in enumerate(design_results):
        with cols[i]:
            st.subheader(f"Span {i+1}")
            fig_sec = plot_section(params['b'], params['h'], 40, res['db'], res['neg_n'], res['pos_n'], None, params['fc'], params['fy'])
            st.pyplot(fig_sec)
            st.info(f"Top: {res['neg_n']}-DB{res['db']}\n\nBot: {res['pos_n']}-DB{res['db']}\n\nStirrup: @{res['shear_s']:.0f}mm")

    # Display Detailed Calcs
    with st.expander("Detailed Calculation Steps"):
        for res in design_results:
            st.write(f"### Span {res['span']}")
            for s in res['steps_pos']: st.latex(s)
            for s in res['steps_v']: st.latex(s)
