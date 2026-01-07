import streamlit as st
import pandas as pd
import numpy as np

# Import custom modules
import input_handler
import solver
import rc_design
import design_view
import section_plotter

st.set_page_config(page_title="Beam Analysis & Design", layout="wide")
st.title("🏗️ RC Beam Analysis & Design Pro")

# --- Sidebar Inputs ---
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("Structure is unstable! Please check supports (Must have at least 3 reaction components).")
else:
    # --- 1. Analyze ---
    x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, loads_df, params)
    
    # Store results
    res_df = pd.DataFrame({
        'x': x_eval,
        'moment': M,
        'shear': V,
        'deflection': D * 1000 # Convert m to mm
    })
    
    # --- 2. Visualization ---
    st.header("1. Analysis Results")
    fig = design_view.plot_analysis_results(res_df, spans, sup_df, loads_df, R)
    st.plotly_chart(fig, use_container_width=True)
    
    # --- 3. RC Design & Calculation Note ---
    st.header("2. Reinforced Concrete Design")
    
    # Material Summary
    st.markdown(f"**Material:** f'c = {params['fc']} MPa, fy = {params['fy']} MPa | **Section:** {params['b']*100:.0f}x{params['h']*100:.0f} cm")
    
    design_res = []
    
    # Loop through spans for design
    span_start = 0
    for i, span_len in enumerate(spans):
        span_end = span_start + span_len
        
        # Get Max Moment in this span (Positive & Negative)
        mask = (res_df['x'] >= span_start) & (res_df['x'] <= span_end)
        span_data = res_df[mask]
        
        mu_pos = span_data['moment'].max() / 1000 # kNm
        mu_neg = abs(span_data['moment'].min()) / 1000 # kNm
        vu_max = span_data['shear'].abs().max() / 1000 # kN
        
        # Effective depth
        cover = 40 # mm
        db_est = 16 # mm
        d = params['h'] - (cover + 10 + db_est/2)/1000
        
        # --- CALCULATION (Modified to get steps) ---
        # 1. Positive Moment Design
        As_pos, rho_pos, stat_pos, steps_pos = rc_design.design_beam_flexure(
            mu_pos, params['b'], d, params['fc'], params['fy']
        )
        
        # 2. Negative Moment Design
        As_neg, rho_neg, stat_neg, steps_neg = rc_design.design_beam_flexure(
            mu_neg, params['b'], d, params['fc'], params['fy']
        )
        
        # 3. Shear Design
        s_req, stat_shear, steps_shear = rc_design.check_shear(
            vu_max, params['b'], d, params['fc'], params['fy']
        )
        
        # Convert As to Rebar count (Example DB16)
        n_pos = int(np.ceil(As_pos / 201)) if As_pos > 0 else 2
        n_neg = int(np.ceil(As_neg / 201)) if As_neg > 0 else 2
        
        design_res.append({
            'span': i+1,
            'pos': {'As': As_pos, 'n': max(2, n_pos)},
            'neg': {'As': As_neg, 'n': max(2, n_neg)},
            'shear': {'s': s_req, 'status': stat_shear},
            'db': 16
        })
        
        # --- DISPLAY CALCULATION NOTE ---
        with st.expander(f"📘 Span {i+1} Calculation Note", expanded=False):
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Flexure (Bottom Steel)")
                for step in steps_pos:
                    st.latex(step)
                st.info(f"Select: {max(2, n_pos)}-DB16 (As = {max(2, n_pos)*201} mm²)")
                    
            with c2:
                st.subheader("Flexure (Top Steel)")
                for step in steps_neg:
                    st.latex(step)
                st.info(f"Select: {max(2, n_neg)}-DB16 (As = {max(2, n_neg)*201} mm²)")
            
            st.divider()
            st.subheader("Shear Design")
            for step in steps_shear:
                st.latex(step)
            st.success(f"Result: Stirrup RB6 @ {min(200, int(s_req if s_req else 200))} mm")

        span_start += span_len

    # --- 4. Detailing Visualization ---
    st.header("3. Detailing")
    
    # 1. Section View (Example for Span 1)
    st.subheader("Cross-Section (Span 1 Mid-span)")
    fig_sec = section_plotter.plot_section(
        params['b'], params['h'], 40, 16, 
        design_res[0]['neg']['n'], design_res[0]['pos']['n'], 
        "RB6@200", params['fc'], params['fy']
    )
    st.pyplot(fig_sec)
    
    # 2. Long Section
    st.subheader("Longitudinal Section")
    fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_res, params['h'], 40)
    st.pyplot(fig_long)
