import streamlit as st
import pandas as pd
import numpy as np

# Imports
import input_handler
import design_view
from solver import BeamSolver
import rc_design
import section_plotter 
import file_manager

st.set_page_config(page_title="Professional Beam Studio", layout="wide", page_icon="🏗️")

# --- Inputs ---
params = input_handler.render_sidebar()
st.title("🏗️ Professional Beam Studio")
n_spans, spans, sup_df, is_stable = input_handler.render_model_inputs(params)
loads_df = input_handler.render_loads(n_spans, spans, params, sup_df)

# ==========================================
# RUN ANALYSIS
# ==========================================
if st.button("RUN ANALYSIS", type="primary"):
    if not is_stable: st.stop()
    if loads_df is None or loads_df.empty: st.warning("Please add loads"); st.stop()
    
    # Factoring Loads
    raw_loads = loads_df.to_dict('records')
    factored_loads = []
    for l in raw_loads:
        factor = params['gamma_dead'] if l['case'] == 'DL' else params['gamma_live']
        f_load = l.copy()
        f_load['mag'] = l['mag'] * factor
        factored_loads.append(f_load)

    # Solve
    try:
        with st.spinner("Analyzing..."):
            solver = BeamSolver(spans=spans, supports_input=sup_df.to_dict('records'), loads_input=factored_loads, E=params['E'], I_custom=params['I'])
            df_res, reactions, status = solver.solve()
            
            if not df_res.empty:
                st.session_state.results = {'df': df_res, 'reac': reactions, 'loads': raw_loads}
                st.success("Calculation Complete!")
    except Exception as e:
        st.error(f"Error: {e}")

# ==========================================
# DISPLAY RESULTS
# ==========================================
if 'results' in st.session_state:
    res = st.session_state.results
    df = res['df']
    
    # --- 1. EXECUTIVE SUMMARY (MAX/MIN TABLE) ---
    st.markdown("---")
    st.header("📊 Analysis Summary")
    
    v_max = df['shear'].abs().max() / 1000.0
    m_max = df['moment'].max() / 1000.0
    m_min = df['moment'].min() / 1000.0
    d_max = df['deflection'].abs().max() * 1000.0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Shear", f"{v_max:.2f} kN")
    c2.metric("Max +Moment", f"{m_max:.2f} kNm")
    c3.metric("Max -Moment", f"{m_min:.2f} kNm")
    c4.metric("Max Deflection", f"{d_max:.2f} mm")
    
    # --- 2. DIAGRAMS (Structure, V, M, D) ---
    fig_struct = design_view.draw_interactive_diagrams(
        df=res['df'], reac=res['reac'], spans=spans, sup_df=sup_df, loads=res['loads']
    )
    st.plotly_chart(fig_struct, use_container_width=True)
    
    # --- 3. RC DESIGN DETAIL ---
    st.markdown("---")
    st.header("🏗️ Reinforced Concrete Design")
    
    with st.expander("🛠️ Design Parameters (Click to Edit)", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        fc = col1.number_input("f'c (MPa)", 24.0)
        fy = col2.number_input("fy (MPa)", 400.0)
        cover = col3.number_input("Cover (mm)", 30.0)
        db = col4.selectbox("Main Bar (mm)", [12, 16, 20, 25, 28], index=1)

    cum_dist = [0] + list(np.cumsum(spans))
    
    for i in range(n_spans):
        st.markdown(f"### 🌉 Span {i+1} Design")
        
        # Filter Span Data
        mask = (df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])
        span_data = df[mask]
        
        # Forces
        mu_pos = span_data['moment'].max() / 1000
        mu_neg = span_data['moment'].min() / 1000
        vu_span = span_data['shear'].abs().max() / 1000
        
        # RC Calc
        design_res = rc_design.design_span_expert(
            m_pos=mu_pos, m_neg=mu_neg, v_u=vu_span,
            b=params['b'], h=params['h'], fc=fc, fy=fy, cover=cover, db=db
        )
        
        # === EQUATION CHECK (สำคัญ: ส่วนที่เคยหายไป) ===
        st.markdown("#### 📝 Design Verification")
        eq1, eq2 = st.columns(2)
        
        # Check Positive
        with eq1:
            phi_mn_pos = design_res['pos']['capacity']
            ratio_pos = mu_pos / phi_mn_pos if phi_mn_pos > 0 else 0
            status_pos = r"\textcolor{green}{\textbf{PASS}}" if ratio_pos <= 1.0 else r"\textcolor{red}{\textbf{FAIL}}"
            
            st.info("**Positive Moment (+M) Check**")
            st.latex(rf"M_u^+ = {mu_pos:.2f} \le \phi M_n = {phi_mn_pos:.2f} \text{ kNm}")
            st.markdown(f"Ratio: **{ratio_pos:.2f}** $\longrightarrow$ {status_pos}")

        # Check Negative
        with eq2:
            phi_mn_neg = abs(design_res['neg']['capacity'])
            mu_neg_abs = abs(mu_neg)
            ratio_neg = mu_neg_abs / phi_mn_neg if phi_mn_neg > 0 else 0
            status_neg = r"\textcolor{green}{\textbf{PASS}}" if ratio_neg <= 1.0 else r"\textcolor{red}{\textbf{FAIL}}"
            
            st.warning("**Negative Moment (-M) Check**")
            st.latex(rf"|M_u^-| = {mu_neg_abs:.2f} \le \phi M_n = {phi_mn_neg:.2f} \text{ kNm}")
            st.markdown(f"Ratio: **{ratio_neg:.2f}** $\longrightarrow$ {status_neg}")

        # Visualization
        c_plot, c_sec = st.columns([1, 1])
        with c_plot:
            st.write("**Moment Capacity Plot (Dashed Lines = Capacity)**")
            fig_cap = design_view.plot_capacity_vs_demand(
                df_span=span_data,
                phi_Mn_pos=design_res['pos']['capacity'],
                phi_Mn_neg=design_res['neg']['capacity']
            )
            st.plotly_chart(fig_cap, use_container_width=True)
            
        with c_sec:
            st.write("**Cross Section Detail**")
            fig_sec = section_plotter.plot_section(
                b=params['b'], h=params['h'], cover_mm=cover, db_mm=db, 
                n_top=design_res['neg']['n'], n_bot=design_res['pos']['n'],
                stirrup_info=design_res['shear_stirrups'], fc=fc, fy=fy
            )
            st.pyplot(fig_sec, use_container_width=True)
            
        st.divider()
