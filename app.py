import streamlit as st
import pandas as pd
import numpy as np
import input_handler
import design_view
from solver import BeamSolver
import rc_design
import section_plotter 
import file_manager

st.set_page_config(page_title="Professional Beam Studio", layout="wide", page_icon="🏗️")

params = input_handler.render_sidebar()
st.title("🏗️ Professional Beam Studio")
n_spans, spans, sup_df, is_stable = input_handler.render_model_inputs(params)
loads_df = input_handler.render_loads(n_spans, spans, params, sup_df)

if st.button("RUN ANALYSIS", type="primary"):
    if not is_stable: st.stop()
    if loads_df is None or loads_df.empty: st.warning("Please add loads"); st.stop()
    
    # Factoring
    raw_loads = loads_df.to_dict('records')
    factored_loads = []
    for l in raw_loads:
        factor = params['gamma_dead'] if l['case'] == 'DL' else params['gamma_live']
        f_load = l.copy()
        f_load['mag'] = l['mag'] * factor
        factored_loads.append(f_load)

    with st.spinner("Analyzing..."):
        try:
            solver = BeamSolver(spans=spans, supports_input=sup_df.to_dict('records'), loads_input=factored_loads, E=params['E'], I_custom=params['I'])
            df_res, reactions, status = solver.solve()
            if not df_res.empty:
                st.session_state.results = {'df': df_res, 'reac': reactions, 'loads': raw_loads}
        except Exception as e:
            st.error(f"Solver Error: {e}")

if 'results' in st.session_state:
    res = st.session_state.results
    df = res['df']
    
    # 1. SUMMARY
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
    
    # 2. DIAGRAMS
    fig_struct = design_view.draw_interactive_diagrams(res['df'], res['reac'], spans, sup_df, res['loads'])
    st.plotly_chart(fig_struct, use_container_width=True)
    
    # 3. RC DESIGN DETAIL
    st.markdown("---")
    st.header("🏗️ Detailed Reinforced Concrete Design")
    
    with st.expander("🛠️ Edit Material & Section Properties", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        fc = col1.number_input("f'c (MPa)", 24.0)
        fy = col2.number_input("fy (MPa)", 400.0)
        cover = col3.number_input("Cover (mm)", 30.0)
        db = col4.selectbox("Main Bar (mm)", [12, 16, 20, 25, 28], index=1)

    cum_dist = [0] + list(np.cumsum(spans))
    
    for i in range(n_spans):
        st.markdown(f"### 🌉 Span {i+1} Calculation Report")
        
        # Filter Data
        mask = (df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])
        span_data = df[mask]
        mu_pos = span_data['moment'].max() / 1000
        mu_neg = span_data['moment'].min() / 1000
        vu_span = span_data['shear'].abs().max() / 1000
        
        # Design Calc
        design_res = rc_design.design_span_expert(mu_pos, mu_neg, vu_span, params['b'], params['h'], fc, fy, cover, db)
        
        # === A. EQUATION CHECK (กลับมาแล้ว) ===
        st.markdown("#### ✅ Design Verification (Code Check)")
        chk1, chk2 = st.columns(2)
        
        # Positive Moment Check
        with chk1:
            phi_mn = design_res['pos']['capacity']
            ratio = mu_pos / phi_mn if phi_mn > 0 else 0
            res_txt = r"\textcolor{green}{\textbf{PASS}}" if ratio <= 1.0 else r"\textcolor{red}{\textbf{FAIL}}"
            st.info(f"**Positive Moment Check (+M)**")
            st.latex(rf"M_u^+ = {mu_pos:.2f} \le \phi M_n = {phi_mn:.2f} \text{{ kNm}}") # Double braces fixed
            st.markdown(f"Ratio: **{ratio:.2f}** $\longrightarrow$ {res_txt}")

        # Negative Moment Check
        with chk2:
            phi_mn_neg = abs(design_res['neg']['capacity'])
            mu_abs = abs(mu_neg)
            ratio = mu_abs / phi_mn_neg if phi_mn_neg > 0 else 0
            res_txt = r"\textcolor{green}{\textbf{PASS}}" if ratio <= 1.0 else r"\textcolor{red}{\textbf{FAIL}}"
            st.warning(f"**Negative Moment Check (-M)**")
            st.latex(rf"|M_u^-| = {mu_abs:.2f} \le \phi M_n = {phi_mn_neg:.2f} \text{{ kNm}}") # Double braces fixed
            st.markdown(f"Ratio: **{ratio:.2f}** $\longrightarrow$ {res_txt}")

        # === B. CALCULATION DETAILS TABLE (เพิ่มตารางสรุปเหล็ก) ===
        st.markdown("#### 📝 Reinforcement Summary")
        res_data = {
            "Position": ["Bottom (+Moment)", "Top (-Moment)", "Shear (Stirrups)"],
            "Design Load": [f"{mu_pos:.2f} kNm", f"{abs(mu_neg):.2f} kNm", f"{vu_span:.2f} kN"],
            "Required Steel ($cm^2$)": [f"{design_res['pos']['as_req']:.2f}", f"{design_res['neg']['as_req']:.2f}", "-"],
            "Provided Steel": [
                f"{design_res['pos']['n']} - DB{db} ({design_res['pos']['as_prov']:.2f} cm²)", 
                f"{design_res['neg']['n']} - DB{db} ({design_res['neg']['as_prov']:.2f} cm²)", 
                design_res['shear_stirrups']['text']
            ]
        }
        st.table(pd.DataFrame(res_data))

        # === C. VISUALIZATION ===
        col_plot, col_draw = st.columns([1, 1])
        with col_plot:
            st.write("**Capacity vs Demand Plot**")
            fig_cap = design_view.plot_capacity_vs_demand(span_data, design_res['pos']['capacity'], design_res['neg']['capacity'])
            st.plotly_chart(fig_cap, use_container_width=True)
            
        with col_draw:
            st.write("**Cross Section Detail**")
            fig_sec = section_plotter.plot_section(
                params['b'], params['h'], cover, db, 
                design_res['neg']['n'], design_res['pos']['n'], design_res['shear_stirrups'], fc, fy
            )
            st.pyplot(fig_sec, use_container_width=True)
            
        st.divider()
