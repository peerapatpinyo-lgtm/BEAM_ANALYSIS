import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import input_handler
    import solver
    import rc_design
    import design_view
    import section_plotter
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

st.set_page_config(page_title="Pro Beam Design", layout="wide")

st.markdown("""
<style>
    .calc-box { background-color: white; border: 1px solid #ddd; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .calc-header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; margin-bottom: 15px; font-weight: bold; }
    .pass { color: green; font-weight: bold; }
    .fail { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ Project Inputs")
    design_std = st.radio("Standard", ["ACI 318", "EIT (Thailand)"])
    
    # [FIX] กำหนด Factor และ Label ให้ชัดเจน
    if "EIT" in design_std:
        # ใช้ 1.7 เป็นตัวแทน Conservative Factor สำหรับโชว์รายการคำนวณเบื้องต้น
        factors = {'DL': 1.4, 'LL': 1.7, 'phi_m': 0.90, 'phi_v': 0.85, 'label': '1.70 (Conservative LL)'}
        avg_factor = 1.7
    else:
        factors = {'DL': 1.2, 'LL': 1.6, 'phi_m': 0.90, 'phi_v': 0.75, 'label': '1.60 (ACI LL)'}
        avg_factor = 1.6
        
    st.info(f"Using Safety Factor ≈ {factors['label']} for envelope scaling.")
    
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()
    st.divider()
    run_btn = st.button("🚀 RUN ANALYSIS", type="primary")

st.header("🏗️ RC Beam Analysis & Design Report")

if run_btn:
    if not stable:
        st.error("🚨 Structure is unstable.")
    else:
        # Prepare Loads
        sw_mag = params['b'] * params['h'] * 24.0 * 1000
        loads_service = []
        if not loads_df.empty: loads_service = loads_df.to_dict('records')
        for i in range(n_spans):
            loads_service.append({"id": f"sw_{i}", "type": "U", "span_index": i, "x": 0.0, "mag": sw_mag, "dist": spans[i], "case": "DL"})

        # Run Solver (Service)
        solver_service = solver.BeamSolver(spans, sup_df.to_dict('records'), loads_service, params['E'], params['b'], params['h'], params['I'])
        res_service, reac_service, status = solver_service.solve()
        
        if "error" in status: st.error(status['error']); st.stop()

        st.session_state.res_service = res_service
        st.session_state.reac_service = reac_service
        st.session_state.loads_df = loads_df
        st.session_state.params = params
        st.session_state.factors = factors
        st.session_state.avg_factor = avg_factor # เก็บค่า Factor ไว้ใช้
        st.session_state.analyzed = True

if st.session_state.get('analyzed'):
    res = st.session_state.res_service
    reac = st.session_state.reac_service
    p = st.session_state.params
    f = st.session_state.factors
    saf_factor = st.session_state.avg_factor
    
    tab1, tab2 = st.tabs(["📊 1. Analysis (Service Load)", "📝 2. Design (Ultimate Load)"])
    
    with tab1:
        st.markdown("### 🔹 Serviceability State (Unfactored)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Max Deflection", f"{res['deflection'].abs().max():.2f} mm")
        col2.metric("Max M (+)", f"{res['moment'].max()/1000:.2f} kNm")
        col3.metric("Max M (-)", f"{res['moment'].min()/1000:.2f} kNm")
        col4.metric("Max V", f"{res['shear'].abs().max()/1000:.2f} kN")
        
        fig = design_view.plot_analysis_results(res, spans, sup_df, st.session_state.loads_df.to_dict('records') if not st.session_state.loads_df.empty else [])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### 🔹 Ultimate Limit State (Factored)")
        
        # Pre-calculate Design Values
        cum_dist = [0] + list(np.cumsum(spans))
        design_data = []
        for i in range(len(spans)):
            x0, x1 = cum_dist[i], cum_dist[i+1]
            span_df = res[(res['x'] >= x0) & (res['x'] <= x1)]
            
            # Service Values
            M_pos_serv = max(0, span_df['moment'].max()) / 1000
            M_neg_serv = abs(min(0, span_df['moment'].min())) / 1000
            V_serv = span_df['shear'].abs().max() / 1000
            
            design_data.append({
                "span": i+1,
                "M_serv_pos": M_pos_serv, "M_serv_neg": M_neg_serv, "V_serv": V_serv,
                "Mu_pos": M_pos_serv * saf_factor,
                "Mu_neg": M_neg_serv * saf_factor,
                "Vu": V_serv * saf_factor,
                "def_act": span_df['deflection'].abs().max(),
                "L": spans[i]
            })

        sel_span = st.selectbox("Select Span", design_data, format_func=lambda x: f"Span {x['span']}")
        
        c_left, c_right = st.columns([1, 1.3])
        
        with c_left:
            st.markdown(f"#### 🛠️ Reinforcement: Span {sel_span['span']}")
            with st.form("rebar_form"):
                cc1, cc2 = st.columns(2)
                n_top = cc1.number_input("Top Bars", 1, 10, 2)
                n_bot = cc2.number_input("Bot Bars", 1, 10, 3)
                cc3, cc4 = st.columns(2)
                db_main = cc3.selectbox("DB (mm)", [12, 16, 20, 25], index=1)
                s_stir = cc4.number_input("Stirrup @ (cm)", 5, 30, 15, 5)
                cover = st.number_input("Cover (mm)", 20, 50, 40)
                st.form_submit_button("Update")
            
            fig_sec = section_plotter.plot_section(p['b'], p['h'], cover, db_main, n_top, n_bot, f"RB6@{s_stir}", p['fc'], p['fy'])
            st.pyplot(fig_sec)

        with c_right:
            st.markdown('<div class="calc-box">', unsafe_allow_html=True)
            st.markdown(f'<div class="calc-header">📝 CALCULATION SHEET: Span {sel_span["span"]}</div>', unsafe_allow_html=True)
            
            # 1. Load Combination (โชว์ที่มาตัวเลข)
            st.markdown("**1. Design Forces (Load Factors)**")
            st.caption(f"Using Conservative Safety Factor = {saf_factor} (based on {design_std})")
            
            # [FIX] แสดงสมการการคูณชัดเจน
            st.latex(rf"M_u^+ = M_{{serv}} \times {saf_factor} = {sel_span['M_serv_pos']:.2f} \times {saf_factor} = \mathbf{{{sel_span['Mu_pos']:.2f}}}\ kNm")
            st.latex(rf"V_u = V_{{serv}} \times {saf_factor} = {sel_span['V_serv']:.2f} \times {saf_factor} = \mathbf{{{sel_span['Vu']:.2f}}}\ kN")

            # 2. Flexure
            st.markdown("---")
            st.markdown("**2. Flexural Check (+M)**")
            As_prov = n_bot * (3.1416 * (db_main/2)**2)
            d = p['h']*1000 - cover - 6 - db_main/2
            a = (As_prov * p['fy']) / (0.85 * p['fc'] * p['b']*1000)
            Mn = As_prov * p['fy'] * (d - a/2) * 1e-6 
            phi_Mn = f['phi_m'] * Mn
            
            st.latex(rf"A_s = {n_bot}\text{{-}}DB{db_main} = {As_prov:.0f}\ mm^2")
            st.latex(rf"\phi M_n = {f['phi_m']} \times {Mn:.2f} = \mathbf{{{phi_Mn:.2f}}}\ kNm")
            
            if phi_Mn >= sel_span['Mu_pos']:
                st.markdown(f'<span class="pass">✅ PASS (Ratio: {sel_span["Mu_pos"]/phi_Mn:.2f})</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="fail">❌ FAIL</span>', unsafe_allow_html=True)

            # 3. Shear
            st.markdown("---")
            st.markdown("**3. Shear Check**")
            Vc = 0.17 * np.sqrt(p['fc']) * p['b']*1000 * d / 1000 
            Av = 2 * (3.1416 * 3**2)
            Vs = (Av * p['fy'] * d) / (s_stir*10) / 1000 
            phi_Vn = f['phi_v'] * (Vc + Vs)
            
            st.latex(rf"\phi V_c = {f['phi_v']*Vc:.2f}\ kN, \quad \phi V_s = {f['phi_v']*Vs:.2f}\ kN")
            st.latex(rf"\phi V_n = {f['phi_v']*Vc:.2f} + {f['phi_v']*Vs:.2f} = \mathbf{{{phi_Vn:.2f}}}\ kN")
             
            if phi_Vn >= sel_span['Vu']:
                 st.markdown(f'<span class="pass">✅ PASS</span>', unsafe_allow_html=True)
            else:
                 st.markdown(f'<span class="fail">❌ FAIL</span>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
