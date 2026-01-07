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
    .calc-box { background-color: white; border: 1px solid #ddd; padding: 25px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .calc-header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; margin-bottom: 20px; font-weight: bold; font-size: 1.2em;}
    .pass { color: #27ae60; font-weight: bold; background-color: #eafaf1; padding: 2px 8px; border-radius: 4px; }
    .fail { color: #c0392b; font-weight: bold; background-color: #fdedec; padding: 2px 8px; border-radius: 4px; }
    .warning { color: #d35400; font-weight: bold; }
    .section-container { display: flex; justify-content: center; align-items: center; }
</style>
""", unsafe_allow_html=True)

# --- Init Session State ---
if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'avg_factor' not in st.session_state: st.session_state.avg_factor = 1.6 
if 'full_loads' not in st.session_state: st.session_state.full_loads = [] 
if 'sw_val' not in st.session_state: st.session_state.sw_val = 0.0

# --- 1. SIDEBAR INPUTS ---
with st.sidebar:
    st.title("⚙️ Project Inputs")
    design_std = st.radio("Standard", ["ACI 318", "EIT (Thailand)"])
    
    if "EIT" in design_std:
        factors = {'DL': 1.4, 'LL': 1.7, 'phi_m': 0.90, 'phi_v': 0.85, 'label': '1.70'}
        current_avg_factor = 1.7
    else:
        factors = {'DL': 1.2, 'LL': 1.6, 'phi_m': 0.90, 'phi_v': 0.75, 'label': '1.60'}
        current_avg_factor = 1.6
        
    st.info(f"Design Factor (Approx) ≈ {factors['label']}")
    
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()
    
    st.divider()
    run_btn = st.button("🚀 RUN ANALYSIS", type="primary")

# --- 2. MAIN AREA ---
st.header("🏗️ RC Beam Analysis & Design Report")

if run_btn:
    if not stable:
        st.error("🚨 Structure is unstable. Please check supports.")
    else:
        # --- A. PREPARE LOADS (Include SW) ---
        sw_mag = params['b'] * params['h'] * 24.0 * 1000 # N/m
        
        loads_combined = []
        if not loads_df.empty:
            loads_combined = loads_df.to_dict('records')
            
        for i in range(n_spans):
            loads_combined.append({
                "id": f"sw_{i}", "type": "U", "span_index": i, "x": 0.0,
                "mag": sw_mag, "dist": spans[i], "case": "DL (SW)"
            })

        # --- B. RUN ANALYSIS ---
        solver_service = solver.BeamSolver(spans, sup_df.to_dict('records'), loads_combined, params['E'], params['b'], params['h'], params['I'])
        res_service, reac_service, status = solver_service.solve()
        
        if "error" in status:
            st.error(status['error'])
            st.stop()

        # --- C. SAVE STATE ---
        st.session_state.res_service = res_service
        st.session_state.reac_service = reac_service
        st.session_state.loads_df = loads_df
        st.session_state.full_loads = loads_combined
        st.session_state.params = params
        st.session_state.factors = factors
        st.session_state.avg_factor = current_avg_factor
        st.session_state.sw_val = sw_mag / 1000.0 # kN/m
        st.session_state.analyzed = True

# --- DISPLAY RESULTS ---
if st.session_state.analyzed:
    res = st.session_state.res_service
    reac = st.session_state.reac_service
    p = st.session_state.params
    f = st.session_state.factors
    saf_factor = st.session_state.get('avg_factor', 1.6)
    full_loads = st.session_state.get('full_loads', [])
    sw_val = st.session_state.get('sw_val', 0.0)
    
    tab1, tab2 = st.tabs(["📊 1. Analysis (Service Load)", "📝 2. Design (Ultimate Load)"])
    
    # ================= TAB 1: SERVICE ANALYSIS =================
    with tab1:
        st.markdown(f"### 🔹 Serviceability State (DL + LL)")
        st.caption(f"**Note:** Analysis includes Self-Weight of Beam = **{sw_val:.2f} kN/m**")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Max Deflection", f"{res['deflection'].abs().max():.2f} mm")
        col2.metric("Max M (+)", f"{res['moment'].max()/1000:.2f} kNm")
        col3.metric("Max M (-)", f"{res['moment'].min()/1000:.2f} kNm")
        col4.metric("Max V", f"{res['shear'].abs().max()/1000:.2f} kN")
        
        fig = design_view.plot_analysis_results(res, spans, sup_df, full_loads)
        st.plotly_chart(fig, use_container_width=True)

    # ================= TAB 2: ULTIMATE DESIGN =================
    with tab2:
        st.markdown("### 🔹 Ultimate Limit State (Factored)")
        
        # Prepare Design Data
        cum_dist = [0] + list(np.cumsum(spans))
        design_data = []
        # Dummy structure for visualization
        design_res_for_plot = [] 
        
        for i in range(len(spans)):
            x0, x1 = cum_dist[i], cum_dist[i+1]
            span_df = res[(res['x'] >= x0) & (res['x'] <= x1)]
            
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
            design_res_for_plot.append({'pos': {'n': 3}, 'neg': {'n': 2}, 'db': 16})

        sel_span = st.selectbox("Select Span", design_data, format_func=lambda x: f"Span {x['span']}")
        
        # --- LAYOUT GRID ---
        col_design_left, col_design_right = st.columns([1, 1.6])
        
        with col_design_left:
            st.markdown(f"#### 🛠️ Reinforcement Control")
            with st.form("rebar_form"):
                cc1, cc2 = st.columns(2)
                n_top = cc1.number_input("Top Bars", 1, 10, 2)
                n_bot = cc2.number_input("Bot Bars", 1, 10, 3)
                cc3, cc4 = st.columns(2)
                db_main = cc3.selectbox("DB (mm)", [12, 16, 20, 25], index=1)
                s_stir = cc4.number_input("Stirrup @ (cm)", 5, 30, 15, 5)
                cover = st.number_input("Cover (mm)", 20, 50, 40)
                update_rb = st.form_submit_button("Update")
            
            if update_rb:
                design_res_for_plot[sel_span['span']-1] = {'pos': {'n': n_bot}, 'neg': {'n': n_top}, 'db': db_main}

            st.markdown("---")
            st.markdown("**Section View:**")
            
            # [FIX] Center the Section Plot using Columns
            c_fill_1, c_plot, c_fill_2 = st.columns([0.15, 0.7, 0.15])
            with c_plot:
                fig_sec = section_plotter.plot_section(p['b'], p['h'], cover, db_main, n_top, n_bot, f"RB6@{s_stir}", p['fc'], p['fy'])
                st.pyplot(fig_sec, use_container_width=False)

        with col_design_right:
            st.markdown('<div class="calc-box">', unsafe_allow_html=True)
            st.markdown(f'<div class="calc-header">📝 ENGINEER CALCULATION SHEET: Span {sel_span["span"]}</div>', unsafe_allow_html=True)
            
            # Variables
            b_mm = p['b'] * 1000
            d_mm = p['h'] * 1000 - cover - 6 - db_main/2
            As_prov = n_bot * (3.1416 * (db_main/2)**2)
            
            # 1. Loads
            st.markdown("**1. Design Forces**")
            st.latex(rf"M_u = {sel_span['Mu_pos']:.2f}\ kNm, \quad V_u = {sel_span['Vu']:.2f}\ kN")

            # 2. Flexure (Improved)
            st.markdown("---")
            st.markdown("**2. Flexural Check (+M)**")
            
            # (a) Min Steel Check
            st.markdown("*(a) Minimum Reinforcement Check ($A_{s,min}$)*")
            As_min1 = (0.25 * np.sqrt(p['fc']) / p['fy']) * b_mm * d_mm
            As_min2 = (1.4 / p['fy']) * b_mm * d_mm
            As_min = max(As_min1, As_min2)
            
            st.latex(rf"A_{{s,min}} = \max\left(\frac{{0.25\sqrt{{f_c'}}}}{{f_y}}, \frac{{1.4}}{{f_y}}\right) b_w d = {As_min:.0f}\ mm^2")
            if As_prov >= As_min:
                st.markdown(f'<span class="pass">✅ OK ($A_{{prov}} = {As_prov:.0f} > {As_min:.0f}$)</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="fail">❌ FAIL (Add Rebar to meet min steel)</span>', unsafe_allow_html=True)
            
            # (b) Capacity
            st.markdown("*(b) Moment Capacity ($\phi M_n$)*")
            a_depth = (As_prov * p['fy']) / (0.85 * p['fc'] * b_mm)
            Mn_kNm = As_prov * p['fy'] * (d_mm - a_depth/2) * 1e-6
            phi_Mn = f['phi_m'] * Mn_kNm
            
            st.latex(rf"a = {a_depth:.2f} mm, \quad \phi M_n = {f['phi_m']} \times {Mn_kNm:.2f} = \mathbf{{{phi_Mn:.2f}}}\ kNm")
            
            if phi_Mn >= sel_span['Mu_pos']:
                st.markdown(f'<span class="pass">✅ PASS (Ratio: {sel_span["Mu_pos"]/phi_Mn:.2f})</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="fail">❌ FAIL (Insufficient Capacity)</span>', unsafe_allow_html=True)

            # (c) Ductility
            st.markdown("*(c) Ductility Check (Strain)*")
            beta1 = 0.85 if p['fc'] <= 30 else max(0.65, 0.85 - 0.05*(p['fc']-30)/7)
            c = a_depth / beta1
            epsilon_t = 0.003 * (d_mm - c) / c
            
            st.latex(rf"c = {c:.2f} mm, \quad \epsilon_t = 0.003 \frac{{d-c}}{{c}} = \mathbf{{{epsilon_t:.4f}}}")
            if epsilon_t >= 0.005:
                st.markdown(f'<span class="pass">✅ OK (Tension Controlled, $\epsilon_t \geq 0.005$)</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="fail">❌ WARNING: Brittle / Transition (Reduce Steel or Increase Depth)</span>', unsafe_allow_html=True)

            # 3. Shear (Improved)
            st.markdown("---")
            st.markdown("**3. Shear Check**")
            Vc = 0.17 * np.sqrt(p['fc']) * b_mm * d_mm / 1000.0
            Av = 2 * (3.1416 * 3**2) 
            Vs_req = (sel_span['Vu']/f['phi_v']) - Vc
            
            # Check Max Spacing
            s_max = d_mm / 2
            st.latex(rf"\phi V_c = {f['phi_v']*Vc:.2f}\ kN")
            
            if Vs_req > 0:
                s_calc = (Av * p['fy'] * d_mm) / (Vs_req * 1000) * 10 # mm to cm roughly
                st.markdown(f"*Strength requires spacing $\leq {s_calc/10:.1f}$ cm*")
            
            Vs_prov = (Av * p['fy'] * d_mm) / (s_stir*10) / 1000.0
            phi_Vn = f['phi_v'] * (Vc + Vs_prov)
            
            st.latex(rf"\phi V_n = \mathbf{{{phi_Vn:.2f}}}\ kN \quad (vs \ V_u = {sel_span['Vu']:.2f})")
            
            shear_status = "✅ PASS" if phi_Vn >= sel_span['Vu'] else "❌ FAIL"
            spacing_status = "✅ Spacing OK" if (s_stir*10) <= s_max else f"❌ Spacing > d/2 ({s_max/10:.1f} cm)"
            
            st.markdown(f"{shear_status} | {spacing_status}")

            st.markdown('</div>', unsafe_allow_html=True)
        
        # --- BOTTOM AREA: LONGITUDINAL PROFILE ---
        st.markdown("---")
        st.markdown("### 🏗️ Longitudinal Reinforcement Profile")
        # Update dummy data with current user selection for visualization of current span
        # Note: In a full app, we would store n_top/n_bot for EACH span in session_state. 
        # Here we just visualize the current one across the board for demo or specific logic.
        design_res_for_plot[sel_span['span']-1] = {'pos': {'n': n_bot}, 'neg': {'n': n_top}, 'db': db_main}
        
        fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_res_for_plot, p['h'], cover)
        st.pyplot(fig_long, use_container_width=True)
