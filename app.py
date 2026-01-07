import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
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

# --- CSS Styling for Calculation Sheet ---
st.markdown("""
<style>
    .calc-box { background-color: white; border: 1px solid #ddd; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .calc-header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; margin-bottom: 15px; font-weight: bold; }
    .calc-sub { color: #7f8c8d; font-size: 0.9em; margin-bottom: 10px; }
    .pass { color: green; font-weight: bold; }
    .fail { color: red; font-weight: bold; }
    .warning { color: orange; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 1. SIDEBAR INPUTS (All in one place) ---
with st.sidebar:
    st.title("⚙️ Project Inputs")
    
    # Design Code Selection
    design_std = st.radio("Standard", ["ACI 318", "EIT (Thailand)"])
    if "EIT" in design_std:
        factors = {'DL': 1.4, 'LL': 1.7, 'phi_m': 0.90, 'phi_v': 0.85}
    else:
        factors = {'DL': 1.2, 'LL': 1.6, 'phi_m': 0.90, 'phi_v': 0.75}
        
    st.info(f"Factors: {factors['DL']}DL + {factors['LL']}LL | φ_m={factors['phi_m']}, φ_v={factors['phi_v']}")
    
    # Render Inputs from Handler
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()
    
    st.divider()
    run_btn = st.button("🚀 RUN ANALYSIS", type="primary")

# --- 2. MAIN AREA ---
st.header("🏗️ RC Beam Analysis & Design Report")

if run_btn:
    if not stable:
        st.error("🚨 Structure is unstable. Please check supports.")
    else:
        # --- A. PREPARE LOADS ---
        # 1. Service Loads (for Deflection & Tab 1)
        sw_mag = params['b'] * params['h'] * 24.0 * 1000 # N/m
        
        loads_service = []
        if not loads_df.empty:
            loads_service = loads_df.to_dict('records')
            
        # Add Self Weight to Service
        for i in range(n_spans):
            loads_service.append({
                "id": f"sw_{i}", "type": "U", "span_index": i, "x": 0.0,
                "mag": sw_mag, "dist": spans[i], "case": "DL"
            })

        # --- B. RUN ANALYSIS (Service) ---
        solver_service = solver.BeamSolver(spans, sup_df.to_dict('records'), loads_service, params['E'], params['b'], params['h'], params['I'])
        res_service, reac_service, status = solver_service.solve()
        
        if "error" in status:
            st.error(status['error'])
            st.stop()

        # --- C. PREPARE RESULTS ---
        st.session_state.res_service = res_service
        st.session_state.reac_service = reac_service
        st.session_state.loads_df = loads_df
        st.session_state.params = params
        st.session_state.factors = factors
        st.session_state.analyzed = True

# --- DISPLAY RESULTS ---
if st.session_state.get('analyzed'):
    res = st.session_state.res_service
    reac = st.session_state.reac_service
    p = st.session_state.params
    f = st.session_state.factors
    
    tab1, tab2 = st.tabs(["📊 1. Analysis (Service Load)", "📝 2. Design (Ultimate Load)"])
    
    # ================= TAB 1: SERVICE ANALYSIS =================
    with tab1:
        st.markdown("### 🔹 Serviceability State (Unfactored Loads)")
        st.caption("Results here include Dead Load + Live Load (1.0 DL + 1.0 LL). Used for Deflection Check.")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        max_defl = res['deflection'].abs().max()
        col1.metric("Max Deflection", f"{max_defl:.2f} mm")
        col2.metric("Max Moment (+)", f"{res['moment'].max()/1000:.2f} kNm")
        col3.metric("Max Moment (-)", f"{res['moment'].min()/1000:.2f} kNm")
        col4.metric("Max Shear", f"{res['shear'].abs().max()/1000:.2f} kN")
        
        # Plot
        fig = design_view.plot_analysis_results(res, spans, sup_df, st.session_state.loads_df.to_dict('records') if not st.session_state.loads_df.empty else [])
        st.plotly_chart(fig, use_container_width=True)
        
        # Reactions Table
        st.markdown("#### Service Reactions")
        r_data = [{"Node": k, "R (kN)": v/1000} for k,v in reac.items()]
        st.dataframe(pd.DataFrame(r_data).set_index("Node").T)

    # ================= TAB 2: ULTIMATE DESIGN =================
    with tab2:
        st.markdown("### 🔹 Ultimate Limit State (Factored Loads)")
        st.caption(f"Using Factors: {f['DL']}DL + {f['LL']}LL")
        
        # Design Logic per Span
        cum_dist = [0] + list(np.cumsum(spans))
        
        # Generate Design Results first
        design_data = []
        for i in range(len(spans)):
            # Slice results for this span
            x0, x1 = cum_dist[i], cum_dist[i+1]
            span_df = res[(res['x'] >= x0) & (res['x'] <= x1)]
            
            # Get Max Forces (Service)
            M_pos_serv = max(0, span_df['moment'].max()) / 1000
            M_neg_serv = abs(min(0, span_df['moment'].min())) / 1000
            V_serv = span_df['shear'].abs().max() / 1000
            
            # Apply Factors (Approximation: Scaling max envelope)
            # Note: Rigorously we should run solver again with factored loads, 
            # but scaling max service envelope by avg factor is a standard fast approximation in this context.
            # Using conservative max factor:
            avg_factor = f['LL'] # Conservative approach
            
            Mu_pos = M_pos_serv * avg_factor
            Mu_neg = M_neg_serv * avg_factor
            Vu = V_serv * avg_factor
            
            design_data.append({
                "span": i+1,
                "Mu_pos": Mu_pos, "Mu_neg": Mu_neg, "Vu": Vu,
                "def_act": span_df['deflection'].abs().max(),
                "L": spans[i]
            })

        # --- Interactive Design ---
        sel_span = st.selectbox("Select Span to Design", design_data, format_func=lambda x: f"Span {x['span']}")
        
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
                st.form_submit_button("Update Calculation")
            
            # Draw Section
            fig_sec = section_plotter.plot_section(p['b'], p['h'], cover, db_main, n_top, n_bot, f"RB6@{s_stir}", p['fc'], p['fy'])
            st.pyplot(fig_sec)

        with c_right:
            # --- CALCULATION SHEET ---
            st.markdown('<div class="calc-box">', unsafe_allow_html=True)
            st.markdown(f'<div class="calc-header">📝 CALCULATION SHEET: Span {sel_span["span"]}</div>', unsafe_allow_html=True)
            
            # 1. Properties
            st.latex(rf"b = {p['b']*100:.0f}\ cm, \quad h = {p['h']*100:.0f}\ cm, \quad d \approx {p['h']*1000 - cover - 6 - db_main/2:.0f}\ mm")
            st.latex(rf"f_c' = {p['fc']}\ MPa, \quad f_y = {p['fy']}\ MPa")
            
            # 2. Factored Loads
            st.markdown("---")
            st.markdown("**1. Design Forces (Factored)**")
            st.latex(rf"M_u^+ = \mathbf{{{sel_span['Mu_pos']:.2f}}}\ kNm")
            st.latex(rf"V_u = \mathbf{{{sel_span['Vu']:.2f}}}\ kN")

            # 3. Flexure Check
            st.markdown("---")
            st.markdown("**2. Flexural Capacity Check (+M)**")
            
            # Calc
            As_prov = n_bot * (3.1416 * (db_main/2)**2)
            d = p['h']*1000 - cover - 6 - db_main/2
            a = (As_prov * p['fy']) / (0.85 * p['fc'] * p['b']*1000)
            Mn = As_prov * p['fy'] * (d - a/2) * 1e-6 # kNm
            phi_Mn = f['phi_m'] * Mn
            
            st.latex(rf"A_{{s,prov}} = {n_bot} \times DB{db_main} = {As_prov:.0f}\ mm^2")
            st.latex(rf"a = \frac{{A_s f_y}}{{0.85 f_c' b}} = {a:.2f}\ mm")
            st.latex(rf"\phi M_n = {f['phi_m']} \times {Mn:.2f} = \mathbf{{{phi_Mn:.2f}}}\ kNm")
            
            if phi_Mn >= sel_span['Mu_pos']:
                st.markdown(f'<span class="pass">✅ PASS (Ratio: {sel_span["Mu_pos"]/phi_Mn:.2f})</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="fail">❌ FAIL (Insufficient Moment Capacity)</span>', unsafe_allow_html=True)
                
            # 4. Shear Check
            st.markdown("---")
            st.markdown("**3. Shear Capacity Check**")
            
            Vc = 0.17 * np.sqrt(p['fc']) * p['b']*1000 * d / 1000 # kN
            Av = 2 * (3.1416 * 3**2) # 2 legs of RB6
            Vs = (Av * p['fy'] * d) / (s_stir*10) / 1000 # kN
            phi_Vn = f['phi_v'] * (Vc + Vs)
            
            st.latex(rf"\phi V_c = {f['phi_v']}({Vc:.2f}) = {f['phi_v']*Vc:.2f}\ kN")
            st.latex(rf"\phi V_s = {f['phi_v']}({Vs:.2f}) = {f['phi_v']*Vs:.2f}\ kN")
            st.latex(rf"\phi V_n = \mathbf{{{phi_Vn:.2f}}}\ kN")
            
            if phi_Vn >= sel_span['Vu']:
                 st.markdown(f'<span class="pass">✅ PASS (Ratio: {sel_span["Vu"]/phi_Vn:.2f})</span>', unsafe_allow_html=True)
            else:
                 st.markdown(f'<span class="fail">❌ FAIL (Need tighter stirrups)</span>', unsafe_allow_html=True)

            # 5. Deflection
            st.markdown("---")
            st.markdown("**4. Serviceability (Deflection)**")
            L_allow = (sel_span['L']*1000) / 360
            st.latex(rf"\Delta_{{actual}} = {sel_span['def_act']:.2f}\ mm")
            st.latex(rf"\Delta_{{limit}} = L/360 = {L_allow:.2f}\ mm")
            
            if sel_span['def_act'] <= L_allow:
                 st.markdown(f'<span class="pass">✅ PASS</span>', unsafe_allow_html=True)
            else:
                 st.markdown(f'<span class="fail">❌ FAIL (Too flexible)</span>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
