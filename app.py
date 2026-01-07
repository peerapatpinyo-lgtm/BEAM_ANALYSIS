import streamlit as st
import pandas as pd
import numpy as np

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

# --- CLEAN & PROFESSIONAL CSS ---
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #f8f9fa; }
    
    /* Headers */
    h1, h2, h3 { color: #2c3e50; font-family: 'Segoe UI', sans-serif; }
    
    /* Cards/Containers */
    .css-1r6slb0, .stContainer {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { font-size: 1.2rem !important; color: #333; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem !important; color: #666; }
    
    /* Report Section Styles */
    .rpt-header { 
        font-size: 18px; font-weight: bold; color: #1a202c; 
        border-bottom: 2px solid #3182ce; padding-bottom: 8px; margin-bottom: 16px; 
    }
    .rpt-sub { font-weight: 600; color: #4a5568; margin-top: 12px; }
    .status-ok { color: #276749; background: #c6f6d5; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }
    .status-fail { color: #9b2c2c; background: #fed7d7; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)

# --- Init Session State ---
if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'avg_factor' not in st.session_state: st.session_state.avg_factor = 1.6 
if 'full_loads' not in st.session_state: st.session_state.full_loads = [] 
if 'sw_val' not in st.session_state: st.session_state.sw_val = 0.0

# --- SIDEBAR ---
with st.sidebar:
    st.caption("🏗️ STRUCTURAL SETTINGS")
    design_std = st.radio("Code", ["ACI 318", "EIT 1008"])
    
    if "EIT" in design_std:
        factors = {'DL': 1.4, 'LL': 1.7, 'phi_m': 0.90, 'phi_v': 0.85, 'name': 'EIT 1008'}
        current_avg_factor = 1.7 
    else:
        factors = {'DL': 1.2, 'LL': 1.6, 'phi_m': 0.90, 'phi_v': 0.75, 'name': 'ACI 318'}
        current_avg_factor = 1.6
        
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()
    st.divider()
    run_btn = st.button("🚀 CALCULATE", type="primary", use_container_width=True)

# --- MAIN ---
st.title(f"📄 Beam Design Report: {factors['name']}")

if run_btn:
    if not stable:
        st.error("🚨 Unstable Structure")
    else:
        # Load Prep
        sw_mag = params['b'] * params['h'] * 24.0 * 1000 
        loads_combined = []
        if not loads_df.empty: loads_combined = loads_df.to_dict('records')
        for i in range(n_spans):
            loads_combined.append({"id": f"sw_{i}", "type": "U", "span_index": i, "x": 0.0, "mag": sw_mag, "dist": spans[i]})

        # Solve
        solver_service = solver.BeamSolver(spans, sup_df.to_dict('records'), loads_combined, params['E'], params['b'], params['h'], params['I'])
        res_service, reac_raw, status = solver_service.solve()
        
        if "error" in status: st.error(status['error']); st.stop()

        # Normalize Reac
        final_reac_list = []
        if isinstance(reac_raw, pd.DataFrame):
            if 'fy' in reac_raw.columns: final_reac_list = reac_raw.to_dict('records')
            else: final_reac_list = [{'node_id': i, 'fy': r.iloc[0]} for i, r in reac_raw.iterrows()]
        elif isinstance(reac_raw, dict):
            final_reac_list = [{'node_id': int(k), 'fy': v} for k, v in reac_raw.items()]
        elif isinstance(reac_raw, list):
             # Handle list of floats or dicts
             if len(reac_raw) > 0 and isinstance(reac_raw[0], (float, int)):
                 final_reac_list = [{'node_id': i, 'fy': v} for i, v in enumerate(reac_raw)]
             else:
                 final_reac_list = reac_raw

        clean_reac = [{'node_id': r.get('node_id', r.get('id', i)), 'fy': r.get('fy', 0)} for i, r in enumerate(final_reac_list)]

        # Save State
        st.session_state.res_service = res_service
        st.session_state.reac_service = clean_reac
        st.session_state.loads_df = loads_df
        st.session_state.full_loads = loads_combined
        st.session_state.params = params
        st.session_state.factors = factors
        st.session_state.avg_factor = current_avg_factor
        st.session_state.sw_val = sw_mag / 1000.0
        st.session_state.analyzed = True

if st.session_state.analyzed:
    res = st.session_state.res_service
    reac = st.session_state.reac_service
    p = st.session_state.params
    f = st.session_state.factors
    saf_factor = st.session_state.get('avg_factor', 1.6)
    full_loads = st.session_state.get('full_loads', [])
    sw_val = st.session_state.get('sw_val', 0.0)
    
    # --- LAYOUT: 2 TABS (Analysis vs Design) ---
    t1, t2 = st.tabs(["📊 ANALYSIS RESULTS", "📝 DESIGN CALCULATION SHEET"])
    
    with t1:
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max Deflection", f"{res['deflection'].abs().max():.2f} mm")
        c2.metric("Max Moment", f"{res['moment'].abs().max()/1000:.2f} kNm")
        c3.metric("Max Shear", f"{res['shear'].abs().max()/1000:.2f} kN")
        c4.metric("Self-Weight", f"{sw_val:.2f} kN/m")
        
        # Plot
        fig = design_view.plot_analysis_results(res, spans, sup_df, full_loads, reac)
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        # 1. DESIGN CONTROLS (Top Bar)
        cum_dist = [0] + list(np.cumsum(spans))
        design_data = []
        
        for i in range(len(spans)):
            x0, x1 = cum_dist[i], cum_dist[i+1]
            span_df = res[(res['x'] >= x0) & (res['x'] <= x1)]
            M_max = max(span_df['moment'].abs().max(), 0) / 1000
            V_max = span_df['shear'].abs().max() / 1000
            
            design_data.append({
                "span": i+1,
                "Mu": M_max * saf_factor,
                "Vu": V_max * saf_factor,
                "def_act": span_df['deflection'].abs().max(),
                "L": spans[i]
            })

        with st.container():
            col_sel, col_in = st.columns([1, 4])
            with col_sel:
                sel_span = st.selectbox("📌 Select Span", design_data, format_func=lambda x: f"Span {x['span']} (L={x['L']}m)")
            with col_in:
                with st.form("design_form"):
                    c_in1, c_in2, c_in3, c_in4, c_in5, c_btn = st.columns([1,1,1,1.2,1,1])
                    n_top = c_in1.number_input("Top (Bars)", 1, 10, 2)
                    n_bot = c_in2.number_input("Bot (Bars)", 1, 10, 3)
                    db_main = c_in3.selectbox("DB (mm)", [12, 16, 20, 25, 28], index=1)
                    s_stir = c_in4.number_input("Stirrup @ (cm)", 5, 40, 15, 5)
                    cover = c_in5.number_input("Cov (mm)", 20, 50, 40)
                    c_btn.write("")
                    c_btn.form_submit_button("Update")

        st.markdown("---")

        # 2. CALCULATION SHEET LAYOUT
        # Use columns to mimic paper layout
        
        # --- LEFT: TEXT CALCULATIONS ---
        col_calc, col_draw = st.columns([1.2, 1])
        
        with col_calc:
            st.markdown(f"<div class='rpt-header'>SPAN {sel_span['span']} DESIGN CHECKS</div>", unsafe_allow_html=True)
            
            # A. Flexure
            st.markdown("<div class='rpt-sub'>1. Flexural Capacity</div>", unsafe_allow_html=True)
            b, h = p['b']*1000, p['h']*1000
            d = h - cover - 6 - db_main/2
            As = n_bot * 3.1416 * (db_main/2)**2
            
            # Calc
            a = (As * p['fy']) / (0.85 * p['fc'] * b)
            Mn = As * p['fy'] * (d - a/2) * 1e-6
            phi_Mn = f['phi_m'] * Mn
            
            c_res1, c_res2 = st.columns(2)
            c_res1.latex(rf"M_u = {sel_span['Mu']:.2f}\ kNm")
            c_res2.latex(rf"\phi M_n = {phi_Mn:.2f}\ kNm")
            
            if phi_Mn >= sel_span['Mu']: 
                st.markdown(f"<span class='status-ok'>✅ SAFE (Ratio: {sel_span['Mu']/phi_Mn:.2f})</span>", unsafe_allow_html=True)
            else: 
                st.markdown(f"<span class='status-fail'>❌ UNSAFE (Ratio: {sel_span['Mu']/phi_Mn:.2f})</span>", unsafe_allow_html=True)

            # B. Shear
            st.markdown("<div class='rpt-sub'>2. Shear Capacity</div>", unsafe_allow_html=True)
            Vc = 0.17 * np.sqrt(p['fc']) * b * d / 1000
            phi_Vc = f['phi_v'] * Vc
            Vs = (2 * 28.27 * p['fy'] * d) / (s_stir*10) / 1000
            phi_Vs = f['phi_v'] * Vs
            phi_Vn = phi_Vc + phi_Vs
            
            c_sh1, c_sh2 = st.columns(2)
            c_sh1.latex(rf"V_u = {sel_span['Vu']:.2f}\ kN")
            c_sh2.latex(rf"\phi V_n = {phi_Vn:.2f}\ kN")
            
            st.caption(f"Concrete: {phi_Vc:.2f} kN | Stirrups: {phi_Vs:.2f} kN (RB6@{s_stir}cm)")
            
            if phi_Vn >= sel_span['Vu']: st.markdown(f"<span class='status-ok'>✅ SAFE</span>", unsafe_allow_html=True)
            else: st.markdown(f"<span class='status-fail'>❌ UNSAFE</span>", unsafe_allow_html=True)

            # C. Deflection
            st.markdown("<div class='rpt-sub'>3. Serviceability</div>", unsafe_allow_html=True)
            L_mm = sel_span['L']*1000
            d_all = L_mm/240
            d_act = sel_span['def_act']
            st.write(f"Allowable: {d_all:.1f} mm | Actual: **{d_act:.2f} mm**")
            if d_act <= d_all: st.markdown(f"<span class='status-ok'>✅ PASS</span>", unsafe_allow_html=True)
            else: st.markdown(f"<span class='status-fail'>❌ FAIL</span>", unsafe_allow_html=True)

        # --- RIGHT: SECTION DRAWING ---
        with col_draw:
            st.markdown(f"<div class='rpt-header'>SECTION DETAILS</div>", unsafe_allow_html=True)
            fig_sec = section_plotter.plot_section(p['b'], p['h'], cover, db_main, n_top, n_bot, f"RB6@{s_stir}cm", p['fc'], p['fy'])
            st.pyplot(fig_sec, use_container_width=False)

        # --- BOTTOM: LONGITUDINAL PROFILE (FULL WIDTH) ---
        st.markdown("---")
        st.markdown(f"<div class='rpt-header'>LONGITUDINAL PROFILE (Span {sel_span['span']})</div>", unsafe_allow_html=True)
        st.info(f"Showing reinforcement for Span {sel_span['span']} based on input: {n_top}-DB{db_main} Top, {n_bot}-DB{db_main} Bot, Stirrup RB6@{s_stir}cm")
        
        # [NEW] Call the detailed plotter
        fig_long = section_plotter.plot_longitudinal_detailed(
            sel_span['L'], p['h'], cover, 
            n_top, n_bot, db_main, s_stir, sel_span['span']
        )
        st.pyplot(fig_long, use_container_width=True)
