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

# --- CUSTOM CSS FOR REPORT LOOK ---
st.markdown("""
<style>
    .report-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #ddd;
    }
    .report-header {
        color: #1f2937;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 10px;
        margin-bottom: 20px;
        font-size: 1.5em;
        font-weight: 700;
        text-align: center;
    }
    .section-title {
        color: #374151;
        font-size: 1.1em;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 5px;
        border-left: 4px solid #3b82f6;
        padding-left: 10px;
        background-color: #eff6ff;
    }
    .pass-tag { color: #059669; font-weight: bold; background-color: #d1fae5; padding: 2px 8px; border-radius: 4px; font-size: 0.9em; }
    .fail-tag { color: #dc2626; font-weight: bold; background-color: #fee2e2; padding: 2px 8px; border-radius: 4px; font-size: 0.9em; }
    
    /* Make metrics look cleaner */
    div[data-testid="stMetric"] {
        background-color: #f9fafb;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e5e7eb;
    }
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
    design_std = st.radio("Standard Code", ["ACI 318 (USA)", "EIT 1008 (Thailand)"])
    
    if "EIT" in design_std:
        factors = {'DL': 1.4, 'LL': 1.7, 'phi_m': 0.90, 'phi_v': 0.85, 'name': 'EIT Standard (วสท.)'}
        current_avg_factor = 1.7 
    else:
        factors = {'DL': 1.2, 'LL': 1.6, 'phi_m': 0.90, 'phi_v': 0.75, 'name': 'ACI 318'}
        current_avg_factor = 1.6
        
    st.info(f"Using: **{factors['name']}**\n\nFactors: {factors['DL']}DL + {factors['LL']}LL")
    
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()
    
    st.divider()
    run_btn = st.button("🚀 RUN ANALYSIS", type="primary")

# --- 2. MAIN AREA ---
st.header(f"🏗️ RC Beam Analysis & Design Report")

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
        res_service, reac_raw, status = solver_service.solve()
        
        if "error" in status:
            st.error(status['error'])
            st.stop()

        # --- C. NORMALIZE REACTION DATA (ROBUST FIX) ---
        final_reac_list = []
        if isinstance(reac_raw, pd.DataFrame):
            if 'fy' in reac_raw.columns:
                final_reac_list = reac_raw.to_dict('records')
            else:
                 for idx, row in reac_raw.iterrows():
                     val = row.iloc[0] if len(row) > 0 else 0
                     final_reac_list.append({'node_id': idx, 'fy': val})
        elif isinstance(reac_raw, dict):
            for nid, val in reac_raw.items():
                final_reac_list.append({'node_id': int(nid), 'fy': val})
        elif isinstance(reac_raw, list):
            if len(reac_raw) > 0:
                first_item = reac_raw[0]
                if isinstance(first_item, dict):
                    final_reac_list = reac_raw
                elif isinstance(first_item, (int, float, np.number)):
                    for idx, val in enumerate(reac_raw):
                        final_reac_list.append({'node_id': idx, 'fy': val})
            else:
                final_reac_list = []

        clean_reac = []
        for i, r in enumerate(final_reac_list):
            if isinstance(r, dict):
                node = r.get('node_id', r.get('id', i))
                fy = r.get('fy', 0.0)
                clean_reac.append({'node_id': node, 'fy': fy})
            else:
                clean_reac.append({'node_id': i, 'fy': 0.0})

        # --- D. SAVE STATE ---
        st.session_state.res_service = res_service
        st.session_state.reac_service = clean_reac
        st.session_state.loads_df = loads_df
        st.session_state.full_loads = loads_combined
        st.session_state.params = params
        st.session_state.factors = factors
        st.session_state.avg_factor = current_avg_factor
        st.session_state.sw_val = sw_mag / 1000.0
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
    
    tab1, tab2 = st.tabs(["📊 1. Analysis Diagrams", "📝 2. Detailed Design Report"])
    
    # ================= TAB 1: DIAGRAMS =================
    with tab1:
        st.markdown(f"### 🔹 Serviceability Diagrams")
        c1, c2, c3, c4 = st.columns(4)
        max_def = res['deflection'].abs().max()
        max_M = res['moment'].abs().max()/1000
        max_V = res['shear'].abs().max()/1000
        c1.metric("Max Deflection", f"{max_def:.2f} mm")
        c2.metric("Max Moment", f"{max_M:.2f} kNm")
        c3.metric("Max Shear", f"{max_V:.2f} kN")
        c4.metric("Self-Weight", f"{sw_val:.2f} kN/m")
        
        fig = design_view.plot_analysis_results(res, spans, sup_df, full_loads, reac)
        st.plotly_chart(fig, use_container_width=True)

    # ================= TAB 2: DETAILED CALCULATION =================
    with tab2:
        # --- Prepare Data ---
        cum_dist = [0] + list(np.cumsum(spans))
        design_data = []
        design_res_for_plot = [] 
        
        for i in range(len(spans)):
            x0, x1 = cum_dist[i], cum_dist[i+1]
            span_df = res[(res['x'] >= x0) & (res['x'] <= x1)]
            
            M_pos_serv = max(0, span_df['moment'].max()) / 1000
            M_neg_serv = abs(min(0, span_df['moment'].min())) / 1000
            V_serv = span_df['shear'].abs().max() / 1000
            
            design_data.append({
                "span": i+1,
                "M_serv_pos": M_pos_serv, 
                "M_serv_neg": M_neg_serv, 
                "V_serv": V_serv,
                "Mu_pos": M_pos_serv * saf_factor,
                "Mu_neg": M_neg_serv * saf_factor,
                "Vu": V_serv * saf_factor,
                "def_act": span_df['deflection'].abs().max(),
                "L": spans[i]
            })
            design_res_for_plot.append({'pos': {'n': 3}, 'neg': {'n': 2}, 'db': 16})

        # --- 2.1 CONTROLS SECTION (Grouped) ---
        with st.container(border=True):
            st.markdown("**🛠️ Design Controls**")
            col_sel, col_form = st.columns([1, 3])
            
            with col_sel:
                sel_span = st.selectbox("Select Span", design_data, format_func=lambda x: f"Span {x['span']} (L={x['L']}m)")
            
            with col_form:
                with st.form("rebar_form"):
                    cc1, cc2, cc3, cc4, cc5, cc6 = st.columns([1,1,1,1.5,1, 1])
                    n_top = cc1.number_input("Top Bars", 1, 10, 2)
                    n_bot = cc2.number_input("Bot Bars", 1, 10, 3)
                    db_main = cc3.selectbox("DB (mm)", [12, 16, 20, 25], index=1)
                    s_stir = cc4.number_input("Stirrup RB6@", 5, 30, 15, 5)
                    cover = cc5.number_input("Cov(mm)", 20, 50, 40)
                    cc6.write("") # Spacer
                    submitted = cc6.form_submit_button("Update")

        # --- 2.2 REPORT SECTION ---
        # Constants for Calc
        L_mm = sel_span['L'] * 1000
        delta_allow = L_mm / 240.0
        delta_act = sel_span['def_act']
        
        b_mm = p['b'] * 1000
        d_mm = p['h'] * 1000 - cover - 6 - db_main/2
        As_prov = n_bot * (3.1416 * (db_main/2)**2)
        
        # Flexure Calc
        As_min = max((0.25 * np.sqrt(p['fc']) / p['fy']) * b_mm * d_mm, (1.4 / p['fy']) * b_mm * d_mm)
        a_depth = (As_prov * p['fy']) / (0.85 * p['fc'] * b_mm)
        Mn_kNm = As_prov * p['fy'] * (d_mm - a_depth/2) * 1e-6
        phi_Mn = f['phi_m'] * Mn_kNm
        
        # Shear Calc
        Vc_val = 0.17 * np.sqrt(p['fc']) * b_mm * d_mm / 1000.0
        phi_Vc = f['phi_v'] * Vc_val
        Av = 2 * (3.1416 * 3**2) 
        s_mm = s_stir * 10
        Vs_val = (Av * p['fy'] * d_mm) / s_mm / 1000.0
        phi_Vs = f['phi_v'] * Vs_val
        phi_Vn = phi_Vc + phi_Vs

        # Equilibrium Calc
        total_load_y = 0
        for l in full_loads:
            if l['type'] == 'P': total_load_y += l['mag']
            elif l['type'] == 'U': total_load_y += l['mag'] * l['dist']
        sum_reac = sum([abs(r.get('fy', 0)) for r in reac])
        eq_diff = abs(total_load_y - sum_reac)

        # --- DRAWING THE REPORT ---
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="report-header">📝 CALCULATION SHEET: SPAN {sel_span["span"]}</div>', unsafe_allow_html=True)
        
        # Row 1: Checks (Equilibrium & Deflection)
        c_check1, c_check2 = st.columns(2)
        with c_check1:
            st.markdown('<div class="section-title">1. System Equilibrium Check</div>', unsafe_allow_html=True)
            st.latex(rf"\sum F_{{load}} = {total_load_y/1000:.2f} kN, \quad \sum R = {sum_reac/1000:.2f} kN")
            if eq_diff < 10.0: st.markdown('<span class="pass-tag">✅ EQUILIBRIUM OK</span>', unsafe_allow_html=True)
            else: st.markdown(f'<span class="fail-tag">❌ ERROR (Diff: {eq_diff/1000:.2f} kN)</span>', unsafe_allow_html=True)
            
        with c_check2:
            st.markdown('<div class="section-title">2. Serviceability (Deflection)</div>', unsafe_allow_html=True)
            st.latex(rf"\Delta_{{allow}} = {delta_allow:.2f} mm, \quad \Delta_{{actual}} = \mathbf{{{delta_act:.2f}}} mm")
            if delta_act <= delta_allow: st.markdown('<span class="pass-tag">✅ PASS</span>', unsafe_allow_html=True)
            else: st.markdown('<span class="fail-tag">❌ FAIL (Stiffness Insufficient)</span>', unsafe_allow_html=True)

        st.markdown("---")

        # Row 2: Strength Design (Split Columns)
        c_flex, c_shear = st.columns(2)
        
        # --- COL LEFT: FLEXURE ---
        with c_flex:
            st.markdown('<div class="section-title">3. Flexural Design (Moment)</div>', unsafe_allow_html=True)
            st.markdown(f"**Target:** $M_u = {sel_span['Mu_pos']:.2f}$ kNm")
            
            st.markdown("**(a) Steel Area Check**")
            st.latex(rf"A_{{s,prov}} = {As_prov:.0f} mm^2 \quad (A_{{s,min}} = {As_min:.0f})")
            if As_prov >= As_min: st.write("✅ Steel Area OK")
            else: st.markdown('<span class="fail-tag">❌ Check Min Steel</span>', unsafe_allow_html=True)
            
            st.markdown("**(b) Capacity Check**")
            st.latex(rf"\phi M_n = \mathbf{{{phi_Mn:.2f}}} kNm")
            
            if phi_Mn >= sel_span['Mu_pos']: 
                st.markdown(f'<span class="pass-tag">✅ SAFE (Ratio: {sel_span["Mu_pos"]/phi_Mn:.2f})</span>', unsafe_allow_html=True)
            else: 
                st.markdown(f'<span class="fail-tag">❌ UNSAFE (Ratio: {sel_span["Mu_pos"]/phi_Mn:.2f})</span>', unsafe_allow_html=True)

        # --- COL RIGHT: SHEAR ---
        with c_shear:
            st.markdown('<div class="section-title">4. Shear Design</div>', unsafe_allow_html=True)
            st.markdown(f"**Target:** $V_u = {sel_span['Vu']:.2f}$ kN")
            
            st.markdown("**(a) Concrete Capacity**")
            st.latex(rf"\phi V_c = {phi_Vc:.2f} kN")
            
            st.markdown("**(b) Steel Capacity**")
            st.latex(rf"\phi V_s = {phi_Vs:.2f} kN \quad (RB6@{s_stir}cm)")
            
            st.markdown("**(c) Total Capacity**")
            st.latex(rf"\phi V_n = \mathbf{{{phi_Vn:.2f}}} kN")
            
            if phi_Vn >= sel_span['Vu']: 
                st.markdown(f'<span class="pass-tag">✅ SAFE</span>', unsafe_allow_html=True)
            else: 
                st.markdown(f'<span class="fail-tag">❌ UNSAFE (Need closer stirrups)</span>', unsafe_allow_html=True)

        st.markdown("---")
        
        # Row 3: Drawings
        st.markdown('<div class="section-title">5. Detail Drawings</div>', unsafe_allow_html=True)
        c_draw1, c_draw2 = st.columns([1, 2])
        
        with c_draw1:
            st.caption("Cross Section")
            fig_sec = section_plotter.plot_section(p['b'], p['h'], cover, db_main, n_top, n_bot, f"RB6@{s_stir}", p['fc'], p['fy'])
            st.pyplot(fig_sec, use_container_width=True)
            
        with c_draw2:
            st.caption("Longitudinal Profile (Schematic)")
            design_res_for_plot[sel_span['span']-1] = {'pos': {'n': n_bot}, 'neg': {'n': n_top}, 'db': db_main}
            fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_res_for_plot, p['h'], cover)
            st.pyplot(fig_long, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True) # End Report Container
