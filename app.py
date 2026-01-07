import streamlit as st
import pandas as pd
import numpy as np

# Import modules (Ensure these files are in the same directory)
try:
    import input_handler
    import solver
    import rc_design
    import design_view
    import section_plotter
except ImportError as e:
    st.error(f"System Error: {e}")
    st.stop()

st.set_page_config(page_title="Pro Beam Design", layout="wide")

# --- CSS for Reports ---
st.markdown("""
<style>
    .report-frame { border: 1px solid #ddd; padding: 25px; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .header-eng { font-family: 'Helvetica', sans-serif; color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; margin-bottom: 20px; font-weight: bold; }
    .sub-eng { color: #34495e; font-weight: bold; margin-top: 15px; font-size: 1.1em; }
    .calc-table { width: 100%; border-collapse: collapse; font-family: 'Courier New', monospace; font-size: 0.95em; }
    .calc-table th { background-color: #f8f9fa; border-bottom: 2px solid #ddd; padding: 8px; text-align: left; }
    .calc-table td { border-bottom: 1px solid #eee; padding: 8px; }
    .note-box { background-color: #e8f4f8; border-left: 4px solid #0077b6; padding: 10px; font-size: 0.9em; color: #444; margin-bottom: 15px; }
    .status-pass { color: green; font-weight: bold; }
    .status-fail { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- INIT SESSION ---
if 'analyzed' not in st.session_state: st.session_state.analyzed = False

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Project Parameters")
    design_std = st.radio("Design Standard", ["ACI 318-19", "EIT 1008-38"])
    
    # Define Factors based on standard
    if "EIT" in design_std:
        factors = {'DL': 1.4, 'LL': 1.7, 'phi_m': 0.90, 'phi_v': 0.85, 'name': 'EIT 1008'}
        avg_factor = 1.7 
    else:
        factors = {'DL': 1.2, 'LL': 1.6, 'phi_m': 0.90, 'phi_v': 0.75, 'name': 'ACI 318'}
        avg_factor = 1.6
        
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()
    st.divider()
    run_btn = st.button("🚀 EXECUTE ANALYSIS", type="primary", use_container_width=True)

# --- MAIN LOGIC ---
st.title("🏗️ Advanced RC Beam Analysis & Design")

if run_btn:
    if not stable:
        st.error("🚨 Structure is unstable. Please check supports.")
    else:
        # 1. Prepare Loads (Service State for Analysis)
        sw_mag = params['b'] * params['h'] * 24.0 * 1000 # N/m
        
        loads_combined = []
        if not loads_df.empty: 
            loads_combined = loads_df.to_dict('records')
        
        # Add Self-weight
        for i in range(n_spans):
            loads_combined.append({
                "id": f"sw_{i}", "type": "U", "span_index": i, 
                "x": 0.0, "mag": sw_mag, "dist": spans[i]
            })

        # 2. Run Solver
        solver_service = solver.BeamSolver(
            spans, sup_df.to_dict('records'), loads_combined, 
            params['E'], params['b'], params['h'], params['I']
        )
        res, reac_raw, status = solver_service.solve()
        
        if "error" in status: 
            st.error(status['error'])
        else:
            # Process Results
            final_reac_list = []
            if isinstance(reac_raw, list):
                final_reac_list = [{'node_id': i, 'fy': v} for i, v in enumerate(reac_raw)]
            elif isinstance(reac_raw, dict):
                 final_reac_list = [{'node_id': int(k), 'fy': v} for k, v in reac_raw.items()]

            # Store in Session safely
            st.session_state.res = res
            st.session_state.reac = final_reac_list
            st.session_state.loads = loads_combined
            st.session_state.params = params
            st.session_state.factors = factors
            st.session_state.avg_factor = avg_factor
            st.session_state.analyzed = True
            st.rerun() # Rerun to refresh the view immediately

# --- DISPLAY RESULTS ---
# Safe Check: Ensure analyzed is True AND 'res' exists in session state
if st.session_state.analyzed and 'res' in st.session_state:
    
    # Unpack variables using .get() to prevent AttributeErrors
    res = st.session_state.get('res')
    reac = st.session_state.get('reac')
    loads = st.session_state.get('loads')
    p = st.session_state.get('params')
    f = st.session_state.get('factors')
    saf_factor = st.session_state.get('avg_factor', 1.6)
    
    # === TAB LAYOUT ===
    tab1, tab2 = st.tabs(["📊 1. Structural Analysis Results", "📝 2. Detail Design & Checks"])
    
    # --- TAB 1: ANALYSIS ---
    with tab1:
        st.markdown(f"""
        <div class="note-box">
            <b>📜 Analysis Methodology Note:</b><br>
            • <b>Method:</b> Direct Stiffness Method (Finite Element Method - Euler-Bernoulli Beam)<br>
            • <b>Load State:</b> <u>Service Limit State (SLS)</u> - Unfactored Loads (DL + LL).<br>
            • <b>Results:</b> Deflections shown are elastic. For RC Long-term deflection, apply multipliers (ACI 9.5.2.5).
        </div>
        """, unsafe_allow_html=True)

        col_graph, col_data = st.columns([3, 1])
        
        with col_graph:
            st.subheader("Analysis Diagrams")
            # Call Plotter (ensure design_view.py is updated)
            fig = design_view.plot_analysis_results(res, p.get('spans_data',[]), p.get('sup_data',[]), loads, reac)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_data:
            st.markdown("### 🧮 Reaction Calculation")
            
            # --- EQUILIBRIUM CHECK ---
            total_load_down = 0
            for l in loads:
                if l['type'] == 'U': total_load_down += l['mag'] * l['dist']
                elif l['type'] == 'P': total_load_down += l['mag']
            
            total_reac_up = sum([r['fy'] for r in reac])
            
            diff = abs(total_load_down - total_reac_up)
            status_eq = "✅ OK" if diff < 1.0 else "❌ ERROR"
            
            # NOTE: Use double backslash for LaTeX in f-strings
            st.markdown(f"""
            <div style="background-color:#f9f9f9; padding:10px; border-radius:5px; font-size:0.9em;">
            <b>Global Equilibrium Check ($\\Sigma F_y = 0$)</b><br>
            <table class="calc-table">
                <tr><td>Total Load ($\\downarrow$)</td><td>{total_load_down/1000:.2f} kN</td></tr>
                <tr><td>Total Reaction ($\\uparrow$)</td><td>{total_reac_up/1000:.2f} kN</td></tr>
                <tr><td><b>Balance Check</b></td><td><b>{status_eq}</b></td></tr>
            </table>
            </div>
            <br>
            """, unsafe_allow_html=True)
            
            st.markdown("**Detailed Reactions:**")
            for r in reac:
                val = r['fy']/1000
                st.markdown(f"**Node {r['node_id']+1}:** $R_y = {val:.2f}$ kN")

    # --- TAB 2: DESIGN & CALCULATION ---
    with tab2:
        spans_data = p.get('spans_data', [])
        
        if len(spans_data) > 0:
            cum_dist = [0] + list(np.cumsum(spans_data))
            design_data = []
            
            # Prepare Design Data
            for i in range(len(spans_data)):
                x0, x1 = cum_dist[i], cum_dist[i+1]
                span_df = res[(res['x'] >= x0) & (res['x'] <= x1)]
                
                if not span_df.empty:
                    M_max = max(span_df['moment'].abs().max(), 0) / 1000
                    V_max = span_df['shear'].abs().max() / 1000
                    design_data.append({
                        "span": i+1, "Mu": M_max * saf_factor, "Vu": V_max * saf_factor,
                        "def_act": span_df['deflection'].abs().max(), "L": spans_data[i]
                    })
                
            # 2.1 INPUT BAR
            with st.container(border=True):
                st.markdown("#### 🛠️ Reinforcement Configuration")
                col_sel, col_in = st.columns([1, 4])
                
                with col_sel:
                    if design_data:
                        sel_span = st.selectbox("Select Span", design_data, format_func=lambda x: f"Span {x['span']} (Mu={x['Mu']:.1f})")
                    else:
                        st.warning("No data.")
                        st.stop()
                        
                with col_in:
                    with st.form("rebar_form"):
                        c1, c2, c3, c4, c5, c6 = st.columns([1,1,1,1.5,1,1])
                        n_top = c1.number_input("Top", 2, 10, 2)
                        n_bot = c2.number_input("Bot", 2, 10, 3)
                        db_main = c3.selectbox("DB", [12, 16, 20, 25, 28], index=1)
                        s_stir = c4.number_input("Stirrup@(cm)", 5, 50, 15, 5)
                        cover = c5.number_input("Cov(mm)", 20, 50, 40)
                        c6.write("")
                        c6.form_submit_button("Update")
            
            st.write("")

            # 2.2 CALCULATION REPORT
            col_calc, col_img = st.columns([1.5, 1])
            
            # --- Left: Detailed Calculation ---
            with col_calc:
                st.markdown('<div class="report-frame">', unsafe_allow_html=True)
                st.markdown(f'<div class="header-eng">📝 CALCULATION SHEET: SPAN {sel_span["span"]}</div>', unsafe_allow_html=True)
                
                # Variables
                b_mm, h_mm = p['b']*1000, p['h']*1000
                d_mm = h_mm - cover - 9 - db_main/2 
                As_prov = n_bot * 3.1416 * (db_main/2)**2
                
                # FLEXURE
                st.markdown('<div class="sub-eng">1. FLEXURAL DESIGN</div>', unsafe_allow_html=True)
                st.write(f"• **Moment Demand:** $M_u = {sel_span['Mu']:.2f}$ kNm")
                st.write(f"• **Section:** $b={b_mm:.0f}, h={h_mm:.0f}, d={d_mm:.1f}$ mm")
                st.write(f"• **Reinforcement:** {n_bot}-DB{db_main} ($A_s = {As_prov:.0f} mm^2$)")
                
                # Calc
                a = (As_prov * p['fy']) / (0.85 * p['fc'] * b_mm)
                Mn = As_prov * p['fy'] * (d_mm - a/2) * 1e-6
                phi_Mn = f.get('phi_m', 0.9) * Mn
                
                st.latex(rf"a = \frac{{A_s f_y}}{{0.85 f'_c b}} = {a:.2f} mm")
                st.latex(rf"\phi M_n = \phi A_s f_y (d - a/2) = {phi_Mn:.2f} kNm")
                
                status_color = "green" if phi_Mn >= sel_span['Mu'] else "red"
                status_text = "✅ OK" if phi_Mn >= sel_span['Mu'] else "❌ FAIL"
                st.markdown(f'<div style="color:{status_color}; font-weight:bold;">{status_text} (Ratio: {sel_span["Mu"]/phi_Mn:.2f})</div>', unsafe_allow_html=True)

                # SHEAR
                st.markdown('<div class="sub-eng">2. SHEAR DESIGN</div>', unsafe_allow_html=True)
                Vc = 0.17 * np.sqrt(p['fc']) * b_mm * d_mm / 1000
                phi_Vc = f.get('phi_v', 0.85) * Vc
                Av = 2 * 28.27 # RB6
                Vs = (Av * p['fy'] * d_mm) / (s_stir*10) / 1000
                phi_Vs = f.get('phi_v', 0.85) * Vs
                phi_Vn = phi_Vc + phi_Vs
                
                st.write(f"• **Demand:** $V_u = {sel_span['Vu']:.2f}$ kN")
                st.latex(rf"\phi V_c = {phi_Vc:.2f} kN, \quad \phi V_s = {phi_Vs:.2f} kN")
                st.latex(rf"\phi V_n = {phi_Vn:.2f} kN")
                
                if phi_Vn >= sel_span['Vu']: st.success("✅ SHEAR OK")
                else: st.error("❌ SHEAR FAIL")
                
                # DEFLECTION
                st.markdown('<div class="sub-eng">3. SERVICEABILITY</div>', unsafe_allow_html=True)
                d_all = (sel_span['L']*1000)/240
                if sel_span['def_act'] <= d_all: st.write(f"✅ Deflection: {sel_span['def_act']:.2f} mm < {d_all:.1f} mm")
                else: st.write(f"❌ Deflection: {sel_span['def_act']:.2f} mm > {d_all:.1f} mm")
                
                st.markdown('</div>', unsafe_allow_html=True)

            # --- Right: Cross Section Image ---
            with col_img:
                st.write("### Cross Section")
                fig_sec = section_plotter.plot_section(p['b'], p['h'], cover, db_main, n_top, n_bot, f"RB6@{s_stir}cm", p['fc'], p['fy'])
                st.pyplot(fig_sec, use_container_width=True)

            # 2.3 LONGITUDINAL PROFILE
            st.markdown("---")
            st.subheader(f"Longitudinal Profile: Span {sel_span['span']}")
            fig_long = section_plotter.plot_longitudinal_detailed(sel_span['L'], p['h'], cover, n_top, n_bot, db_main, s_stir, sel_span['span'])
            st.pyplot(fig_long, use_container_width=True)

elif st.session_state.analyzed and 'res' not in st.session_state:
    # Handle lost session state smoothly
    st.warning("⚠️ Session expired or data cleared. Please click 'EXECUTE ANALYSIS' again to regenerate results.")
