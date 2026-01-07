import streamlit as st
import pandas as pd
import numpy as np

# Import modules
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

# --- CSS ---
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

if 'analyzed' not in st.session_state: st.session_state.analyzed = False

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Project Parameters")
    design_std = st.radio("Design Standard", ["ACI 318-19", "EIT 1008-38"])
    
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
        # 1. Prepare Loads
        sw_mag = params['b'] * params['h'] * 24.0 * 1000 
        loads_combined = []
        if not loads_df.empty: 
            loads_combined = loads_df.to_dict('records')
        for i in range(n_spans):
            loads_combined.append({"id": f"sw_{i}", "type": "U", "span_index": i, "x": 0.0, "mag": sw_mag, "dist": spans[i]})

        # 2. Run Solver
        solver_service = solver.BeamSolver(
            spans, sup_df.to_dict('records'), loads_combined, 
            params['E'], params['b'], params['h'], params['I']
        )
        res, reac_raw, status = solver_service.solve()
        
        if "error" in status: 
            st.error(status['error'])
        else:
            final_reac_list = []
            if isinstance(reac_raw, list):
                final_reac_list = [{'node_id': i, 'fy': v} for i, v in enumerate(reac_raw)]
            elif isinstance(reac_raw, dict):
                 final_reac_list = [{'node_id': int(k), 'fy': v} for k, v in reac_raw.items()]

            # --- CRITICAL FIX: Add Extra Data to Params before saving ---
            params['spans_data'] = spans
            params['sup_data'] = sup_df.to_dict('records') # Convert DF to list here for safety

            # Store in Session
            st.session_state.res = res
            st.session_state.reac = final_reac_list
            st.session_state.loads = loads_combined
            st.session_state.params = params
            st.session_state.factors = factors
            st.session_state.avg_factor = avg_factor
            st.session_state.analyzed = True
            st.rerun()

# --- DISPLAY RESULTS ---
if st.session_state.analyzed and 'res' in st.session_state:
    
    res = st.session_state.get('res')
    reac = st.session_state.get('reac')
    loads = st.session_state.get('loads')
    p = st.session_state.get('params')
    f = st.session_state.get('factors')
    saf_factor = st.session_state.get('avg_factor', 1.6)
    
    # Check if critical keys exist in 'p', if not provide defaults
    spans_data = p.get('spans_data', []) 
    sup_data = p.get('sup_data', []) 

    tab1, tab2 = st.tabs(["📊 1. Structural Analysis Results", "📝 2. Detail Design & Checks"])
    
    # --- TAB 1: ANALYSIS ---
    with tab1:
        st.markdown(f"""
        <div class="note-box">
            <b>📜 Methodology:</b> Direct Stiffness Method (SLS Loads).<br>
            <b>Note:</b> Deflections are elastic.
        </div>
        """, unsafe_allow_html=True)

        col_graph, col_data = st.columns([3, 1])
        
        with col_graph:
            st.subheader("Analysis Diagrams")
            # Now passing list of dicts (sup_data) which design_view can handle
            fig = design_view.plot_analysis_results(res, spans_data, sup_data, loads, reac)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_data:
            st.markdown("### 🧮 Reactions")
            total_load_down = sum([l['mag']*l['dist'] if l['type']=='U' else l['mag'] for l in loads])
            total_reac_up = sum([r['fy'] for r in reac])
            diff = abs(total_load_down - total_reac_up)
            status_eq = "✅ OK" if diff < 1.0 else "❌ ERROR"
            
            st.markdown(f"""
            <div style="background-color:#f9f9f9; padding:10px;">
            <b>Equilibrium Check ($\\Sigma F_y = 0$)</b><br>
            Load: {total_load_down/1000:.2f} kN<br>
            Reac: {total_reac_up/1000:.2f} kN<br>
            Result: <b>{status_eq}</b>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("---")
            for r in reac:
                st.markdown(f"**Node {r['node_id']+1}:** {r['fy']/1000:.2f} kN")

    # --- TAB 2: DESIGN ---
    with tab2:
        if len(spans_data) > 0:
            cum_dist = [0] + list(np.cumsum(spans_data))
            design_data = []
            
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
                
            with st.container(border=True):
                c_sel, c_in = st.columns([1, 3])
                with c_sel:
                    sel_span = st.selectbox("Select Span", design_data, format_func=lambda x: f"Span {x['span']}")
                with c_in:
                    with st.form("rebar"):
                        c1,c2,c3,c4,c5,c6 = st.columns(6)
                        n_top = c1.number_input("Top", 2, 10, 2)
                        n_bot = c2.number_input("Bot", 2, 10, 3)
                        db = c3.selectbox("DB", [12,16,20,25], index=1)
                        stir = c4.number_input("Stir@(cm)", 5, 50, 15)
                        cov = c5.number_input("Cov", 20, 50, 30)
                        c6.write(""); c6.form_submit_button("Update")

            col_calc, col_img = st.columns([1.5, 1])
            with col_calc:
                st.markdown(f"### Design Span {sel_span['span']}")
                b, h = p['b']*1000, p['h']*1000
                d = h - cov - 9 - db/2
                As = n_bot * 3.1416 * (db/2)**2
                
                # Simple Calcs
                a = (As*p['fy']) / (0.85*p['fc']*b)
                PhiMn = f.get('phi_m',0.9) * As * p['fy'] * (d - a/2) * 1e-6
                
                st.write(f"**Flexure:** Mu={sel_span['Mu']:.2f} kNm | PhiMn={PhiMn:.2f} kNm")
                if PhiMn >= sel_span['Mu']: st.success(f"✅ OK (Ratio {sel_span['Mu']/PhiMn:.2f})")
                else: st.error(f"❌ FAIL (Ratio {sel_span['Mu']/PhiMn:.2f})")
                
                st.write("---")
                
                Vc = 0.17 * np.sqrt(p['fc']) * b * d / 1000
                Vs = (2*28.27 * p['fy'] * d) / (stir*10) / 1000
                PhiVn = f.get('phi_v',0.85) * (Vc + Vs)
                
                st.write(f"**Shear:** Vu={sel_span['Vu']:.2f} kN | PhiVn={PhiVn:.2f} kN")
                if PhiVn >= sel_span['Vu']: st.success("✅ OK")
                else: st.error("❌ FAIL")

            with col_img:
                fig_sec = section_plotter.plot_section(p['b'], p['h'], cov, db, n_top, n_bot, f"@{stir}", p['fc'], p['fy'])
                st.pyplot(fig_sec)

elif st.session_state.analyzed:
    st.warning("Data lost. Please run analysis again.")
