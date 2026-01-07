import streamlit as st
import pandas as pd
import numpy as np

try:
    import input_handler
    import solver
    import design_view
    import section_plotter # Now this exists!
except ImportError as e:
    st.error(f"Missing Module: {e}")
    st.stop()

st.set_page_config(page_title="Pro Beam Design", layout="wide")

# CSS
st.markdown("""
<style>
    .report-frame { border: 1px solid #ddd; padding: 20px; border-radius: 8px; background: white; }
    .status-ok { color: green; font-weight: bold; }
    .status-fail { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

if 'analyzed' not in st.session_state: st.session_state.analyzed = False

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("⚙️ Project Setup")
    design_std = st.radio("Standard", ["ACI 318", "EIT 1008"])
    
    # Factors
    if "EIT" in design_std:
        factors = {'DL': 1.4, 'LL': 1.7, 'phi_m': 0.90, 'phi_v': 0.85}
        avg_f = 1.7
    else:
        factors = {'DL': 1.2, 'LL': 1.6, 'phi_m': 0.90, 'phi_v': 0.75}
        avg_f = 1.6

    # Render Inputs
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()
    
    st.divider()
    run_btn = st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True)

# --- EXECUTION ---
if run_btn:
    if not stable:
        st.error("Structure Unstable!")
    else:
        # 1. Prepare Data
        sw = params['b'] * params['h'] * 24000 # N/m
        loads_list = loads_df.to_dict('records') if not loads_df.empty else []
        
        # Add SW
        for i in range(n_spans):
            loads_list.append({"id":f"sw{i}", "type":"U", "span_index":i, "x":0.0, "mag":sw, "dist":spans[i]})
            
        # 2. Solve
        s = solver.BeamSolver(spans, sup_df.to_dict('records'), loads_list, params['E'], params['b'], params['h'], params['I'])
        res, reac, status = s.solve()
        
        if "error" in status:
            st.error(status['error'])
        else:
            # 3. SAVE EVERYTHING TO SESSION
            st.session_state.res = res
            st.session_state.reac = [{'node_id':i, 'fy':v} for i,v in enumerate(reac)] if isinstance(reac, list) else [{'node_id':int(k), 'fy':v} for k,v in reac.items()]
            st.session_state.loads = loads_list
            
            # Critical: Save inputs explicitly
            params['spans_data'] = spans
            params['sup_data'] = sup_df.to_dict('records')
            st.session_state.params = params
            st.session_state.factors = factors
            st.session_state.avg_f = avg_f
            
            st.session_state.analyzed = True
            st.rerun()

# --- REPORT VIEW ---
if st.session_state.analyzed and 'res' in st.session_state:
    res = st.session_state.res
    reac = st.session_state.reac
    loads = st.session_state.loads
    p = st.session_state.params
    f = st.session_state.factors
    saf_f = st.session_state.avg_f
    
    # Retrieve Lists safely
    spans_data = p.get('spans_data', [5.0]) # Default fallback
    sup_data = p.get('sup_data', [])

    tab1, tab2 = st.tabs(["📊 Analysis Results", "📝 Detail Design"])
    
    with tab1:
        # Plot
        st.plotly_chart(design_view.plot_analysis_results(res, spans_data, sup_data, loads, reac), use_container_width=True)
        
        # Equilibrium
        sum_load = sum([l['mag']*l['dist'] if l['type']=='U' else l['mag'] for l in loads])
        sum_reac = sum([r['fy'] for r in reac])
        st.info(f"Equilibrium Check: Load {sum_load/1000:.1f} kN vs Reac {sum_reac/1000:.1f} kN")

    with tab2:
        # Prepare Design Data
        if spans_data:
            cum_dist = [0] + list(np.cumsum(spans_data))
            design_list = []
            
            for i in range(len(spans_data)):
                x0, x1 = cum_dist[i], cum_dist[i+1]
                sub = res[(res['x']>=x0) & (res['x']<=x1)]
                if not sub.empty:
                    Mu = max(sub['moment'].abs().max(), 0) / 1000 * saf_f
                    Vu = sub['shear'].abs().max() / 1000 * saf_f
                    design_list.append({'id': i+1, 'Mu': Mu, 'Vu': Vu, 'L': spans_data[i]})
            
            if design_list:
                c1, c2 = st.columns([1, 2])
                with c1:
                    sel = st.selectbox("Select Span", design_list, format_func=lambda x: f"Span {x['id']}")
                    with st.form("design_in"):
                        col_a, col_b = st.columns(2)
                        n_bot = col_a.number_input("Bot Bars", 2, 10, 3)
                        n_top = col_b.number_input("Top Bars", 2, 10, 2)
                        db = col_a.selectbox("DB (mm)", [12,16,20,25], index=1)
                        stir = col_b.number_input("Stirrup (cm)", 5, 50, 15)
                        st.form_submit_button("Update")
                
                with c2:
                    st.subheader(f"Design Span {sel['id']}")
                    # Use the new section plotter
                    try:
                        fig = section_plotter.plot_section(p['b'], p['h'], 40, db, n_top, n_bot, f"@{stir}", p['fc'], p['fy'])
                        st.pyplot(fig)
                        
                        st.write("---")
                        fig2 = section_plotter.plot_longitudinal_detailed(sel['L'], p['h'], 40, n_top, n_bot, db, stir, sel['id'])
                        st.pyplot(fig2)
                    except Exception as e:
                        st.error(f"Plotting Error: {e}")
            else:
                st.warning("No span data found.")
        else:
            st.error("Spans data missing.")
            
elif st.session_state.analyzed:
    st.warning("State lost. Please Re-Run.")
