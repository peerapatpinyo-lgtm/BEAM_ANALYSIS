import streamlit as st
import pandas as pd
import numpy as np

try:
    import input_handler
    import solver
    import design_view
    import section_plotter
except ImportError as e:
    st.error(f"Missing Module: {e}")
    st.stop()

st.set_page_config(page_title="Pro Beam Design", layout="wide")

# CSS
st.markdown("""
<style>
    .report-box { border: 1px solid #ccc; padding: 20px; background: #fff; margin-bottom: 20px; border-radius: 5px; }
    .header-eng { font-size: 1.2em; font-weight: bold; border-bottom: 2px solid #333; margin-bottom: 10px; color: #2c3e50; }
    .sub-head { font-weight: bold; color: #007bff; margin-top: 10px; }
    .pass { color: green; font-weight: bold; }
    .fail { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

if 'analyzed' not in st.session_state: st.session_state.analyzed = False

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Settings")
    design_std = st.radio("Standard", ["ACI 318", "EIT 1008"])
    if "EIT" in design_std:
        factors = {'DL': 1.4, 'LL': 1.7, 'phi_m': 0.90, 'phi_v': 0.85}; avg_f = 1.7
    else:
        factors = {'DL': 1.2, 'LL': 1.6, 'phi_m': 0.90, 'phi_v': 0.75}; avg_f = 1.6

    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()
    st.divider()
    run_btn = st.button("🚀 EXECUTE ANALYSIS", type="primary", use_container_width=True)

# EXECUTION
if run_btn:
    if not stable: st.error("Unstable!")
    else:
        sw = params['b'] * params['h'] * 24000
        loads = loads_df.to_dict('records') if not loads_df.empty else []
        for i in range(n_spans):
            loads.append({"id":f"sw{i}", "type":"U", "span_index":i, "x":0.0, "mag":sw, "dist":spans[i]})

        # SOLVER call
        s = solver.BeamSolver(spans, sup_df.to_dict('records'), loads, params['E'], params['b'], params['h'], params['I'])
        res, reac, status = s.solve()

        if "error" in status: st.error(status['error'])
        else:
            st.session_state.res = res
            st.session_state.reac = reac # dict {node_id: val}
            st.session_state.loads = loads
            params['spans_data'] = spans
            params['sup_data'] = sup_df.to_dict('records')
            st.session_state.params = params
            st.session_state.factors = factors
            st.session_state.avg_f = avg_f
            st.session_state.analyzed = True
            st.rerun()

# RESULTS
if st.session_state.analyzed and 'res' in st.session_state:
    res = st.session_state.get('res')
    reac = st.session_state.get('reac')
    loads = st.session_state.get('loads')
    p = st.session_state.get('params')
    f = st.session_state.get('factors')
    saf_f = st.session_state.get('avg_f', 1.6)
    
    spans = p.get('spans_data', [])
    sups = p.get('sup_data', [])
    
    # Convert reac dict to list for plotter if needed
    reac_list = [{'node_id': k, 'fy': v} for k,v in reac.items()]

    tab1, tab2 = st.tabs(["📊 Analysis Results", "📝 Detail Design"])

    with tab1:
        st.info("ℹ️ Calculation Method: **Timoshenko Beam Theory** (Includes Shear Deformation)")
        st.plotly_chart(design_view.plot_analysis_results(res, spans, sups, loads, reac_list), use_container_width=True)
        
        # Equilibrium
        sum_load = sum([l['mag']*l['dist'] if l['type']=='U' else l['mag'] for l in loads])
        sum_reac = sum(reac.values())
        st.write(f"**Check:** Load {sum_load/1000:.2f} kN vs Reac {sum_reac/1000:.2f} kN")

    with tab2:
        # Prepare Data
        cum_dist = [0] + list(np.cumsum(spans))
        design_list = []
        for i in range(len(spans)):
            x0, x1 = cum_dist[i], cum_dist[i+1]
            sub = res[(res['x']>=x0) & (res['x']<=x1)]
            if not sub.empty:
                Mu = max(sub['moment'].abs().max(), 0)/1000 * saf_f
                Vu = sub['shear'].abs().max()/1000 * saf_f
                design_list.append({'id':i+1, 'Mu':Mu, 'Vu':Vu, 'L':spans[i]})
        
        if design_list:
            c1, c2 = st.columns([1, 1.5])
            with c1:
                sel = st.selectbox("Select Span", design_list, format_func=lambda x: f"Span {x['id']}")
                with st.form("design_input"):
                    cc1, cc2 = st.columns(2)
                    n_bot = cc1.number_input("Bot Bars", 2, 10, 3)
                    n_top = cc2.number_input("Top Bars", 2, 10, 2)
                    db = cc1.selectbox("DB", [12,16,20,25], index=1)
                    stir = cc2.number_input("Stirrup (cm)", 5, 40, 15)
                    cov = st.number_input("Cover (mm)", 20, 50, 30)
                    st.form_submit_button("Calc")
            
            with c2:
                # --- CALCULATION SHEET RESTORED ---
                st.markdown('<div class="report-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="header-eng">📄 CALCULATION SHEET: SPAN {sel["id"]}</div>', unsafe_allow_html=True)
                
                b, h = p['b']*1000, p['h']*1000
                d = h - cov - 9 - db/2
                As = n_bot * 3.1416 * (db/2)**2
                
                # 1. Flexure
                st.markdown('<div class="sub-head">1. FLEXURE DESIGN</div>', unsafe_allow_html=True)
                st.write(f"• Demand: $M_u = {sel['Mu']:.2f}$ kNm")
                
                a = (As * p['fy']) / (0.85 * p['fc'] * b)
                Mn = As * p['fy'] * (d - a/2) * 1e-6
                PhiMn = f.get('phi_m', 0.9) * Mn
                
                st.latex(rf"a = {a:.2f} mm")
                st.latex(rf"\phi M_n = {PhiMn:.2f} kNm")
                
                if PhiMn >= sel['Mu']: st.markdown('<span class="pass">✅ OK</span>', unsafe_allow_html=True)
                else: st.markdown('<span class="fail">❌ FAIL</span>', unsafe_allow_html=True)
                
                # 2. Shear
                st.markdown('<div class="sub-head">2. SHEAR DESIGN</div>', unsafe_allow_html=True)
                st.write(f"• Demand: $V_u = {sel['Vu']:.2f}$ kN")
                
                Vc = 0.17 * np.sqrt(p['fc']) * b * d / 1000
                Vs = (2 * 28.27 * p['fy'] * d) / (stir*10) / 1000
                PhiVn = f.get('phi_v', 0.85) * (Vc + Vs)
                
                st.latex(rf"\phi V_c = {f.get('phi_v', 0.85)*Vc:.2f} kN")
                st.latex(rf"\phi V_s = {f.get('phi_v', 0.85)*Vs:.2f} kN (RB6@{stir}cm)")
                st.latex(rf"\phi V_n = {PhiVn:.2f} kN")
                
                if PhiVn >= sel['Vu']: st.markdown('<span class="pass">✅ OK</span>', unsafe_allow_html=True)
                else: st.markdown('<span class="fail">❌ FAIL</span>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

            # Images
            st.markdown("---")
            c_img1, c_img2 = st.columns(2)
            with c_img1:
                 st.pyplot(section_plotter.plot_section(p['b'], p['h'], cov, db, n_top, n_bot, f"@{stir}", p['fc'], p['fy']))
            with c_img2:
                 st.pyplot(section_plotter.plot_longitudinal_detailed(sel['L'], p['h'], cov, n_top, n_bot, db, stir, sel['id']))

elif st.session_state.analyzed:
    st.warning("Data Missing. Please Re-run Analysis.")
