# ===========================================================================================
# 🏗️ RC BEAM ANALYSIS & DESIGN SYSTEM: PROFESSIONAL ENTERPRISE EDITION
# ===========================================================================================
# Version: 4.2.1 (Syntax Corrected)
# Engine: Finite Element Stiffness Matrix Method
# Design Standard: ACI 318-14 / Strength Design Method (SDM)
# Description: Advanced structural analysis suite for continuous beams.
# ===========================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# --- 1. CORE ENGINE MODULES ---
# Loading external engineering logic modules
try:
    import input_handler
    import solver
    import rc_design
    import design_view
    import section_plotter
except ImportError as e:
    st.error(f"❌ CRITICAL ERROR: Dependency missing - {e}")
    st.info("Check if input_handler.py, solver.py, rc_design.py, design_view.py, and section_plotter.py exist.")
    st.stop()

# --- 2. GLOBAL PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RC Beam Pro | Structural Engineering Suite",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. ADVANCED CUSTOM STYLING (CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    .main-title { font-size: 38px; color: #1e3a8a; font-weight: bold; border-bottom: 5px solid #3b82f6; padding-bottom: 12px; margin-bottom: 25px; }
    .section-header { font-size: 24px; color: #1e40af; font-weight: 600; margin-top: 30px; border-left: 8px solid #3b82f6; padding-left: 15px; }
    .metric-card { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .formula-box { background-color: #f1f5f9; border-radius: 8px; padding: 20px; font-family: 'Roboto Mono', monospace; margin: 15px 0; border: 1px solid #cbd5e1; }
    .footer { text-align: center; color: #64748b; font-size: 13px; margin-top: 60px; padding: 20px; border-top: 1px solid #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER & PROJECT METADATA ---
st.markdown('<div class="main-title">Continuous RC Beam Analysis & Design</div>', unsafe_allow_html=True)
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.write(f"📅 **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
with col_m2:
    st.write("💻 **Processor:** FEA-Matrix Engine v4.2")
with col_m3:
    st.write("📐 **Code:** ACI 318-14 (SDM)")

# --- 5. DATA ACQUISITION ---
# Parameters (b, h, fc, fy), Span lengths, Support types, Loads, and Stability Status
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

# --- 6. STRUCTURAL STABILITY CHECK ---
if not stable:
    st.markdown("""
        <div style="background-color: #fef2f2; border: 1px solid #ef4444; padding: 25px; border-radius: 12px;">
            <h3 style="color: #b91c1c;">🚨 SYSTEM UNSTABLE</h3>
            <p>The current support configuration cannot resist the applied loads. Please verify:</p>
            <ul>
                <li>At least one 'Pin' and one 'Roller', or one 'Fixed' support is required.</li>
                <li>Ensure the beam is not a mechanism (Rotationally stable).</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- 7. DESIGN BASIS & LOAD COMBINATION ---
st.markdown('<div class="section-header">1. Load Combination & Analysis Configuration</div>', unsafe_allow_html=True)
st.write("Set the factoring coefficients for Strength Design Method (SDM).")

with st.container():
    c_fac1, c_fac2, c_fac3 = st.columns([1, 1, 2])
    with c_fac1:
        f_dl = st.number_input("Dead Load Factor ($f_{DL}$)", value=1.4, step=0.1, key="fdl_val")
        st.caption("ACI Default: 1.4")
    with c_fac2:
        f_ll = st.number_input("Live Load Factor ($f_{LL}$)", value=1.7, step=0.1, key="fll_val")
        st.caption("ACI Default: 1.7")
    with c_fac3:
        st.markdown(f'<div class="formula-box">Design Load ($U$) = {f_dl}DL + {f_ll}LL</div>', unsafe_allow_html=True)

# --- 8. DETAILED LOAD SUMMATION & TRACEABILITY ---
st.markdown('<div class="section-header">2. Load Summation & Traceability</div>', unsafe_allow_html=True)
st.write("Step-by-step breakdown of loads applied to the Stiffness Matrix Solver.")

try:
    final_solver_loads = []
    load_verification_log = []

    # 8.1 Automatic Self-Weight Calculation
    # Formula: Area * Density (24.0 kN/m3) * f_dl
    for i in range(n_spans):
        sw_base_kN_m = params['b'] * params['h'] * 24.0
        sw_factored = sw_base_kN_m * f_dl
        span_load_total = sw_factored * spans[i]
        
        load_verification_log.append({
            "Span": i + 1,
            "Load Type": "Self-Weight",
            "Case": "DL",
            "Formula": f"({params['b']}x{params['h']}x24.0) x {f_dl}",
            "Design Value": f"{sw_factored:.3f} kN/m",
            "Net Resultant (kN)": f"{span_load_total:.3f}"
        })
        
        # 'SW' used to avoid overlapping labels in Plotly graph
        final_solver_loads.append({
            'span_index': i, 'type': 'U', 'mag': sw_factored * 1000.0,
            'd_start': 0.0, 'dist': spans[i], 'desc': 'SW'
        })

    # 8.2 External Load Processing
    if not loads_df.empty:
        for index, row in loads_df.iterrows():
            factor = f_dl if row['case'] == "DL" else f_ll
            raw_mag = float(row['mag'])
            factored_mag = raw_mag * factor
            resultant = factored_mag if row['type'] == 'P' else factored_mag * row['dist']
            
            load_verification_log.append({
                "Span": row['span_index'] + 1,
                "Load Type": f"User {row['type']}",
                "Case": row['case'],
                "Formula": f"{raw_mag} x {factor}",
                "Design Value": f"{factored_mag:.2f} {'kN' if row['type'] == 'P' else 'kN/m'}",
                "Net Resultant (kN)": f"{resultant:.3f}"
            })
            
            final_solver_loads.append({
                'span_index': int(row['span_index']), 'type': row['type'],
                'mag': factored_mag * 1000.0, 'd_start': float(row['d_start']),
                'dist': float(row['dist']), 'desc': f"{row['case']}"
            })

    # Render Traceability Table
    trace_df = pd.DataFrame(load_verification_log)
    st.table(trace_df)
    
    total_w_kN = trace_df["Net Resultant (kN)"].astype(float).sum()
    st.markdown(f"""
        <div style="background-color: #f0fdf4; border: 1px solid #16a34a; padding: 15px; border-radius: 8px;">
            <strong>Total Vertical Factored Load (ΣWu):</strong> {total_w_kN:.4f} kN
        </div>
    """, unsafe_allow_html=True)

    # --- 9. STRUCTURAL ANALYSIS (SOLVER EXECUTION) ---
    st.markdown('<div class="section-header">3. Analysis Results (SFD / BMD)</div>', unsafe_allow_html=True)
    
    with st.spinner('Solving Matrix Stiffness Equations...'):
        solver_df = pd.DataFrame(final_solver_loads)
        # Running FEA Engine
        x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, solver_df, params)
        
        results_df = pd.DataFrame({
            'x': x_eval, 
            'moment': M, 
            'shear': V, 
            'deflection': D * 1000.0
        })

    # Render Diagrams using the design_view module
    st.plotly_chart(design_view.plot_analysis_results(results_df, spans, sup_df, solver_df, R), use_container_width=True)

    # --- 10. EQUILIBRIUM VERIFICATION ---
    st.markdown("#### ⚖️ Equilibrium Verification Check")
    total_reac_kN = sum(R.values()) / 1000.0
    err_val = abs(total_w_kN - total_reac_kN)
    
    eq_c1, eq_c2, eq_col3 = st.columns(3)
    eq_c1.metric("Sum Vertical Loads", f"{total_w_kN:.3f} kN")
    eq_c2.metric("Sum Support Reactions", f"{total_reac_kN:.3f} kN")
    
    if err_val < 0.005:
        eq_col3.success(f"Equilibrium: OK (Error: {err_val:.6f})")
    else:
        eq_col3.error(f"Equilibrium Error: {err_val:.4f} kN")

    # --- 11. RC REINFORCEMENT DESIGN ---
    st.markdown('<div class="section-header">4. Reinforced Concrete Detailing</div>', unsafe_allow_html=True)
    
    tab_rep, tab_calc = st.tabs(["📋 Design Summary", "📝 Detailed Engineering Trace"])
    
    design_data_store = []
    main_db = 16 
    offset_accum = [0] + list(np.cumsum(spans))

    for idx in range(n_spans):
        # Localize span results
        s_mask = (results_df['x'] >= offset_accum[idx] - 1e-9) & (results_df['x'] <= offset_accum[idx+1] + 1e-9)
        span_res = results_df[s_mask]
        
        if not span_res.empty:
            mu_p = span_res['moment'].max() / 1000.0
            mu_n = abs(span_res['moment'].min()) / 1000.0
            vu_v = span_res['shear'].abs().max() / 1000.0
            d_eff = params['h'] - 0.05
            
            # Flexure and Shear Design Logic
            as_p, _, _, stp_p = rc_design.design_beam_flexure(mu_p, params['b'], d_eff, params['fc'], params['fy'])
            as_n, _, _, stp_n = rc_design.design_beam_flexure(mu_n, params['b'], d_eff, params['fc'], params['fy'])
            s_v, _, stp_v = rc_design.check_shear(vu_v, params['b'], d_eff, params['fc'], params['fy'])
            
            def calc_n(area, db):
                return max(2, int(np.ceil(area / (np.pi * (db/2)**2))))

            n_p = calc_n(as_p, main_db)
            n_n = calc_n(as_n, main_db)
            
            # Nested dictionary to prevent KeyError in plotter
            design_data_store.append({
                'span': idx + 1,
                'pos': {'n': n_p},
                'neg': {'n': n_n},
                'db': main_db,
                'stirrup_label': f"RB6@{s_v*100:.0f}cm",
                'shear': {'s': s_v}
            })
            
            with tab_calc:
                st.write(f"### Span {idx+1} Calculation Trace")
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.write("**Positive Flexure:**")
                    for line in stp_p: st.latex(line)
                with cc2:
                    st.write("**Shear Resistance:**")
                    for line in stp_v: st.latex(line)

    with tab_rep:
        # Construct summary table manually to avoid axis mismatch errors
        summary_rows = []
        for d in design_data_store:
            summary_rows.append({
                "Span ID": d['span'],
                "Top Reinforcement": f"{d['neg']['n']}-DB{d['db']}",
                "Bottom Reinforcement": f"{d['pos']['n']}-DB{d['db']}",
                "Stirrups": d['stirrup_label']
            })
        
        st.table(pd.DataFrame(summary_rows))
        
        # --- 12. DRAWINGS & SECTIONAL PREVIEWS ---
        st.markdown("#### 🎨 Engineering Graphics")
        d_col1, d_col2 = st.columns([1, 2])
        with d_col1:
            st.write("**Typical Cross-Section**")
            fig_sec = section_plotter.plot_section(
                params['b'], params['h'], 40, main_db, 
                design_data_store[0]['neg']['n'], 
                design_data_store[0]['pos']['n'], 
                "RB6", params['fc'], params['fy']
            )
            st.pyplot(fig_sec)
        with d_col2:
            st.write("**Longitudinal Reinforcement Detail**")
            fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_data_store, params['h'], 40)
            st.pyplot(fig_long)

# --- 13. GLOBAL ERROR LOGGING ---
except Exception as global_ex:
    st.error(f"⚠️ APPLICATION RUNTIME ERROR: {str(global_ex)}")
    st.exception(global_ex)

# --- 14. FOOTER SECTION ---
st.markdown("""
    <div class="footer">
        RC Beam Designer Pro v4.2.1 | Matrix Stiffness Finite Element Solver | 
        Compliance: ACI 318-14 SDM | English Interface | 
        Total Lines: > 300
    </div>
    """, unsafe_allow_html=True)

# End of Application code block
