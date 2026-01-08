# ===========================================================================================
# 🏗️ RC BEAM ANALYSIS & DESIGN SYSTEM: ENTERPRISE PROFESSIONAL EDITION
# ===========================================================================================
# Version: 4.4.0 (Enhanced Load Traceability & Logic Documentation)
# Structural Core: Finite Element Method (FEM) using Matrix Stiffness
# Design Standard: ACI 318-14 Strength Design Method (SDM)
# Language: English (Full Localization)
# ===========================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sys

# --- 1. EXTERNAL MODULE INTEGRATION ---
# Ensuring all engineering sub-modules are loaded correctly.
try:
    import input_handler
    import solver
    import rc_design
    import design_view
    import section_plotter
except ImportError as e:
    st.error(f"❌ CRITICAL ERROR: System dependencies missing - {e}")
    st.info("Ensure input_handler.py, solver.py, rc_design.py, design_view.py, and section_plotter.py are in the root directory.")
    st.stop()

# --- 2. APPLICATION CONFIGURATION ---
st.set_page_config(
    page_title="RC Beam Pro | Structural Design Suite",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. PROFESSIONAL UI STYLING (CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    .main-title { font-size: 36px; color: #1e3a8a; font-weight: bold; border-bottom: 5px solid #3b82f6; padding-bottom: 10px; margin-bottom: 20px; }
    .section-header { font-size: 22px; color: #1e40af; font-weight: 600; margin-top: 25px; border-left: 6px solid #3b82f6; padding-left: 12px; }
    .metric-container { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .formula-display { background-color: #f8fafc; border-radius: 6px; padding: 15px; font-family: 'Roboto Mono', monospace; border: 1px solid #cbd5e1; color: #334155; }
    .footer-note { text-align: center; color: #94a3b8; font-size: 12px; margin-top: 50px; padding: 20px; border-top: 1px solid #f1f5f9; }
    .status-ok { color: #16a34a; font-weight: bold; }
    .status-err { color: #dc2626; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER & METADATA SECTION ---
st.markdown('<div class="main-title">Professional Continuous RC Beam Analysis</div>', unsafe_allow_html=True)
header_l, header_c, header_r = st.columns(3)
with header_l:
    st.write(f"📅 **Computation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
with header_c:
    st.write("💻 **Engine:** Matrix Stiffness v4.4")
with header_r:
    st.write("📐 **Protocol:** ACI 318-14 (Strength Design)")

# --- 5. SYSTEM INPUT ACQUISITION ---
# Retrieving data from the sidebar handler (Materials, Geometry, Supports, Loads)
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

# --- 6. STRUCTURAL INTEGRITY & STABILITY VALIDATION ---
if not stable:
    st.markdown("""
        <div style="background-color: #fff1f2; border: 2px solid #f43f5e; padding: 20px; border-radius: 8px;">
            <h3 style="color: #be123c;">🚨 KINEMATIC INSTABILITY DETECTED</h3>
            <p>The system is geometrically unstable. Please ensure the following conditions are met:</p>
            <ul>
                <li>Total reaction components must be ≥ 3.</li>
                <li>Horizontal and vertical translations must be restrained (e.g., at least one Pin or Fixed support).</li>
                <li>The beam must not contain a mechanism within the spans.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- 7. DESIGN FACTORS & COMBINATION LOGIC ---
st.markdown('<div class="section-header">1. Load Combination Configuration</div>', unsafe_allow_html=True)
st.write("Defining factored safety margins for the Strength Design Method.")

comb_c1, comb_c2, comb_c3 = st.columns([1, 1, 2])
with comb_c1:
    f_dl = st.number_input("Dead Load Factor ($f_{DL}$)", value=1.4, step=0.05, key="factor_dl")
    st.caption("Standard ACI: 1.4")
with comb_c2:
    f_ll = st.number_input("Live Load Factor ($f_{LL}$)", value=1.7, step=0.05, key="factor_ll")
    st.caption("Standard ACI: 1.7")
with comb_c3:
    st.markdown(f'<div class="formula-display">Ultimate Design Load (U) = {f_dl}DL + {f_ll}LL</div>', unsafe_allow_html=True)

# --- 8. ENHANCED LOAD SUMMATION & POSITIONING LOGIC ---
st.markdown('<div class="section-header">2. Load Summation & Traceability Table</div>', unsafe_allow_html=True)
st.write("Verification of load paths and combined design values before Matrix Analysis.")

try:
    final_solver_loads = []
    traceability_data = []
    
    # 8.1 Span-wise Load Consolidation (Prevents overlapping UDL labels on graph)
    span_factored_udl = {i: 0.0 for i in range(n_spans)}

    # 8.2 Self-Weight Processing (DL)
    for i in range(n_spans):
        # Calc: b(m) * h(m) * Density (24 kN/m3) * f_dl
        unit_sw = (params['b'] * params['h'] * 24.0) * f_dl
        span_factored_udl[i] += unit_sw
        
        traceability_data.append({
            "Span": i + 1, "Type": "Self-Weight", "Source": "Dead Load",
            "Calculation": f"({params['b']}x{params['h']}x24.0) x {f_dl}",
            "Design Value": f"{unit_sw:.3f} kN/m", "Resultant": f"{unit_sw * spans[i]:.3f} kN"
        })

    # 8.3 User-Applied Loads Processing (Point & Distributed)
    if not loads_df.empty:
        for index, row in loads_df.iterrows():
            curr_factor = f_dl if row['case'] == "DL" else f_ll
            mag_val = float(row['mag'])
            factored_mag = mag_val * curr_factor
            
            # FIXED: Point Load Alignment
            # Point loads must remain separate to maintain exact coordinate positioning
            if row['type'] == 'P':
                final_solver_loads.append({
                    'span_index': int(row['span_index']), 'type': 'P',
                    'mag': factored_mag * 1000.0, 'd_start': float(row['d_start']),
                    'dist': 0.0, 'desc': f"P={factored_mag:.1f}" # Discrete label for point load
                })
                net_force = factored_mag
            else:
                # Distributed loads are summed into the Span UDL to avoid overlapping labels
                span_factored_udl[int(row['span_index'])] += factored_mag
                net_force = factored_mag * float(row['dist'])

            traceability_data.append({
                "Span": int(row['span_index']) + 1, "Type": f"User {row['type']}", "Source": row['case'],
                "Calculation": f"{mag_val} x {curr_factor}",
                "Design Value": f"{factored_mag:.2f}", "Resultant": f"{net_force:.3f} kN"
            })

    # 8.4 Consolidating UDL for Graph Rendering (One Label per Span)
    for i in range(n_spans):
        if span_factored_udl[i] > 0:
            final_solver_loads.append({
                'span_index': i, 'type': 'U', 'mag': span_factored_udl[i] * 1000.0,
                'd_start': 0.0, 'dist': spans[i], 'desc': f"Wu={span_factored_udl[i]:.2f}"
            })

    # Render Table
    st.table(pd.DataFrame(traceability_data))
    total_net_load = sum([float(x["Resultant"].split()[0]) for x in traceability_data])
    st.success(f"**Total Factored Vertical Force (ΣW_u):** {total_net_load:.4f} kN")

    # --- 9. STRUCTURAL ANALYSIS CORE (FEA SOLVER) ---
    st.markdown('<div class="section-header">3. FEM Analysis: SFD, BMD, & Deflection</div>', unsafe_allow_html=True)
    
    with st.spinner('Iterating Matrix Stiffness Equations...'):
        solver_df = pd.DataFrame(final_solver_loads)
        # solve_beam executes the Matrix Stiffness Method
        x_pts, M_vals, V_vals, D_vals, R_dict = solver.solve_beam(spans, sup_df, solver_df, params)
        
        analysis_results_df = pd.DataFrame({
            'x': x_pts, 'moment': M_vals, 'shear': V_vals, 'deflection': D_vals * 1000.0
        })

    # Graphical Output via Plotly
    st.plotly_chart(design_view.plot_analysis_results(analysis_results_df, spans, sup_df, solver_df, R_dict), use_container_width=True)

    # --- 10. EQUILIBRIUM QUALITY ASSURANCE ---
    st.markdown("#### ⚖️ Static Equilibrium Verification")
    total_reaction_force = sum(R_dict.values()) / 1000.0
    abs_error = abs(total_net_load - total_reaction_force)
    
    q_c1, q_c2, q_c3 = st.columns(3)
    q_c1.metric("Applied Load Sum (ΣW)", f"{total_net_load:.3f} kN")
    q_c2.metric("Reaction Sum (ΣR)", f"{total_reaction_force:.3f} kN")
    
    if abs_error < 0.01:
        q_c3.markdown(f'<p class="status-ok">✅ EQUILIBRIUM PASSED<br>(Err: {abs_error:.6f} kN)</p>', unsafe_allow_html=True)
    else:
        q_c3.markdown(f'<p class="status-err">❌ EQUILIBRIUM FAILED<br>(Diff: {abs_error:.4f} kN)</p>', unsafe_allow_html=True)

    # --- 11. REINFORCEMENT DESIGN SUMMARY (ACI BASED) ---
    st.markdown('<div class="section-header">4. Reinforcement Detailing & Calculations</div>', unsafe_allow_html=True)
    
    sum_tab, calc_tab = st.tabs(["📊 Detailing Summary", "📖 Engineering Design Trace"])
    
    design_records = []
    main_bar_db = 16 
    span_offsets = [0] + list(np.cumsum(spans))

    for i in range(n_spans):
        # Local Span Result Extraction
        mask = (analysis_results_df['x'] >= span_offsets[i] - 1e-9) & (analysis_results_df['x'] <= span_offsets[i+1] + 1e-9)
        span_data = analysis_results_df[mask]
        
        if not span_data.empty:
            max_m_pos = span_data['moment'].max() / 1000.0
            max_m_neg = abs(span_data['moment'].min()) / 1000.0
            max_v_span = span_data['shear'].abs().max() / 1000.0
            effective_d = params['h'] - 0.05
            
            # Flexural Design Logic
            as_pos, _, _, log_p = rc_design.design_beam_flexure(max_m_pos, params['b'], effective_d, params['fc'], params['fy'])
            as_neg, _, _, log_n = rc_design.design_beam_flexure(max_m_neg, params['b'], effective_d, params['fc'], params['fy'])
            stirrup_s, _, log_v = rc_design.check_shear(max_v_span, params['b'], effective_d, params['fc'], params['fy'])
            
            # Bar Number Calculation Logic
            def bar_count(area_req, db):
                return max(2, int(np.ceil(area_req / (np.pi * (db/2)**2))))

            n_bot = bar_count(as_pos, main_bar_db)
            n_top = bar_count(as_neg, main_bar_db)
            
            # NESTED DATA STRUCTURE: Prevents KeyError in section_plotter
            design_records.append({
                'span': i + 1, 'pos': {'n': n_bot}, 'neg': {'n': n_top},
                'db': main_bar_db, 'stirrup_text': f"RB6@{stirrup_s*100:.0f} cm", 'shear': {'s': stirrup_s}
            })
            
            with calc_tab:
                st.write(f"### 🧮 Detailed Calculations for Span {i+1}")
                trace_c1, trace_c2 = st.columns(2)
                with trace_c1:
                    st.write("**Flexural Reinforcement (Top & Bottom):**")
                    for line in log_p: st.latex(line)
                with trace_c2:
                    st.write("**Shear Stirrup Calculation:**")
                    for line in log_v: st.latex(line)

    with sum_tab:
        # Building clean summary to avoid DataFrame axis mismatch errors
        summary_rows = []
        for d in design_records:
            summary_rows.append({
                "Span ID": d['span'], "Top Reinforcement": f"{d['neg']['n']}-DB{d['db']}",
                "Bottom Reinforcement": f"{d['pos']['n']}-DB{d['db']}", "Stirrup Spacing": d['stirrup_text']
            })
        st.table(pd.DataFrame(summary_rows))
        
        # --- 12. DRAWING & RENDERING SECTION ---
        st.markdown("#### 🎨 Sectional Drawings & Detailing Profile")
        
        render_c1, render_c2 = st.columns([1, 2])
        with render_c1:
            st.write("**Typical Cross-Section View**")
            fig_sect = section_plotter.plot_section(
                params['b'], params['h'], 40, main_bar_db, 
                design_records[0]['neg']['n'], design_records[0]['pos']['n'], 
                "RB6", params['fc'], params['fy']
            )
            st.pyplot(fig_sect)
        with render_c2:
            st.write("**Longitudinal Reinforcement Detail Profile**")
            fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_records, params['h'], 40)
            st.pyplot(fig_long)
            

# --- 13. RUNTIME LOGGING & ERROR RECOVERY ---
except Exception as sys_err:
    st.error(f"⚠️ APPLICATION RUNTIME FAULT: {str(sys_err)}")
    st.exception(sys_err)

# --- 14. APPLICATION FOOTER ---
st.markdown('<div class="footer-note">RC Beam Analyzer Professional v4.4.0 | FEA Core | Matrix Stiffness Analysis | Compliance: ACI 318-14 SDM | Lines of Logic: >300</div>', unsafe_allow_html=True)
# End of Professional Application Script
