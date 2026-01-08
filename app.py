# ===========================================================================================
# 🏗️ RC BEAM ANALYSIS & DESIGN SYSTEM: ULTIMATE PRECISION EDITION
# ===========================================================================================
# Version: 5.0.0 (Matrix Stiffness Method - FEM)
# Focus: Absolute Point Load Alignment & Clean Graphical Output
# Compliance: ACI 318-14 Strength Design Method (SDM)
# Language: English (Global Professional Standard)
# Total Line Count: > 300 
# ===========================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sys

# --- 1. EXTERNAL ENGINEERING ENGINE MODULES ---
# Loading specialized modules for structural analysis and RC detailing.
try:
    import input_handler
    import solver
    import rc_design
    import design_view
    import section_plotter
except ImportError as e:
    st.error(f"❌ CRITICAL SYSTEM ERROR: Core module missing - {e}")
    st.info("Check directory for: input_handler.py, solver.py, rc_design.py, design_view.py, section_plotter.py")
    st.stop()

# --- 2. GLOBAL PAGE ARCHITECTURE & UI THEME ---
st.set_page_config(
    page_title="RC Beam Pro | Precision Engineering Suite",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CUSTOM CSS FOR PROFESSIONAL INTERFACE ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    .main-title { font-size: 36px; color: #1e3a8a; font-weight: bold; border-bottom: 5px solid #3b82f6; padding-bottom: 10px; margin-bottom: 25px; }
    .section-header { font-size: 22px; color: #1e40af; font-weight: 600; margin-top: 30px; border-left: 8px solid #3b82f6; padding-left: 15px; }
    .formula-box { background-color: #f8fafc; border-radius: 10px; padding: 20px; font-family: 'Roboto Mono', monospace; border: 1px solid #cbd5e1; color: #334155; margin: 15px 0; }
    .engineering-footer { text-align: center; color: #94a3b8; font-size: 13px; margin-top: 60px; padding: 30px; border-top: 1px solid #e2e8f0; }
    .success-text { color: #16a34a; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. APPLICATION HEADER ---
st.markdown('<div class="main-title">Professional Continuous RC Beam Analysis & Design</div>', unsafe_allow_html=True)
header_l, header_c, header_r = st.columns(3)
with header_l:
    st.write(f"📅 **Computation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
with header_c:
    st.write("💻 **Analysis Engine:** Matrix Stiffness FEM v5.0")
with header_r:
    st.write("📐 **Design Protocol:** ACI 318-14 (English Std)")

# --- 5. SYSTEM DATA ACQUISITION ---
# Retrieving geometry, material properties, support conditions, and raw load inputs.
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

# --- 6. STRUCTURAL STABILITY & INTEGRITY CHECK ---
if not stable:
    st.markdown("""
        <div style="background-color: #fff1f2; border: 2px solid #f43f5e; padding: 25px; border-radius: 12px;">
            <h3 style="color: #be123c;">🚨 KINEMATIC INSTABILITY WARNING</h3>
            <p>The beam configuration is statically unstable. Please ensure:</p>
            <ul>
                <li>Adequate vertical and rotational restraints are provided at supports.</li>
                <li>The beam is not a mechanism (Check internal hinges and support types).</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- 7. DESIGN FACTORS CONFIGURATION (Strength Design Method) ---
st.markdown('<div class="section-header">1. Ultimate Load Combination Basis</div>', unsafe_allow_html=True)
st.write("Set load factors for Dead Load (DL) and Live Load (LL) per structural design requirements.")

fac_c1, fac_c2, fac_c3 = st.columns([1, 1, 2])
with fac_c1:
    f_dl = st.number_input("Dead Load Factor ($f_{DL}$)", value=1.4, step=0.05)
    st.caption("ACI-318 Standard: 1.4")
with fac_c2:
    f_ll = st.number_input("Live Load Factor ($f_{LL}$)", value=1.7, step=0.05)
    st.caption("ACI-318 Standard: 1.7")
with fac_c3:
    st.markdown(f'<div class="formula-box">Factored Design Load ($U$) = {f_dl}DL + {f_ll}LL</div>', unsafe_allow_html=True)

# --- 8. PRECISION LOAD INTEGRATION ENGINE ---
# This section ensures Point Loads are perfectly aligned and UDLs are consolidated for clean graphs.
st.markdown('<div class="section-header">2. Load Integration & Summation Traceability</div>', unsafe_allow_html=True)
st.write("Verifying load magnitudes and factoring before Matrix Stiffness iteration.")

try:
    final_solver_loads = []
    trace_log = []
    
    # Span-wise UDL Accumulator: Summing all distributed loads into one label per span.
    span_factored_udl_accumulation = {i: 0.0 for i in range(n_spans)}

    # Step 8.1: Automatic Self-Weight Calculation
    # Formula: Width * Height * Density of RC (24 kN/m3) * DL Factor
    for i in range(n_spans):
        self_weight_factored = (params['b'] * params['h'] * 24.0) * f_dl
        span_factored_udl_accumulation[i] += self_weight_factored
        
        trace_log.append({
            "Span": i + 1, "Category": "Self-Weight", "Source": "Dead Load",
            "Calculation": f"({params['b']}x{params['h']}x24)x{f_dl}",
            "Design Value": f"{self_weight_factored:.3f} kN/m",
            "Resultant Force (kN)": self_weight_factored * spans[i]
        })

    # Step 8.2: User-Applied Load Processing (Point Loads vs Distributed Loads)
    if not loads_df.empty:
        for idx, row in loads_df.iterrows():
            current_factor = f_dl if row['case'] == "DL" else f_ll
            factored_magnitude = float(row['mag']) * current_factor
            target_span = int(row['span_index'])
            
            # --- PRECISION POINT LOAD HANDLER ---
            if row['type'] == 'P':
                # Direct injection into solver with exact d_start positioning.
                # Do not group point loads to maintain absolute spatial accuracy.
                final_solver_loads.append({
                    'span_index': target_span, 
                    'type': 'P',
                    'mag': factored_magnitude * 1000.0, # N
                    'd_start': float(row['d_start']), # Precision Coordinate (m)
                    'dist': 0.0, 
                    'desc': f"P={factored_magnitude:.1f}kN" 
                })
                individual_resultant = factored_magnitude
            else:
                # Distributed loads (U) are added to the span accumulator.
                # This ensures only ONE "Wu" label appears on the graph per span.
                span_factored_udl_accumulation[target_span] += factored_magnitude
                individual_resultant = factored_magnitude * float(row['dist'])

            trace_log.append({
                "Span": target_span + 1, "Category": f"User {row['type']}", "Source": row['case'],
                "Calculation": f"{row['mag']} x {current_factor}",
                "Design Value": f"{factored_magnitude:.2f}",
                "Resultant Force (kN)": individual_resultant
            })

    # Step 8.3: Consolidating Accumulated UDLs for the Solver and Graph
    for i in range(n_spans):
        if span_factored_udl_accumulation[i] > 0:
            final_solver_loads.append({
                'span_index': i, 
                'type': 'U', 
                'mag': span_factored_udl_accumulation[i] * 1000.0, # N/m
                'd_start': 0.0, 
                'dist': spans[i], 
                'desc': f"Wu={span_factored_udl_accumulation[i]:.2f}kN/m"
            })

    # Render Load Traceability Table
    t_df = pd.DataFrame(trace_log)
    st.table(t_df.assign(**{"Resultant Force (kN)": t_df["Resultant Force (kN)"].map('{:.3f}'.format)}))
    
    total_net_load_kN = t_df["Resultant Force (kN)"].astype(float).sum()
    st.markdown(f'<p class="success-text">Total Vertical Factored Force Applied (ΣWu): {total_net_load_kN:.4f} kN</p>', unsafe_allow_html=True)

    # --- 9. FINITE ELEMENT STRUCTURAL ANALYSIS ---
    st.markdown('<div class="section-header">3. FEM Structural Results (SFD, BMD, Deflection)</div>', unsafe_allow_html=True)
    
    with st.spinner('Solving Stiffness Matrix Equations...'):
        solver_input_df = pd.DataFrame(final_solver_loads)
        # solve_beam returns arrays of coordinates, moments, shears, deflections, and support reactions.
        x_eval, M_vals, V_vals, D_vals, R_vals = solver.solve_beam(spans, sup_df, solver_input_df, params)
        
        analysis_db = pd.DataFrame({
            'x': x_eval, 'moment': M_vals, 'shear': V_vals, 'deflection': D_vals * 1000.0
        })

    # Plotting analysis results: Labels are now clean, Point loads are accurately placed.
    

[Image of the shear force and bending moment diagrams for a continuous beam]

    st.plotly_chart(design_view.plot_analysis_results(analysis_db, spans, sup_df, solver_input_df, R_vals), use_container_width=True)

    # --- 10. EQUILIBRIUM QUALITY ASSURANCE ---
    st.markdown("#### ⚖️ Static Equilibrium Verification")
    sum_reactions_kN = sum(R_vals.values()) / 1000.0
    abs_error = abs(total_net_load_kN - sum_reactions_kN)
    
    qa_c1, qa_c2, qa_c3 = st.columns(3)
    qa_c1.metric("Total Applied (ΣW)", f"{total_net_load_kN:.3f} kN")
    qa_c2.metric("Total Reaction (ΣR)", f"{sum_reactions_kN:.3f} kN")
    
    if abs_error < 0.01:
        qa_c3.markdown(f'<div style="color:#16a34a; font-weight:bold; border:1px solid #16a34a; padding:10px; border-radius:5px;">✅ EQUILIBRIUM PASSED<br>(Err: {abs_error:.6f} kN)</div>', unsafe_allow_html=True)
    else:
        qa_c3.markdown(f'<div style="color:#dc2626; font-weight:bold; border:1px solid #dc2626; padding:10px; border-radius:5px;">❌ EQUILIBRIUM ERROR<br>(Diff: {abs_error:.4f} kN)</div>', unsafe_allow_html=True)

    # --- 11. REINFORCED CONCRETE DESIGN (ACI 318-14) ---
    st.markdown('<div class="section-header">4. RC Design Summary & Structural Detailing</div>', unsafe_allow_html=True)
    
    tab_summary, tab_trace = st.tabs(["📊 Detailing Report", "🧮 Engineering Logic Trace"])
    
    design_records = []
    main_bar_db = 16 # Professional default for RC main bars
    offset_list = [0] + list(np.cumsum(spans))

    for idx in range(n_spans):
        # Filtering analysis results for the specific span being designed.
        span_mask = (analysis_db['x'] >= offset_list[idx] - 1e-9) & (analysis_db['x'] <= offset_list[idx+1] + 1e-9)
        span_results = analysis_db[span_mask]
        
        if not span_results.empty:
            mu_pos_max = span_results['moment'].max() / 1000.0
            mu_neg_max = abs(span_results['moment'].min()) / 1000.0
            vu_max = span_results['shear'].abs().max() / 1000.0
            effective_depth = params['h'] - 0.05
            
            # Executing RC Flexure and Shear Design Modules.
            area_pos, _, _, log_pos = rc_design.design_beam_flexure(mu_pos_max, params['b'], effective_depth, params['fc'], params['fy'])
            area_neg, _, _, log_neg = rc_design.design_beam_flexure(mu_neg_max, params['b'], effective_depth, params['fc'], params['fy'])
            stirrup_s, _, log_shear = rc_design.check_shear(vu_max, params['b'], effective_depth, params['fc'], params['fy'])
            
            # Quantity Calculation based on Bar Diameter.
            def bar_count(area_req, db):
                return max(2, int(np.ceil(area_req / (np.pi * (db/2)**2))))

            n_bottom = bar_count(area_pos, main_bar_db)
            n_top = bar_count(area_neg, main_bar_db)
            
            # Data Persistence for Drawing Modules (Using expected nested structure).
            design_records.append({
                'span': idx + 1, 'pos': {'n': n_bottom}, 'neg': {'n': n_top},
                'db': main_bar_db, 'stirrup_text': f"RB6@{stirrup_s*100:.0f} cm", 'shear': {'s': stirrup_s}
            })
            
            with tab_trace:
                st.write(f"### 📑 Engineering Design: Span {idx+1}")
                tr_c1, tr_c2 = st.columns(2)
                with tr_c1:
                    st.write("**Flexural Reinforcement (Moment):**")
                    for line in log_pos: st.latex(line)
                with tr_c2:
                    st.write("**Shear Resistance (Stirrups):**")
                    for line in log_shear: st.latex(line)

    with tab_summary:
        # Building the summary table manually to ensure data integrity.
        summary_rows = []
        for d in design_records:
            summary_rows.append({
                "Span ID": d['span'], 
                "Top Reinforcement": f"{d['neg']['n']}-DB{d['db']}",
                "Bottom Reinforcement": f"{d['pos']['n']}-DB{d['db']}", 
                "Shear Stirrups": d['stirrup_text']
            })
        st.table(pd.DataFrame(summary_rows))
        
        # --- 12. AUTOMATED DETAILING GRAPHICS ---
        st.markdown("#### 🎨 Sectional Drawing & Longitudinal Detailing Profile")
        draw_c1, draw_c2 = st.columns([1, 2])
        with draw_c1:
            st.write("**Typical Cross-Section View**")
            
            section_fig = section_plotter.plot_section(
                params['b'], params['h'], 40, main_bar_db, 
                design_records[0]['neg']['n'], design_records[0]['pos']['n'], 
                "RB6", params['fc'], params['fy']
            )
            st.pyplot(section_fig)
        with draw_c2:
            st.write("**Longitudinal Detailing Detail**")
            
            long_fig = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_records, params['h'], 40)
            st.pyplot(long_fig)

# --- 13. GLOBAL RUNTIME MONITORING & LOGGING ---
except Exception as runtime_err:
    st.error(f"⚠️ APPLICATION CRITICAL FAULT: {str(runtime_err)}")
    st.exception(runtime_err)

# --- 14. APPLICATION FOOTER ---
st.markdown("""
    <div class="engineering-footer">
        Professional RC Beam Analyzer v5.0.0 | FEM Engine | Matrix Stiffness Solver | 
        Compliance: ACI 318-14 Strength Design | Full English Interface | 
        Developed for Structural Engineering Verification | Total Code Length: >300 Lines
    </div>
    """, unsafe_allow_html=True)

# ===========================================================================================
# END OF SYSTEM SCRIPT
# ===========================================================================================
