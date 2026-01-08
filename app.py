import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# =====================================================================
# 1. MODULE IMPORTS & INITIALIZATION
# =====================================================================
import input_handler
import solver
import rc_design
import design_view
import section_plotter

# Setup Page Configuration for Wide Layout and Professional Look
st.set_page_config(
    page_title="Ultimate RC Beam Analysis & Engineering Report",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Table and Report Sections
st.markdown("""
    <style>
    .report-header { background-color: #1e3a8a; color: white; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem; }
    .metric-card { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 0.5rem; padding: 1rem; text-align: center; }
    .formula-box { background-color: #f1f5f9; border-left: 5px solid #3b82f6; padding: 15px; font-family: 'Courier New', monospace; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================================
# 2. APPLICATION HEADER & METADATA
# =====================================================================
st.markdown("""
    <div class="report-header">
        <h1>🏗️ Structural Beam Analysis & RC Design System</h1>
        <p>Integrated FEA Solver for Continuous Beams | Load Combination | RC Detailing</p>
    </div>
    """, unsafe_allow_html=True)

# Display Current Project Info
col_meta1, col_meta2, col_meta3 = st.columns(3)
with col_meta1:
    st.info(f"📅 **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
with col_meta2:
    st.info("💻 **Solver Engine:** Timoshenko Beam Theory")
with col_meta3:
    st.info("📐 **Design Code:** ACI 318-14 / Strength Design")

# =====================================================================
# 3. SIDEBAR DATA COLLECTION (INPUT_HANDLER)
# =====================================================================
# ดึงค่าพารามิเตอร์และโหลดทั้งหมดมาจาก Module input_handler
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

# =====================================================================
# 4. STABILITY CHECK & ERROR HANDLING
# =====================================================================
if not stable:
    st.error("🚨 **CRITICAL ERROR: SYSTEM UNSTABLE**")
    st.warning("โครงสร้างไม่มีเสถียรภาพ (Instability) เนื่องจากจุดรองรับไม่เพียงพอ กรุณาตรวจสอบ:")
    st.markdown("- ตรวจสอบว่ามี Reaction อย่างน้อย 3 จุด (เช่น Pin 1 ตัว และ Roller 1 ตัวเป็นอย่างน้อย)")
    st.markdown("- หากเป็นคานปลายยื่น (Cantilever) ต้องมีจุดรองรับแบบ 'Fixed' ที่จุดเริ่มต้น")
    st.stop()

# =====================================================================
# 5. LOAD COMBINATION & ANALYSIS SETTINGS
# =====================================================================
st.subheader("⚙️ 1. Analysis Parameters & Load Factors")
st.markdown("ระบุตัวคูณน้ำหนักบรรทุก (Load Factors) เพื่อใช้ในวิธี Strength Design Method (SDM)")

with st.container():
    c_fac1, c_fac2, c_fac3 = st.columns([1, 1, 2])
    
    with c_fac1:
        f_dl = st.number_input("Dead Load Factor ($f_{DL}$)", value=1.4, step=0.1, key="f_dl_input")
        st.caption("Default ACI: 1.4")
        
    with c_fac2:
        f_ll = st.number_input("Live Load Factor ($f_{LL}$)", value=1.7, step=0.1, key="f_ll_input")
        st.caption("Default ACI: 1.7")
        
    with c_fac3:
        analysis_mode = st.radio("Calculation Basis:", 
                                ["Total Factored Load ($W_u = 1.4DL + 1.7LL$)", 
                                 "Service Load (DL + LL)"], horizontal=True)

# =====================================================================
# 6. DETAILED LOAD CALCULATION & TRACEABILITY
# =====================================================================
st.markdown("---")
st.subheader("🧮 2. Detailed Load Summation & Verification")

try:
    final_loads_for_solver = []
    load_summary_report = []
    
    # 6.1 SELF-WEIGHT CALCULATION (Automatic for all spans)
    st.markdown("#### A. Structural Self-Weight (Concrete Density = 24.0 kN/m³)")
    
    for i in range(n_spans):
        # Calculation Logic: Area * Density * DL Factor
        base_sw_kN_m = params['b'] * params['h'] * 24.0
        factored_sw = base_sw_kN_m * f_dl
        span_length = spans[i]
        total_sw_span = factored_sw * span_length
        
        # Add to Solver Input (Newton units)
        final_loads_for_solver.append({
            'span_index': i, 'type': 'U', 'mag': factored_sw * 1000.0,
            'd_start': 0.0, 'dist': span_length, 'desc': 'Self-Weight'
        })
        
        # Add to Report Trace
        load_summary_report.append({
            "Span": i + 1, "Source": "Concrete Self-Weight", "Case": "DL",
            "Intensity": f"{base_sw_kN_m:.2f} kN/m", "Factor": f"x{f_dl}",
            "Design Value": f"{factored_sw:.2f} kN/m", "Total Force (kN)": f"{total_sw_span:.2f}"
        })

    # 6.2 USER-DEFINED LOADS (DL/LL Separation Logic)
    if not loads_df.empty:
        st.markdown("#### B. External Applied Loads (User Input)")
        for _, row in loads_df.iterrows():
            current_factor = f_dl if row['case'] == "DL" else f_ll
            raw_mag = float(row['mag'])
            factored_mag = raw_mag * current_factor
            
            # Resultant Force Calculation
            if row['type'] == 'P':
                net_resultant = factored_mag
                unit_label = "kN"
            else:
                net_resultant = factored_mag * row['dist']
                unit_label = "kN/m"

            load_summary_report.append({
                "Span": row['span_index'] + 1, "Source": f"User Added ({row['case']})", 
                "Case": row['case'], "Intensity": f"{raw_mag:.2f} {unit_label}", 
                "Factor": f"x{current_factor}", "Design Value": f"{factored_mag:.2f} {unit_label}", 
                "Total Force (kN)": f"{net_resultant:.2f}"
            })

            # Format for Solver
            final_loads_for_solver.append({
                'span_index': int(row['span_index']), 'type': row['type'],
                'mag': factored_mag * 1000.0, 'd_start': row['d_start'],
                'dist': row['dist'], 'desc': f"User {row['case']}"
            })

    # Display Load Summation Table
    calc_df = pd.DataFrame(load_summary_report)
    st.table(calc_df)
    
    # Checksum of Applied Loads
    total_w_kN = calc_df["Total Force (kN)"].astype(float).sum()
    st.markdown(f"""
        <div class="formula-box">
            <b>Total Net Factored Load Applied ($\Sigma W_u$):</b> {total_w_kN:.3f} kN <br>
            <i>Note: This is the total vertical force that must be balanced by support reactions.</i>
        </div>
    """, unsafe_allow_html=True)

    # =====================================================================
    # 7. STRUCTURAL ANALYSIS EXECUTION (SOLVER MODULE)
    # =====================================================================
    st.markdown("---")
    st.subheader("📊 3. Structural Analysis Results")
    
    # Run Finite Element Solver
    solver_load_df = pd.DataFrame(final_loads_for_solver)
    x_points, moment_vals, shear_vals, defl_vals, reactions = solver.solve_beam(
        spans, sup_df, solver_load_df, params
    )
    
    # Assemble Analysis DataFrame
    analysis_res_df = pd.DataFrame({
        'x': x_points, 'moment': moment_vals, 'shear': shear_vals, 'deflection': defl_vals * 1000.0
    })

    # Visualization
    st.plotly_chart(design_view.plot_analysis_results(analysis_res_df, spans, sup_df, solver_load_df, reactions), use_container_width=True)

    # =====================================================================
    # 8. EQUILIBRIUM & REACTION VERIFICATION
    # =====================================================================
    with st.expander("⚖️ Static Equilibrium Verification", expanded=True):
        total_reaction_kN = sum(reactions.values()) / 1000.0
        error_diff = abs(total_w_kN - total_reaction_kN)
        
        v_col1, v_col2, v_col3 = st.columns(3)
        v_col1.metric("Sum Applied Loads (ΣW)", f"{total_w_kN:.3f} kN")
        v_col2.metric("Sum Reaction Forces (ΣR)", f"{total_reaction_kN:.3f} kN")
        
        if error_diff < 0.01:
            v_col3.success(f"Equilibrium: OK\nDiff: {error_diff:.6f}")
        else:
            v_col3.error(f"Equilibrium: FAIL\nDiff: {error_diff:.4f}")

    # =====================================================================
    # 9. RC REINFORCEMENT DESIGN (ACI 318-14)
    # =====================================================================
    st.markdown("---")
    st.subheader("🧱 4. Reinforced Concrete Design & Detailing")
    
    tab_summary, tab_details = st.tabs(["📌 Design Summary", "📝 Step-by-Step Calculations"])
    
    design_data_summary = []
    main_bar_db = 16 # Default main reinforcement size
    offsets = [0] + list(np.cumsum(spans))

    for idx in range(n_spans):
        # Filter analysis data for the current span
        span_mask = (analysis_res_df['x'] >= offsets[idx] - 1e-6) & (analysis_res_df['x'] <= offsets[idx+1] + 1e-6)
        span_results = analysis_res_df[span_mask]
        
        if not span_results.empty:
            mu_max_pos = span_results['moment'].max() / 1000.0
            mu_max_neg = abs(span_results['moment'].min()) / 1000.0
            vu_max = span_results['shear'].abs().max() / 1000.0
            d_effective = params['h'] - 0.05
            
            # Flexure Design
            as_pos, rho_p, _, steps_p = rc_design.design_beam_flexure(mu_max_pos, params['b'], d_effective, params['fc'], params['fy'])
            as_neg, rho_n, _, steps_n = rc_design.design_beam_flexure(mu_max_neg, params['b'], d_effective, params['fc'], params['fy'])
            
            # Shear Design
            spacing, stirrup_type, steps_v = rc_design.check_shear(vu_max, params['b'], d_effective, params['fc'], params['fy'])
            
            # Calculate Number of Bars
            bar_area = np.pi * (main_bar_db/2)**2
            n_pos = max(2, int(np.ceil(as_pos / bar_area)))
            n_neg = max(2, int(np.ceil(as_neg / bar_area)))
            
            design_data_summary.append({
                'span': idx + 1, 'mu_pos': mu_max_pos, 'mu_neg': mu_max_neg,
                'n_pos': n_pos, 'n_neg': n_neg, 'stirrup': f"{stirrup_type}@{spacing*100:.0f} cm"
            })
            
            with tab_details:
                st.markdown(f"#### Span {idx+1} Detailed Design")
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.write("**Positive Moment Steel (Bottom):**")
                    for s in steps_p: st.latex(s)
                with sc2:
                    st.write("**Shear Stirrup Calculation:**")
                    for s in steps_v: st.latex(s)
    
    with tab_summary:
        st.table(pd.DataFrame(design_data_summary))
        
        # 10. DETAILING PLOTS
        st.markdown("### 🎨 Engineering Drawings")
        det_col1, det_col2 = st.columns([1, 2])
        with det_col1:
            st.write("**Cross Section View (Typical Span 1)**")
            fig_sec = section_plotter.plot_section(params['b'], params['h'], 40, main_bar_db, 
                                                  design_data_summary[0]['n_neg'], 
                                                  design_data_summary[0]['n_pos'], 
                                                  "RB6", params['fc'], params['fy'])
            st.pyplot(fig_sec)
        with det_col2:
            st.write("**Longitudinal Reinforcement Detail**")
            fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_data_summary, params['h'], 40)
            st.pyplot(fig_long)

# =====================================================================
# 10. SYSTEM LOGS & ERROR CATCHING
# =====================================================================
except Exception as error:
    st.error(f"⚠️ **Application Runtime Error:** {str(error)}")
    st.exception(error)

st.markdown("---")
st.caption("© 2024 BeamDesign Professional Edition. Optimized for ACI Standard Calculations.")

# END OF APP.PY (Approximately 250+ Lines maintained for completeness)
