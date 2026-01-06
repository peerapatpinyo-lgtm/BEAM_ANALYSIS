import streamlit as st
import pandas as pd
import numpy as np

# Import Local Modules
import input_handler
import solver
import design_view
import rc_design
import section_plotter

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Pro Beam Studio",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better look
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
    .reportview-container .main .block-container{ max-width: 1000px; }
    h1, h2, h3 { color: #2C3E50; }
    .highlight { background-color: #F4F6F7; padding: 15px; border-radius: 10px; border-left: 5px solid #3498DB; }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ Professional RC Beam Studio")
st.markdown("---")

# ==========================================
# 2. SIDEBAR & INPUTS (Connect to input_handler)
# ==========================================
# เรียกใช้ input_handler เพื่อจัดการ Sidebar และ Load File
params = input_handler.render_sidebar()

# เพื่อความสมบูรณ์ เราต้องกำหนดค่า fc และ fy ที่ input_handler อาจจะไม่ได้ส่งออกมาโดยตรง
# (คำนวณย้อนกลับจาก E หรือกำหนดค่า Default)
if 'fc' not in params:
    # Estimate fc from E approx (E = 4700 sqrt(fc)) -> fc = (E/4700)^2
    est_fc = (params['E'] / 4700)**2
    # Clamp value to standard range
    if est_fc < 15: est_fc = 20
    if est_fc > 50: est_fc = 35
    params['fc'] = float(int(est_fc))

# เพิ่ม Input สำหรับ Strength เหล็กเสริม (fy) ใน Sidebar เพิ่มเติม
with st.sidebar:
    st.markdown("---")
    st.subheader("4. Rebar Strength")
    params['fy'] = st.number_input("Yield Strength (fy)", value=400.0, step=10.0, format="%.1f")
    st.caption(f"Using fc' ≈ {params['fc']:.1f} MPa")
    
    # Concrete Cover
    params['cover'] = st.slider("Cover (mm)", 20, 75, 25)
    # Rebar Diameter Main
    params['db_main'] = st.selectbox("Main Bar DB (mm)", [12, 16, 20, 25, 28, 32], index=1)

# Render Model Geometry Inputs
n_spans, spans, sup_df, stable = input_handler.render_model_inputs(params)

# Render Loads Input
loads_df = input_handler.render_loads(n_spans, spans, params, sup_df)

# ==========================================
# 3. MAIN EXECUTION & ANALYSIS
# ==========================================

# Check Stability before enabling run button
if not stable:
    st.error("⚠️ Structure is Unstable! Please check supports (Must have at least 2 supports or 1 Fixed).")
else:
    col_run, col_status = st.columns([1, 4])
    with col_run:
        run_btn = st.button("🚀 Run Analysis", type="primary")
    
    if run_btn:
        with st.spinner("Solving Finite Element Model..."):
            # A. PREPARE DATA FOR SOLVER
            # Convert DataFrames to List of Dicts for Solver
            sup_list = sup_df.to_dict('records') if not sup_df.empty else []
            load_list = loads_df.to_dict('records') if loads_df is not None else []
            
            # B. CALL SOLVER
            # Load Factors
            gamma_D = params['gamma_dead']
            gamma_L = params['gamma_live']
            
            # Apply Load Factors to Solver Loads
            factored_loads = []
            for l in load_list:
                f = gamma_D if l['case'] == 'DL' else gamma_L
                new_l = l.copy()
                new_l['mag'] = l['mag'] * f
                factored_loads.append(new_l)

            # Instantiate Solver
            beam_solver = solver.BeamSolver(
                spans=spans,
                supports_input=sup_list,
                loads_input=factored_loads, # Send Factored Loads
                E=params['E'],
                b=params['b'],
                h=params['h'],
                I_custom=params['I']
            )
            
            # Solve!
            df_res, reac, eq_check = beam_solver.solve()
            
            if df_res.empty:
                st.error("Solver returned no result. Please check inputs.")
                st.stop()

        # ==========================================
        # 4. DISPLAY RESULTS
        # ==========================================
        st.success("✅ Analysis Complete!")
        
        # --- TAB 1: DIAGRAMS ---
        tab1, tab2 = st.tabs(["📊 Analysis Diagrams", "🏗️ RC Design & Detailing"])
        
        with tab1:
            st.subheader("Shear, Moment & Deflection Diagrams")
            # Call design_view to draw interactive plots
            fig_diagram = design_view.draw_interactive_diagrams(df_res, reac, spans, sup_df, load_list)
            st.plotly_chart(fig_diagram, use_container_width=True)
            
            # Show Reaction Table
            st.markdown("#### Support Reactions (Factored)")
            reac_data = [{"Node": k, "Reaction (kN)": v/1000.0} for k, v in reac.items()]
            st.dataframe(pd.DataFrame(reac_data).set_index("Node").T)

        # --- TAB 2: RC DESIGN ---
        with tab2:
            st.subheader("Reinforced Concrete Design (ACI 318)")
            
            # Loop through each span to design
            cum_dist = [0] + list(np.cumsum(spans))
            
            for i in range(n_spans):
                st.markdown(f"### Span {i+1} (L = {spans[i]} m)")
                
                # 1. Extract Forces for this span
                start_x = cum_dist[i]
                end_x = cum_dist[i+1]
                
                # Filter results for this span
                mask = (df_res['x'] >= start_x) & (df_res['x'] <= end_x)
                span_res = df_res[mask]
                
                # Get Critical Moments & Shear
                # Convert N-m to kNm and N to kN
                m_max_pos = span_res['moment'].max() / 1000.0
                m_max_neg = span_res['moment'].min() / 1000.0
                v_max_abs = span_res['shear'].abs().max() / 1000.0
                
                if m_max_pos < 0: m_max_pos = 0 # Ignore if no positive moment
                
                # 2. Call RC Design Logic
                design_res = rc_design.design_span_expert(
                    m_pos=m_max_pos,
                    m_neg=m_max_neg,
                    v_u=v_max_abs,
                    b=params['b'],
                    h=params['h'],
                    fc=params['fc'],
                    fy=params['fy'],
                    cover=params['cover'],
                    db=params['db_main']
                )
                
                # 3. Layout: Graphics vs Data
                c1, c2 = st.columns([1.5, 1])
                
                with c1:
                    # Draw Cross Section
                    # We create a composite plot logic or just plot the critical section (e.g., Midspan)
                    # For simplicity, let's plot the Positive Moment Section (Midspan-ish)
                    st.markdown("**SECTION DETAIL (Midspan / Positive Region)**")
                    fig_sec = section_plotter.plot_section(
                        b=params['b'],
                        h=params['h'],
                        cover_mm=params['cover'],
                        db_mm=params['db_main'],
                        n_top=design_res['pos']['n'] if design_res['pos']['n'] > 0 else 2, # Show dummy top bars (hangers) if 0
                        n_bot=design_res['pos']['n'],
                        stirrup_info=design_res['shear_stirrups'], # <--- FIXED: using string directly
                        fc=params['fc'],
                        fy=params['fy']
                    )
                    st.pyplot(fig_sec)

                with c2:
                    st.markdown("**DESIGN CHECK LIST**")
                    
                    # Create Summary Dataframe
                    res_summary = {
                        "Zone": ["Positive (Mid)", "Negative (Sup)"],
                        "Mu (kNm)": [f"{m_max_pos:.2f}", f"{m_max_neg:.2f}"],
                        "Rebar Qty": [
                            f"{design_res['pos']['n']} - DB{params['db_main']}",
                            f"{design_res['neg']['n']} - DB{params['db_main']}"
                        ],
                        "Note": [design_res['pos']['note'], design_res['neg']['note']],
                        "Capacity (kNm)": [
                            f"{design_res['pos']['capacity']:.2f}",
                            f"{design_res['neg']['capacity']:.2f}"
                        ]
                    }
                    st.dataframe(pd.DataFrame(res_summary))
                    
                    st.info(f"🛡️ Shear Design (Vu = {v_max_abs:.2f} kN): **{design_res['shear_stirrups']}**")

                # 4. Capacity Plot (Interaction)
                with st.expander(f"📉 View Capacity vs Demand Diagram (Span {i+1})"):
                    # Get Capacity Limits
                    phi_mn_pos = design_res['pos']['capacity']
                    phi_mn_neg = design_res['neg']['capacity']
                    
                    # Create Local Span Coordinate for plotting
                    df_span_local = span_res.copy()
                    df_span_local['x'] = df_span_local['x'] - start_x
                    
                    fig_cap = design_view.plot_capacity_vs_demand(
                        df_span_local, 
                        phi_mn_pos, 
                        -abs(phi_mn_neg) # Ensure negative is plotted negatively
                    )
                    st.plotly_chart(fig_cap, use_container_width=True)
                
                st.markdown("---")
