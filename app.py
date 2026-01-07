import streamlit as st
import pandas as pd

# Import modules
# ตรวจสอบว่าไฟล์เหล่านี้อยู่ในโฟลเดอร์เดียวกับ app.py
try:
    import input_handler
    import file_manager
    import solver
    import rc_design
    import design_view
    import section_plotter
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# --- 1. Page Config ---
st.set_page_config(
    page_title="Pro Beam Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        font-weight: bold;
        background-color: #2C3E50;
        color: white;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Session State Init ---
if 'project_data' not in st.session_state:
    st.session_state.project_data = None

# --- 3. Sidebar & Inputs ---
params = input_handler.render_sidebar()

st.title("🏗️ Professional RC Beam Designer")
st.caption("Finite Element Analysis & ACI 318 Design | Version 1.0")

# File Management (Top Bar)
col_file1, col_file2 = st.columns([1, 4])
with col_file1:
    uploaded_file = st.file_uploader("📂 Load Project (.json)", type=["json"])
    if uploaded_file:
        loaded = file_manager.load_data(uploaded_file)
        if loaded:
            st.session_state.project_data = loaded
            st.success("Loaded!")

# Inputs
n_spans, spans, sup_df, stable = input_handler.render_model_inputs(params)
loads_df = input_handler.render_loads(n_spans, spans, params, sup_df)

# Save Button Logic
if loads_df is not None and not loads_df.empty:
    json_str = file_manager.export_data(params, spans, sup_df, loads_df)
    st.sidebar.download_button(
        label="💾 Save Project",
        data=json_str,
        file_name="my_beam_project.json",
        mime="application/json"
    )

st.markdown("---")

# --- 4. Main Process (Solver & Design) ---
if st.button("🚀 Run Analysis & Design", type="primary"):
    if not stable:
        st.error("🚨 Structure is Unstable! Please check supports (Need at least 2 supports or 1 Fixed).")
    else:
        # A. Analysis (Solver)
        st.info("Computing Finite Element Analysis...")
        
        # Prepare Data for Solver
        # Convert Dataframes to list of dicts for the solver class
        sup_list = sup_df.to_dict('records') if not sup_df.empty else []
        load_list = loads_df.to_dict('records') if (loads_df is not None and not loads_df.empty) else []
        
        beam_solver = solver.BeamSolver(spans, sup_list, load_list, params['E'], params['b'], params['h'], params['I'])
        res_df, reactions, status = beam_solver.solve()
        
        if "error" in status:
            st.error(f"Analysis Failed: {status['error']}")
        else:
            # B. RC Design
            st.success("Analysis Complete! Running Concrete Design...")
            
            design_res = []
            # Loop check each span for max moment
            # (Simplified: Extract max pos/neg moment from analysis results per span)
            
            # Create segments for design
            cum_dist = [0] + list(pd.Series(spans).cumsum())
            
            for i in range(len(spans)):
                x_start = cum_dist[i]
                x_end = cum_dist[i+1]
                
                # Filter results for this span
                span_res = res_df[(res_df['x'] >= x_start) & (res_df['x'] <= x_end)]
                
                if span_res.empty:
                    m_pos, m_neg = 0, 0
                    v_max = 0
                else:
                    # Factored Loads (User defined factors in sidebar)
                    # Note: In real practice, we run separate Load Combinations.
                    # Here we assume the user input loads are already Working Loads, 
                    # and we apply a simplified factor for Ultimate Design.
                    # Let's take an average factor of 1.4 for simplicity in this demo, 
                    # or use the inputs from sidebar if applied to specific cases.
                    
                    # For this demo: assume Analysis Results are "Service", scale to Ultimate
                    # (Or simpler: Just take the max M and V and design)
                    factor = 1.4 # Simplified U_factor
                    
                    m_max = span_res['moment'].max()
                    m_min = span_res['moment'].min() # Negative moment
                    
                    # Convert to Design Moments (Ultimate)
                    Mu_pos = max(0, m_max) * factor
                    Mu_neg = abs(min(0, m_min)) * factor
                    
                    vu_val = span_res['shear'].abs().max() * factor
                    
                    # Call RC Design Module
                    des_span = rc_design.design_span_expert(
                        Mu_pos, Mu_neg, vu_val, 
                        params['b'], params['h'], 
                        24, 400, # fc, fy (Simplified, can be inputs)
                        40, 16 # Cover, db main
                    )
                    
                    # Add span info
                    des_span['span_id'] = i
                    des_span['db'] = 16
                    design_res.append(des_span)
            
            # --- 5. Visualization Results ---
            
            # Tab 1: Analysis Diagrams
            t1, t2, t3 = st.tabs(["📊 Analysis Results", "🏗️ Design & Detailing", "📝 Calculation Report"])
            
            with t1:
                st.subheader("Shear & Moment Diagrams")
                fig_ana = design_view.plot_analysis_results(res_df, spans)
                st.plotly_chart(fig_ana, use_container_width=True)
                
                # Show Reactions
                st.write("Reaction Forces (kN/kNm):", reactions)
                
            with t2:
                st.subheader("Reinforcement Detailing")
                
                # Longitudinal View
                fig_long = section_plotter.plot_longitudinal_section(spans, sup_df, design_res, params['h'], 40)
                st.pyplot(fig_long)
                
                st.divider()
                
                # Cross Sections per Span
                cols = st.columns(len(spans))
                for i, c in enumerate(cols):
                    d = design_res[i]
                    with c:
                        st.markdown(f"**Span {i+1}**")
                        fig_sec = section_plotter.plot_section(
                            params['b'], params['h'], 40, 16,
                            d['neg']['n'], d['pos']['n'], 
                            d['shear_stirrups'], 24, 400
                        )
                        st.pyplot(fig_sec)
                        
                        # Status Tags
                        if d['shear_status'] == 'Fail':
                            st.error("Shear: Fail")
                        else:
                            st.success(f"Shear: {d['shear_status']}")
                            
            with t3:
                st.subheader("Design Summary Table")
                # Format a nice table for report
                report_data = []
                for idx, res in enumerate(design_res):
                    report_data.append({
                        "Span": idx+1,
                        "Top Bars": f"{res['neg']['n']}-DB16",
                        "Bot Bars": f"{res['pos']['n']}-DB16",
                        "Stirrups": res['shear_stirrups'],
                        "Capacity + (kNm)": f"{res['pos']['capacity']:.2f}",
                        "Capacity - (kNm)": f"{res['neg']['capacity']:.2f}",
                        "Note": res['pos']['note']
                    })
                st.dataframe(pd.DataFrame(report_data))
