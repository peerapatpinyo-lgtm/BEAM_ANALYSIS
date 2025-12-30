import streamlit as st
import pandas as pd
import numpy as np
import solver
import input_handler
import design_view
from datetime import datetime

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Structural Beam Analysis", layout="wide", page_icon="🏗️")

# Custom CSS for Professional Look
st.markdown("""
<style>
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
        color: #212529;
    }
    
    /* Headings */
    h1 { color: #0D47A1; font-weight: 700; font-size: 2.2rem; border-bottom: 2px solid #0D47A1; padding-bottom: 10px; }
    h2 { color: #1565C0; font-weight: 600; font-size: 1.5rem; margin-top: 20px; }
    h3 { color: #424242; font-weight: 600; font-size: 1.2rem; }
    
    /* Metrics Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #0D47A1; }
    .metric-label { font-size: 0.9rem; color: #616161; text-transform: uppercase; letter-spacing: 0.5px; }
    
    /* Buttons */
    .stButton>button {
        background-color: #1565C0; color: white; border: none; border-radius: 4px;
        font-weight: 600; padding: 0.5rem 1rem;
    }
    .stButton>button:hover { background-color: #0D47A1; }
    
    /* Tables */
    div[data-testid="stDataFrame"] { border: 1px solid #E0E0E0; border-radius: 5px; }
    
    /* Success/Error Messages */
    .stAlert { border-radius: 4px; }
    
    /* Footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- 2. HELPER FUNCTIONS ---
def display_metric_card(col, label, value, unit, color="blue"):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color};">{value} <span style="font-size: 1rem; color: #757575;">{unit}</span></div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # --- HEADER SECTION ---
    with st.container():
        c1, c2 = st.columns([3, 1])
        with c1:
            st.title("🏗️ Professional Beam Analysis")
            st.markdown(f"**Date:** {datetime.now().strftime('%d %B %Y')} | **Standard:** ACI 318 / EIT")
        with c2:
            st.info("💡 **Engine:** Matrix Stiffness Method\n\n✅ Exact Theory Verification")

    st.markdown("---")

    # --- SIDEBAR: PROJECT INFO & INPUTS ---
    with st.sidebar:
        st.header("📝 Project Details")
        st.text_input("Project Name", placeholder="e.g., Residence 2 Storey")
        st.text_input("Engineer", placeholder="Senior Eng. Somchai")
        
        st.markdown("---")
        st.header("⚙️ Model Configuration")
        
        # Calls existing input_handler (Keep your existing logic)
        params = input_handler.render_sidebar()
    
    # --- MAIN WORKSPACE ---
    col_input, col_output = st.columns([1, 2], gap="large")
    
    # === LEFT COLUMN: MODELING ===
    with col_input:
        st.subheader("1. Geometry & Loading")
        with st.container(border=True):
            n, spans, sup_df, stable = input_handler.render_model_inputs(params)
            loads = input_handler.render_loads(n, spans, params)
            
            st.markdown("###")
            run_btn = st.button("🚀 Analyze Structure", type="primary", use_container_width=True, disabled=not stable)

            if not stable:
                st.error("⚠️ Structure is Unstable! Check supports.")

    # === RIGHT COLUMN: ANALYSIS REPORT ===
    with col_output:
        if run_btn or st.session_state.get('analysis_done'):
            try:
                if run_btn:
                    with st.spinner("Solving Stiffness Matrix..."):
                        # Call your NEW CORRECT SOLVER
                        engine = solver.BeamSolver(spans, sup_df, loads)
                        df_res, reactions = engine.solve()
                        
                        if df_res is None:
                            st.error("Calculation Error: Singular Matrix")
                            st.stop()
                            
                        # Save to Session
                        st.session_state['analysis_done'] = True
                        st.session_state['df_res'] = df_res
                        st.session_state['reactions'] = reactions
                        st.session_state['spans'] = spans
                        st.session_state['sup_df'] = sup_df
                        st.session_state['loads'] = loads

                # Retrieve Data
                df = st.session_state['df_res']
                reac = st.session_state['reactions']
                
                # --- SECTION 2: EXECUTIVE SUMMARY ---
                st.subheader("2. Governing Forces (Executive Summary)")
                
                # Calculate Max Values
                max_pos_m = df['moment'].max()
                max_neg_m = df['moment'].min()
                max_shear = df['shear'].abs().max()
                max_reac = np.max(np.abs(reac[::2])) # Take every 2nd element for Fy
                
                m1, m2, m3, m4 = st.columns(4)
                display_metric_card(m1, "Max +Moment", f"{max_pos_m:.2f}", "kg-m", "#2E7D32") # Green
                display_metric_card(m2, "Max -Moment", f"{max_neg_m:.2f}", "kg-m", "#C62828") # Red
                display_metric_card(m3, "Max Shear", f"{max_shear:.2f}", "kg", "#F57F17")    # Orange
                display_metric_card(m4, "Max Reaction", f"{max_reac:.2f}", "kg", "#1565C0")  # Blue
                
                st.markdown("###")

                # --- SECTION 3: TABS FOR DETAILS ---
                tab1, tab2, tab3 = st.tabs(["📈 Diagrams (SFD/BMD)", "📋 Reactions & Tables", "🏗️ RC Design"])
                
                with tab1:
                    st.markdown("#### Internal Force Diagrams")
                    # เรียกกราฟจาก design_view (อันเดิมที่คุณมี)
                    design_view.draw_diagrams(df, st.session_state['spans'], st.session_state['sup_df'], 
                                              st.session_state['loads'], params['u_force'], params['u_len'])
                
                with tab2:
                    st.markdown("#### Support Reactions")
                    # Format Reaction Data nicely
                    r_data = []
                    for i in range(len(st.session_state['spans']) + 1):
                        fy = reac[2*i]
                        mz = reac[2*i+1]
                        r_data.append({
                            "Node": i+1,
                            "Vertical Reaction (Ry) [kg]": f"{fy:.2f}",
                            "Moment Reaction (Mz) [kg-m]": f"{mz:.2f}"
                        })
                    st.dataframe(pd.DataFrame(r_data), use_container_width=True, hide_index=True)
                    
                    with st.expander("Show Detailed Calculation Points (Table)"):
                        st.dataframe(df)

                with tab3:
                    st.markdown("#### Reinforced Concrete Design")
                    st.info("👇 Define Section Properties per Span")
                    
                    # Design Inputs
                    n_spans = len(st.session_state['spans'])
                    d_cols = st.columns(n_spans)
                    span_props = []
                    
                    for i in range(n_spans):
                        with d_cols[i]:
                            st.markdown(f"**Span {i+1}**")
                            with st.container(border=True):
                                b = st.number_input(f"b (cm)", 25.0, step=5.0, key=f"b{i}")
                                h = st.number_input(f"h (cm)", 50.0, step=5.0, key=f"h{i}")
                                span_props.append({"b": b, "h": h, "cv": 3.0})
                    
                    st.markdown("---")
                    # Render Design Results
                    design_view.render_design_results(df, params, st.session_state['spans'], span_props, st.session_state['sup_df'])

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
        else:
            # Placeholder State
            st.info("👈 Please define the beam model on the left and click 'Analyze Structure'.")
            st.markdown("""
            <div style="text-align: center; color: #BDBDBD; margin-top: 50px;">
                <h3>Ready for Analysis</h3>
                <p>Waiting for engineer's input...</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
