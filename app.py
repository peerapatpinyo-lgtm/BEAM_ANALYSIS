import streamlit as st
import pandas as pd
import numpy as np
import solver
import input_handler
import design_view 
from datetime import datetime

st.set_page_config(page_title="Structural Pro", layout="wide", page_icon="🏗️")

# Clean UI CSS
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .stButton button { width: 100%; font-weight: bold; border-radius: 5px; }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    c1, c2 = st.columns([8, 2])
    with c1: st.title("🏗️ Beam Analysis & Design Professional")
    with c2: st.caption(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    st.divider()

    # Split Layout: Input (Left, 25%) | Output (Right, 75%)
    col_input, col_output = st.columns([25, 75], gap="large")

    with col_input:
        st.subheader("⚙️ Inputs")
        with st.expander("1. Geometry & Support", expanded=True):
            params = input_handler.render_sidebar()
            n, spans, sup_df, stable = input_handler.render_model_inputs(params)
        
        with st.expander("2. Loads", expanded=True):
            loads = input_handler.render_loads(n, spans, params)
            
        st.markdown("###")
        run_btn = st.button("RUN ANALYSIS", type="primary", disabled=not stable)
        if not stable: st.error("⚠️ Unstable Structure")

    with col_output:
        if run_btn or st.session_state.get('analysis_done'):
            if run_btn:
                try:
                    engine = solver.BeamSolver(spans, sup_df, loads)
                    df_res, reactions = engine.solve()
                    st.session_state.update({'analysis_done': True, 'df_res': df_res, 'reactions': reactions, 'spans': spans, 'sup_df': sup_df, 'loads': loads})
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            # Retrieve Data
            df = st.session_state['df_res']
            reac = st.session_state['reactions']
            spans = st.session_state['spans']

            # TABS for organized view
            tab1, tab2 = st.tabs(["📊 Analysis Results", "🏗️ RC Design"])
            
            with tab1:
                # 1. Interactive Graph
                design_view.draw_interactive_diagrams(df, spans, st.session_state['sup_df'], st.session_state['loads'], params['u_force'], params['u_len'])
                
                # 2. Reaction Table (Compact)
                st.markdown("##### 📍 Support Reactions")
                r_data = [{"Node": f"Sup {i+1}", "Ry (kg)": f"{reac[2*i]:.2f}", "Mz (kg-m)": f"{reac[2*i+1]:.2f}"} for i in range(len(spans)+1)]
                st.table(pd.DataFrame(r_data).set_index("Node").T)
            
            with tab2:
                # 3. RC Design Module (The Missing Piece Restored!)
                design_view.render_design_sheet(df, spans, params)
                
        else:
            st.info("👈 Please define model and click 'RUN ANALYSIS'")

if __name__ == "__main__":
    main()
