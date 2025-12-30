import streamlit as st
import pandas as pd
import numpy as np
import solver
import input_handler
import design_view 
from datetime import datetime

# --- PRO CONFIG ---
st.set_page_config(page_title="Structural Analysis Pro", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    /* Compact Layout fixes */
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    h1 { font-size: 1.5rem; margin-bottom: 0; }
    .stButton button { width: 100%; border-radius: 4px; font-weight: bold; }
    /* Hide default Streamlit footer */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def main():
    # --- HEADER ---
    c_head_1, c_head_2 = st.columns([8, 2])
    with c_head_1:
        st.title("🏗️ Beam Analysis Professional")
        st.caption("Finite Element Method (Stiffness Matrix) | ACI 318 Standard")
    with c_head_2:
        st.markdown(f"<div style='text-align:right; font-size:0.8rem; color:grey;'>{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>", unsafe_allow_html=True)
    
    st.divider()

    # --- MAIN SPLIT (20% Input / 80% Output) ---
    col_input, col_output = st.columns([2, 8], gap="medium")

    # === LEFT: CONTROLS ===
    with col_input:
        st.subheader("🛠️ Model")
        with st.expander("1. Geometry & Supports", expanded=True):
            params = input_handler.render_sidebar() # ปรับให้ input อยู่ใน expander ได้ถ้ายาวไป
            n, spans, sup_df, stable = input_handler.render_model_inputs(params)
        
        with st.expander("2. Loading Conditions", expanded=True):
            loads = input_handler.render_loads(n, spans, params)
            
        st.markdown("###")
        run_btn = st.button("RUN ANALYSIS ▶", type="primary", disabled=not stable)
        
        if not stable:
            st.error("Structure Unstable")

    # === RIGHT: VISUALIZATION & RESULTS ===
    with col_output:
        if run_btn or st.session_state.get('analysis_done'):
            if run_btn:
                try:
                    engine = solver.BeamSolver(spans, sup_df, loads)
                    df_res, reactions = engine.solve()
                    st.session_state.update({'analysis_done': True, 'df_res': df_res, 'reactions': reactions, 
                                           'spans': spans, 'sup_df': sup_df, 'loads': loads})
                except Exception as e:
                    st.error(f"Computation Error: {e}")
                    st.stop()

            df = st.session_state['df_res']
            reac = st.session_state['reactions']
            spans = st.session_state['spans']

            # 1. THE INTERACTIVE GRAPH (Full Width)
            # Hover over this graph to see exact X values. No slider needed.
            design_view.draw_interactive_diagrams(df, spans, st.session_state['sup_df'], 
                                                  st.session_state['loads'], params['u_force'], params['u_len'])
            
            # 2. PRO DATA TABLE (Compact below graph)
            st.markdown("---")
            design_view.render_engineering_table(df, reac, spans)

        else:
            # Empty State (Professional Placeholder)
            st.info("👈 Please define the beam geometry and loads on the left panel to initialize the Finite Element Solver.")
            st.markdown("""
            **Capabilities:**
            * Matrix Stiffness Method Solver
            * Point Loads & Uniform Distributed Loads
            * Shear & Moment Diagrams (Interactive)
            * Reaction Calculation
            """)

if __name__ == "__main__":
    main()
