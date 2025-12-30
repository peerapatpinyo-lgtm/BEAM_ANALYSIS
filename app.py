import streamlit as st
import pandas as pd
import numpy as np
import solver
import input_handler
import design_view 
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Structural Analysis Pro", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    /* ปรับ Padding ให้ชิดขอบบนมากขึ้น เพื่อประหยัดที่ */
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    /* ปรับปุ่มให้ดูแน่นขึ้น */
    .stButton button { width: 100%; border-radius: 4px; font-weight: bold; }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def main():
    # --- HEADER ---
    c1, c2 = st.columns([8, 2])
    with c1: st.title("🏗️ Beam Analysis Professional")
    with c2: st.caption(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    st.divider()

    # --- MAIN LAYOUT (Input 20% | Output 80%) ---
    col_input, col_output = st.columns([20, 80], gap="medium")

    # === LEFT PANEL: INPUT ===
    with col_input:
        st.subheader("⚙️ Setup")
        
        with st.expander("1. Structure & Loads", expanded=True):
            params = input_handler.render_sidebar() # เรียกใช้ Sidebar settings
            n, spans, sup_df, stable = input_handler.render_model_inputs(params)
            st.markdown("---")
            loads = input_handler.render_loads(n, spans, params)
        
        st.markdown("###")
        run_btn = st.button("RUN ANALYSIS ▶", type="primary", disabled=not stable)
        if not stable: st.error("Unstable Model")

    # === RIGHT PANEL: RESULTS ===
    with col_output:
        if run_btn or st.session_state.get('analysis_done'):
            if run_btn:
                try:
                    engine = solver.BeamSolver(spans, sup_df, loads)
                    df_res, reactions = engine.solve()
                    st.session_state.update({'analysis_done': True, 'df_res': df_res, 'reactions': reactions, 
                                           'spans': spans, 'sup_df': sup_df, 'loads': loads})
                except Exception as e:
                    st.error(f"Solver Error: {e}")
                    st.stop()

            df = st.session_state['df_res']
            reac = st.session_state['reactions']
            spans = st.session_state['spans']

            # 1. INTERACTIVE DIAGRAMS (Fit Screen)
            # ไม่ต้องใช้ Slider แล้วครับ ใช้เมาส์ชี้ที่กราฟได้เลย
            design_view.draw_interactive_diagrams(df, spans, st.session_state['sup_df'], 
                                                  st.session_state['loads'], params['u_force'], params['u_len'])
            
            # 2. COMPACT TABLE
            st.markdown("---")
            design_view.render_engineering_table(df, reac, spans)

        else:
            st.info("👈 Please define model and click 'RUN ANALYSIS' to start.")

if __name__ == "__main__":
    main()
