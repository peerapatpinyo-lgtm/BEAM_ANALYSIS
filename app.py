import streamlit as st
import pandas as pd
import numpy as np
import solver
import input_handler
import design_view 
from datetime import datetime

st.set_page_config(page_title="Pro Beam Analysis", layout="wide", page_icon="🏗️")

# CSS (เหมือนเดิม)
st.markdown("""<style>...</style>""", unsafe_allow_html=True) 

def main():
    st.title("🏗️ Structural Analysis Suite")
    
    with st.sidebar:
        params = input_handler.render_sidebar()

    col_input, col_output = st.columns([4, 6], gap="large")

    with col_input:
        n, spans, sup_df, stable = input_handler.render_model_inputs(params)
        st.markdown("###")
        loads = input_handler.render_loads(n, spans, params)
        st.markdown("---")
        run_btn = st.button("🚀 Analyze Structure", type="primary", use_container_width=True, disabled=not stable)

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
            total_len = sum(st.session_state['spans'])

            # Summary Cards
            # (Code เดิม: แสดง Max Shear, Max Moment)
            
            # --- INTERACTIVE PROBE TOOL ---
            st.markdown("###") # Spacer
            c_label, c_slider = st.columns([2, 8])
            with c_label:
                st.markdown("**🔍 Inspect Graph at (m):**")
            with c_slider:
                # Slider จะส่งค่า x กลับไปที่ฟังก์ชันวาดกราฟ
                probe_val = st.slider("Probe Location", 0.0, float(total_len), float(total_len)/2, 0.1, label_visibility="collapsed")

            # TABS
            t1, t2, t3 = st.tabs(["📈 Diagrams", "🏗️ Design", "📋 Data"])
            
            with t1:
                # ส่ง probe_val เข้าไปวาดเส้น
                design_view.draw_diagrams(df, st.session_state['spans'], st.session_state['sup_df'], 
                                          st.session_state['loads'], params['u_force'], params['u_len'],
                                          probe_x=probe_val)
            
            with t2:
                # Design UI (เหมือนเดิม)
                st.info("Design Section Loop") 
                
            with t3:
                # Data Tables (เหมือนเดิม)
                st.dataframe(df)

        else:
            st.info("👈 Set up model to begin.")

if __name__ == "__main__":
    main()
