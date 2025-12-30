import streamlit as st
import pandas as pd
import numpy as np
import solver
import input_handler
import design_view 
from datetime import datetime

st.set_page_config(page_title="Pro Structural Analysis", layout="wide", page_icon="🏗️")

# Custom CSS for polished look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Kanit', sans-serif; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 45px; border-radius: 5px; background-color: #F1F5F9; font-weight: 600; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { font-size: 1.5rem; color: #1E3A8A; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem; color: #64748B; }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🏗️ Professional Beam Analysis Suite")
    st.caption(f"Engineering Standard: ACI 318 / EIT | Date: {datetime.now().strftime('%d %b %Y')}")

    with st.sidebar:
        params = input_handler.render_sidebar()

    # --- MAIN LAYOUT (40% Input / 60% Output) ---
    col_input, col_output = st.columns([4, 6], gap="large")

    with col_input:
        n, spans, sup_df, stable = input_handler.render_model_inputs(params)
        st.markdown("###")
        loads = input_handler.render_loads(n, spans, params)
        
        st.markdown("---")
        run_btn = st.button("🚀 Analyze Structure", type="primary", use_container_width=True, disabled=not stable)
        if not stable: st.error("⚠️ Structure is Unstable")

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

            # Results
            df = st.session_state['df_res']
            reac = st.session_state['reactions']
            
            # 1. Summary Metrics
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Max M(+)", f"{df['moment'].max():.2f}", "kg-m")
                c2.metric("Max M(-)", f"{df['moment'].min():.2f}", "kg-m")
                c3.metric("Max Shear", f"{df['shear'].abs().max():.2f}", "kg")
                c4.metric("Max Reaction", f"{np.max(np.abs(reac[::2])):.2f}", "kg")

            # 2. Tabs
            t1, t2, t3 = st.tabs(["📈 Diagrams & Model", "🏗️ RC Design", "📋 Tables & Reactions"])
            
            with t1:
                design_view.draw_diagrams(df, st.session_state['spans'], st.session_state['sup_df'], 
                                          st.session_state['loads'], params['u_force'], params['u_len'])
            
            with t2:
                # (เรียกใช้ design_view.render_design_results ตามปกติ)
                # เพื่อความสมบูรณ์ อย่าลืม copy loop input จากเวอร์ชั่นก่อนมาใส่ตรงนี้
                st.info("Please insert the Design Input Loop here (same as previous version).")
                
            with t3:
                c_reac, c_dat = st.columns([1, 1.5])
                
                with c_reac:
                    st.markdown("###### 📍 Support Reactions")
                    r_data = []
                    for i in range(len(st.session_state['spans']) + 1):
                        r_data.append({
                            "Support": f"Node {i+1}",
                            "Ry (kg)": f"{reac[2*i]:.2f}",       # <--- ใส่หน่วยชัดเจน
                            "Mz (kg-m)": f"{reac[2*i+1]:.2f}"    # <--- ใส่หน่วยชัดเจน
                        })
                    st.dataframe(pd.DataFrame(r_data), use_container_width=True, hide_index=True)
                
                with c_dat:
                    st.markdown("###### 🔢 Internal Forces Table")
                    st.dataframe(df, use_container_width=True, height=300)

        else:
            st.info("👈 Please define spans, supports, and loads to proceed.")

if __name__ == "__main__":
    main()
