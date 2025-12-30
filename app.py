import streamlit as st
import pandas as pd
import numpy as np
import solver
import input_handler
import design_view # Make sure to import the updated one containing draw_diagrams
from datetime import datetime

# (Configuration & CSS เหมือนเดิม ... Copy มาวางได้เลย)
st.set_page_config(page_title="Pro RC Beam", layout="wide", page_icon="🏗️")

# CSS (Keep your existing CSS for cards/shadows)
st.markdown("""
<style>
    /* ... (Copy CSS from previous response) ... */
    .metric-card { background: white; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🏗️ Structural Analysis & Design")

    with st.sidebar:
        params = input_handler.render_sidebar()

    # --- ADJUSTED LAYOUT RATIO ---
    # Give input column more width (45%) to fix "Narrow Input" issue
    col_input, col_output = st.columns([45, 55], gap="large")

    with col_input:
        st.subheader("1. Model Input")
        n, spans, sup_df, stable = input_handler.render_model_inputs(params)
        st.markdown("###")
        loads = input_handler.render_loads(n, spans, params)
        
        st.markdown("###")
        run_btn = st.button("🚀 Analyze Structure", type="primary", use_container_width=True, disabled=not stable)

    with col_output:
        if run_btn or st.session_state.get('analysis_done'):
            if run_btn:
                # ... (Solver Logic เหมือนเดิม) ...
                try:
                    engine = solver.BeamSolver(spans, sup_df, loads)
                    df_res, reactions = engine.solve()
                    st.session_state.update({
                        'analysis_done': True, 'df_res': df_res, 'reactions': reactions,
                        'spans': spans, 'sup_df': sup_df, 'loads': loads
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            # Retrieve Data
            df = st.session_state['df_res']
            reac = st.session_state['reactions']
            
            # Summary Cards
            st.subheader("2. Results Summary")
            c1, c2, c3, c4 = st.columns(4)
            # Simple metric display
            c1.metric("Max M(+)", f"{df['moment'].max():.2f}", "kg-m")
            c2.metric("Max M(-)", f"{df['moment'].min():.2f}", "kg-m")
            c3.metric("Max Shear", f"{df['shear'].abs().max():.2f}", "kg")
            c4.metric("Max Reac.", f"{np.max(np.abs(reac[::2])):.2f}", "kg")

            # TABS
            tab1, tab2, tab3 = st.tabs(["📈 Diagrams", "🏗️ Design", "📋 Data & Reactions"])
            
            with tab1:
                # เรียกใช้ Function ใหม่ที่วาด Beam Model + Graphs
                design_view.draw_diagrams(df, st.session_state['spans'], st.session_state['sup_df'], 
                                          st.session_state['loads'], params['u_force'], params['u_len'])
            
            with tab2:
                # (Design Logic UI - Copy from previous version)
                # ... 
                # (เพื่อความสั้น ผมขอละไว้ แต่คุณต้องใส่ Code loop span input ตรงนี้นะครับ)
                st.info("Design Section (Please copy loop code from previous version)")
                # design_view.render_design_results(...) 

            with tab3:
                c_reac, c_table = st.columns([1, 2])
                
                with c_reac:
                    st.markdown("##### 📍 Support Reactions")
                    r_data = []
                    for i in range(len(st.session_state['spans']) + 1):
                        r_data.append({
                            "Node": f"#{i+1}",
                            "Ry (kg)": f"{reac[2*i]:.2f}",       # เพิ่มหน่วยชัดเจน
                            "Mz (kg-m)": f"{reac[2*i+1]:.2f}"    # เพิ่มหน่วยชัดเจน
                        })
                    st.dataframe(pd.DataFrame(r_data), use_container_width=True, hide_index=True)

                with c_table:
                    st.markdown("##### 🔢 Detailed Forces (V, M)")
                    # แสดงตาราง Shear/Moment ตามที่ขอ
                    st.dataframe(df.style.format("{:.2f}"), use_container_width=True, height=300)

        else:
            st.info("👈 Please setup model and click Run.")

if __name__ == "__main__":
    main()
