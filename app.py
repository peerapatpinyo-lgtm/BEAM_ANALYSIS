import streamlit as st
import pandas as pd
import numpy as np
import beam_analysis
import input_handler
import design_view

# ==============================================================================
# CONFIG
# ==============================================================================
st.set_page_config(page_title="RC Beam Pro: Engineer Edition", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; font-size: 14px; }
    h1, h2, h3 { color: #0D47A1; font-weight: 700; }
    .stApp { background-color: #F5F5F5; }
    .header-box {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        color: white; padding: 20px; border-radius: 8px;
        text-align: center; margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    .section-header {
        border-left: 5px solid #1565C0; padding-left: 15px;
        font-size: 1.2rem; font-weight: 700; margin-top: 30px; margin-bottom: 15px;
        background-color: #E3F2FD; padding: 10px; border-radius: 0 5px 5px 0;
        color: #0D47A1; 
    }
    .card {
        background: white; padding: 15px; border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #E0E0E0;
        margin-bottom: 10px;
    }
    .stNumberInput input { font-weight: bold; color: #1565C0; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    st.markdown('<div class="header-box"><h1>🏗️ RC Beam Pro: Engineer Edition</h1></div>', unsafe_allow_html=True)
    
    # 1. Inputs
    params = input_handler.render_sidebar()
    c_geo, c_load = st.columns([1, 1.4])
    with c_geo:
        n, spans, sup_df, stable = input_handler.render_geometry()
    with c_load:
        loads = input_handler.render_loads(n, spans, params)
        
    st.markdown("---")
    if st.button("🚀 RUN ANALYSIS & DESIGN", type="primary", use_container_width=True, disabled=not stable):
        try:
            with st.spinner("Calculating Stiffness Matrix & Designing..."):
                engine = beam_analysis.BeamAnalysisEngine(spans, sup_df, loads)
                df_res, reactions = engine.solve()
                
                if df_res is None:
                    st.error("Structure Unstable (Singular Matrix)")
                    st.stop()
                
                # --- RESULTS ---
                st.markdown('<div class="section-header">3️⃣ Analysis Results</div>', unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Support Reactions**")
                    n_nodes = n + 1
                    reac_vals = reactions[:n_nodes*2].reshape(-1, 2)
                    df_reac = pd.DataFrame(reac_vals, columns=[f"Fy ({params['u_force']})", f"Mz ({params['u_force']}-{params['u_len']})"])
                    df_reac.insert(0, "Node", range(1, n_nodes+1))
                    st.dataframe(df_reac, use_container_width=True, hide_index=True)
                    
                with c2:
                    st.markdown("**Critical Forces**")
                    st.info(f"Max Shear: **{df_res['shear'].abs().max():,.2f} {params['u_force']}**")
                    st.info(f"Max Moment: **{df_res['moment'].abs().max():,.2f} {params['u_force']}-{params['u_len']}**")

                design_view.draw_diagrams(df_res, spans, sup_df, loads, params['u_force'], params['u_len'])
                
                # ส่ง spans และ sup_df ไปให้ฟังก์ชัน design results เพื่อวาดรูปเหล็กตามยาว
                design_view.render_design_results(df_res, params, spans, sup_df)
                
        except Exception as e:
            st.error(f"System Error: {e}")

if __name__ == "__main__":
    main()
