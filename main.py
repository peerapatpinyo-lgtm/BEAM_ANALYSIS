import streamlit as st
import pandas as pd
import numpy as np
import beam_analysis
import input_handler
import design_view

# ==============================================================================
# CSS & CONFIG
# ==============================================================================
st.set_page_config(page_title="RC Beam Pro: Ultimate Edition", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; font-size: 15px; }
    h1, h2, h3 { color: #0D47A1; font-weight: 700; }
    .stApp { background-color: #FAFAFA; }
    .header-box {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        color: white; padding: 25px; border-radius: 10px;
        text-align: center; margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .section-header {
        border-left: 6px solid #1565C0; padding-left: 15px;
        font-size: 1.4rem; font-weight: 700; margin-top: 35px; margin-bottom: 20px;
        background-color: #E3F2FD; padding: 12px; border-radius: 0 8px 8px 0;
        color: #0D47A1; display: flex; align-items: center;
    }
    .card {
        background: white; padding: 20px; border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); border: 1px solid #E0E0E0;
        margin-bottom: 15px;
    }
    .stNumberInput input { font-weight: bold; color: #1565C0; }
    .dataframe { width: 100% !important; font-size: 14px; }
    .dataframe th { background-color: #1565C0 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.markdown('<div class="header-box"><h1>🏗️ RC Beam Pro: Analysis & Design Suite</h1></div>', unsafe_allow_html=True)
    
    # 1. Inputs
    params = input_handler.render_sidebar()
    c_geo, c_load = st.columns([1, 1.3])
    with c_geo:
        n, spans, sup_df, stable = input_handler.render_geometry()
    with c_load:
        loads = input_handler.render_loads(n, spans, params)
        
    st.markdown("---")
    if st.button("🚀 RUN ANALYSIS & DESIGN", type="primary", use_container_width=True, disabled=not stable):
        try:
            with st.spinner("Solving Stiffness Matrix..."):
                # Initialize Engine
                engine = beam_analysis.BeamAnalysisEngine(spans, sup_df, loads)
                df_res, reactions = engine.solve()
                
                # --- RESULTS ---
                st.markdown('<div class="section-header">3️⃣ Analysis Results</div>', unsafe_allow_html=True)
                
                # Summary Tables
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Support Reactions (Raw Matrix Vector)**")
                    st.dataframe(pd.DataFrame(reactions[:(n+1)*2].reshape(-1,2), columns=["Fy", "Mz"]), use_container_width=True)
                with c2:
                    st.markdown("**Max Forces**")
                    st.write(f"Max Shear: **{df_res['shear'].abs().max():,.2f}**")
                    st.write(f"Max Moment: **{df_res['moment'].abs().max():,.2f}**")

                # Diagrams
                design_view.draw_diagrams(df_res, spans, sup_df, loads)
                
                # Design
                design_view.render_design_results(df_res, params)
                
        except Exception as e:
            st.error(f"Calculation Error: {e}")
            st.code(e)

if __name__ == "__main__":
    main()
