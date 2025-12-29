import streamlit as st
import pandas as pd
import numpy as np
import beam_analysis
import input_handler
import design_view

st.set_page_config(page_title="RC Beam Pro: Senior Engineer", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; font-size: 14px; }
    h1, h2, h3 { color: #0D47A1; font-weight: 700; }
    .header-box { background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 20px; }
    .section-header { border-left: 5px solid #1565C0; padding-left: 15px; font-size: 1.2rem; font-weight: 700; margin-top: 20px; background-color: #E3F2FD; padding: 10px; border-radius: 0 5px 5px 0; color: #0D47A1; }
    .stNumberInput input { font-weight: bold; color: #1565C0; }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="header-box"><h1>🏗️ RC Beam Pro: Senior Engineer Edition</h1></div>', unsafe_allow_html=True)
    
    params = input_handler.render_sidebar()
    c_geo, c_load = st.columns([1, 1.4])
    with c_geo:
        n, spans, sup_df, stable = input_handler.render_geometry()
    with c_load:
        loads = input_handler.render_loads(n, spans, params)
        
    st.markdown("---")
    if st.button("🚀 EXECUTE ANALYSIS & DESIGN", type="primary", use_container_width=True, disabled=not stable):
        try:
            with st.spinner("Analyzing Stiffness Matrix & Designing Components..."):
                engine = beam_analysis.BeamAnalysisEngine(spans, sup_df, loads)
                df_res, reactions = engine.solve()
                
                if df_res is None:
                    st.error("Matrix Singularity: Structure is unstable.")
                    st.stop()
                
                # --- RESULTS ---
                st.markdown('<div class="section-header">3️⃣ Analysis Results</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Node Reactions**")
                    n_nodes = n + 1
                    reac_vals = reactions[:n_nodes*2].reshape(-1, 2)
                    df_reac = pd.DataFrame(reac_vals, columns=[f"Fy ({params['u_force']})", f"Mz ({params['u_force']}-m)"])
                    df_reac.insert(0, "Node", range(1, n_nodes+1))
                    st.dataframe(df_reac.style.format("{:.2f}"), use_container_width=True, hide_index=True)
                with c2:
                    st.markdown("**Critical Envelope**")
                    st.info(f"V_max: **{df_res['shear'].abs().max():,.2f}**")
                    st.info(f"M_max: **{df_res['moment'].abs().max():,.2f}**")

                design_view.draw_diagrams(df_res, spans, sup_df, loads, params['u_force'], 'm')
                design_view.render_design_results(df_res, params, spans, sup_df)
                
        except Exception as e:
            st.error(f"Calculation Error: {e}")

if __name__ == "__main__":
    main()
