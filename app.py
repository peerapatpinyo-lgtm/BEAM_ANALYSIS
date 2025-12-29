import streamlit as st
import pandas as pd
import numpy as np
import beam_analysis
import input_handler
import design_view

st.set_page_config(page_title="RC Beam Pro", layout="wide")

st.markdown("## 🏗️ RC Beam Design: Professional Edition")

def main():
    params = input_handler.render_sidebar()
    
    col_geo, col_load = st.columns([1, 1.5])
    with col_geo:
        n, spans, sup_df, stable = input_handler.render_geometry()
    with col_load:
        loads = input_handler.render_loads(n, spans, params)
        
    st.markdown("---")
    
    if st.button("🚀 Calculate & Design", type="primary", disabled=not stable):
        try:
            # Analysis
            engine = beam_analysis.BeamAnalysisEngine(spans, sup_df, loads)
            df_res, reactions = engine.solve()
            
            if df_res is None:
                st.error("Structure Unstable!")
                return
            
            # Results
            st.success("Analysis Complete!")
            design_view.draw_diagrams(df_res, spans, sup_df, loads, params['u_force'], 'm')
            
            # Reactions Table
            n_nodes = n + 1
            reac_vals = reactions[:n_nodes*2].reshape(-1, 2)
            df_reac = pd.DataFrame(reac_vals, columns=[f"Fy ({params['u_force']})", "Mz"])
            df_reac.insert(0, "Node", range(1, n_nodes+1))
            st.table(df_reac)
            
            # Design
            design_view.render_design_results(df_res, params, spans, sup_df)
            
        except Exception as e:
            st.error(f"Error during calculation: {e}")
            # Hint: If you see 'NoneType' object is not subscriptable, check input loads.

if __name__ == "__main__":
    main()
