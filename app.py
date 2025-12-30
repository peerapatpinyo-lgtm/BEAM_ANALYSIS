import streamlit as st
import pandas as pd
import beam_analysis
import input_handler
import design_view

st.set_page_config(page_title="RC Beam Pro", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    .stApp { font-family: 'Sarabun', sans-serif; }
    h1, h2, h3 { color: #1565C0; }
    .stButton>button { background-color: #1565C0; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ RC Beam Design: Professional Edition")

def main():
    params = input_handler.render_sidebar()
    
    col_geo, col_load = st.columns([1, 1.5])
    with col_geo:
        n, spans, sup_df, stable = input_handler.render_geometry()
    with col_load:
        loads = input_handler.render_loads(n, spans, params)
        
    st.markdown("---")
    
    if st.button("🚀 EXECUTE ANALYSIS & DESIGN", type="primary", use_container_width=True, disabled=not stable):
        with st.spinner("Analyzing Structure..."):
            try:
                engine = beam_analysis.BeamAnalysisEngine(spans, sup_df, loads)
                df_res, reactions = engine.solve()
                
                if df_res is None:
                    st.error("Matrix Singularity Error: Structure is unstable.")
                    return
                
                st.success("✅ Analysis Complete!")
                
                st.markdown("### 3️⃣ Analysis Results")
                
                n_nodes = n + 1
                reac_vals = reactions[:n_nodes*2].reshape(-1, 2)
                df_reac = pd.DataFrame(reac_vals, columns=[f"Fy ({params['u_force']})", f"Mz ({params['u_force']}-m)"])
                df_reac.insert(0, "Node", range(1, n_nodes+1))
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown("**Reaction Forces**")
                    st.dataframe(df_reac.style.format("{:.2f}"), hide_index=True, use_container_width=True)
                with c2:
                    design_view.draw_diagrams(df_res, spans, sup_df, loads, params['u_force'], 'm')
                
                design_view.render_design_results(df_res, params, spans, sup_df)
            
            except Exception as e:
                st.error(f"Calculation Error: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()
