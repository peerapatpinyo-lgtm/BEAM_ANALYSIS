import streamlit as st
import pandas as pd
import beam_analysis
import input_handler
import design_view

st.set_page_config(page_title="RC Beam Pro", layout="wide", page_icon="🏗️")
st.markdown("""<style>.stApp { font-family: 'Sarabun', sans-serif; } h1, h2, h3 { color: #0D47A1; } .stButton>button { background-color: #1565C0; color: white; border-radius: 8px; font-weight: bold; }</style>""", unsafe_allow_html=True)

st.title("🏗️ RC Beam Design: Professional Edition (Rev.3)")

def main():
    params = input_handler.render_sidebar()
    
    c_geo, c_load = st.columns([1.2, 1.5])
    with c_geo:
        # FIX: Receive 5 values (Added span_props for variable sections)
        n, spans, sup_df, stable, span_props = input_handler.render_geometry()
    with c_load:
        loads = input_handler.render_loads(n, spans, params)
        
    st.markdown("---")
    
    if st.button("🚀 RUN ANALYSIS & DESIGN", type="primary", use_container_width=True, disabled=not stable):
        with st.spinner("Executing Finite Element Analysis & RC Design..."):
            try:
                engine = beam_analysis.BeamAnalysisEngine(spans, sup_df, loads)
                df_res, reactions = engine.solve()
                
                if df_res is None:
                    st.error("Analysis Failed: Structure is unstable (Singularity Matrix).")
                    return
                
                st.success("✅ Analysis Complete!")
                
                # Analysis Visualization
                design_view.draw_diagrams(df_res, spans, sup_df, loads, params['u_force'], 'm')
                
                # Reaction Table
                with st.expander("Show Reaction Forces"):
                    n_nodes = n + 1
                    r_vals = reactions[:n_nodes*2].reshape(-1, 2)
                    df_r = pd.DataFrame(r_vals, columns=[f"Fy ({params['u_force']})", f"Mz ({params['u_force']}-m)"])
                    df_r.insert(0, "Node", [f"N{i+1}" for i in range(n_nodes)])
                    st.dataframe(df_r.style.format("{:.2f}"), hide_index=True, use_container_width=True)

                # Design Visualization (Must pass span_props)
                design_view.render_design_results(df_res, params, spans, span_props, sup_df)
            
            except Exception as e:
                st.error(f"Calculation Error: {e}")
                # Optional: st.exception(e) for debugging

if __name__ == "__main__":
    main()
