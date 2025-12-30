import streamlit as st
import pandas as pd
import beam_analysis
import input_handler
import design_view

st.set_page_config(page_title="RC Beam Pro", layout="wide", page_icon="🏗️")
st.markdown("""<style>.stApp { font-family: 'Sarabun', sans-serif; } h1, h2, h3 { color: #0D47A1; } .stButton>button { background-color: #1565C0; color: white; border-radius: 8px; font-weight: bold; }</style>""", unsafe_allow_html=True)

st.title("🏗️ RC Beam Design: Professional Edition")

def main():
    params = input_handler.render_sidebar()
    
    c_geo, c_load = st.columns([1.2, 1.5])
    with c_geo:
        n, spans, sup_df, stable, span_props = input_handler.render_geometry()
    with c_load:
        loads = input_handler.render_loads(n, spans, params)
        
    st.markdown("---")
    
    if st.button("🚀 RUN ANALYSIS & DESIGN", type="primary", use_container_width=True, disabled=not stable):
        with st.spinner("Processing..."):
            try:
                engine = beam_analysis.BeamAnalysisEngine(spans, sup_df, loads)
                df_res, reactions = engine.solve()
                
                if df_res is None:
                    st.error("Structure is unstable.")
                    return
                
                st.success("✅ Analysis Complete!")
                
                # 1. Plot Diagrams
                design_view.draw_diagrams(df_res, spans, sup_df, loads, params['u_force'], 'm')
                
                # 2. Show Reactions Table
                st.markdown(f"#### ⚓ Support Reactions ({params['u_force']})")
                n_nodes = n + 1
                
                react_data = []
                for i in range(n_nodes):
                    # 🔴 FIX: Force float conversion before formatting
                    ry = float(reactions[2*i])
                    mz = float(reactions[2*i+1])
                    
                    react_data.append({
                        "Node": f"N{i+1}",
                        f"Fy": f"{ry:,.2f}" if abs(ry)>0.01 else "-",
                        f"Mz": f"{mz:,.2f}" if abs(mz)>0.01 else "-"
                    })
                
                st.table(pd.DataFrame(react_data))

                # 3. Design Results
                design_view.render_design_results(df_res, params, spans, span_props, sup_df)
            
            except Exception as e:
                st.error(f"Critical Error: {e}")
                st.info("Check your inputs: Ensure all Loads and Dimensions are valid numbers.")

if __name__ == "__main__":
    main()
