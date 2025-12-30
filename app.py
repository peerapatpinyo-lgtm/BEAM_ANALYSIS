import streamlit as st
import pandas as pd
import beam_analysis
import input_handler
import design_view

st.set_page_config(page_title="RC Beam Pro", layout="wide", page_icon="🏗️")
st.markdown("""<style>.stApp { font-family: 'Sarabun', sans-serif; background-color: #F8F9FA; } h1, h2, h3 { color: #1565C0; } .stButton>button { background-color: #1565C0; color: white; height: 3em; }</style>""", unsafe_allow_html=True)

def main():
    st.title("🏗️ RC Beam Analysis & Design (Professional)")
    
    # 1. Sidebar
    params = input_handler.render_sidebar()
    
    # 2. Main Input Area
    col_input, col_preview = st.columns([1, 1.5])
    
    with col_input:
        n, spans, sup_df, stable = input_handler.render_model_inputs(params)
        loads = input_handler.render_loads(n, spans, params)
        
        st.markdown("###")
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=not stable):
            with st.spinner("Analyzing..."):
                try:
                    engine = beam_analysis.BeamAnalysisEngine(spans, sup_df, loads)
                    df_res, reactions = engine.solve() # Now returns high-res data
                    
                    if df_res is not None:
                        st.session_state['analysis_done'] = True
                        st.session_state['df_res'] = df_res
                        st.session_state['reactions'] = reactions
                        st.session_state['spans'] = spans
                        st.session_state['sup_df'] = sup_df
                        st.session_state['loads'] = loads # Save for plotting
                    else:
                        st.error("Structure Unstable!")
                except Exception as e:
                    st.error(f"Error: {e}")

    # 3. Results Area
    if st.session_state.get('analysis_done'):
        df = st.session_state['df_res']
        
        with col_preview:
            # Plot Diagrams
            design_view.draw_diagrams(df, st.session_state['spans'], st.session_state['sup_df'], 
                                      st.session_state['loads'], params['u_force'], params['u_len'])
            
        # Design Section (Below)
        st.markdown("---")
        st.header("✨ Section Design")
        st.info("👇 Define beam size here. Reinforcement updates automatically.")
        
        n_spans = len(st.session_state['spans'])
        cols = st.columns(n_spans)
        span_props = []
        
        for i in range(n_spans):
            with cols[i]:
                st.markdown(f"**Span {i+1}**")
                with st.container(border=True):
                    b = st.number_input(f"b (cm)", value=25.0, key=f"d_b_{i}")
                    h = st.number_input(f"h (cm)", value=50.0, key=f"d_h_{i}")
                    cv = st.number_input(f"Cov (cm)", value=3.0, key=f"d_c_{i}")
                    span_props.append({"b": b, "h": h, "cv": cv})

        design_view.render_design_results(df, params, st.session_state['spans'], span_props, st.session_state['sup_df'])
        
    else:
        with col_preview:
            st.info("👈 Define structure and Click 'Run Analysis'")

if __name__ == "__main__":
    main()
