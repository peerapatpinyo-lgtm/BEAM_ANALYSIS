import streamlit as st
import pandas as pd
import beam_analysis
import input_handler
import design_view

st.set_page_config(page_title="RC Beam Pro", layout="wide", page_icon="🏗️")

# Custom CSS for cleaner look
st.markdown("""
<style>
    .stApp { font-family: 'Sarabun', sans-serif; background-color: #F8F9FA; }
    h1, h2, h3 { color: #1565C0; }
    .stButton>button { background-color: #1565C0; color: white; border-radius: 6px; height: 3em; }
    div[data-testid="stExpander"] { background-color: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🏗️ RC Beam Analysis & Design")
    
    # 1. Sidebar Settings
    params = input_handler.render_sidebar()
    
    # 2. Model Inputs (Left & Right layout for desktop)
    col_input, col_preview = st.columns([1, 1.5])
    
    with col_input:
        n, spans, sup_df, stable = input_handler.render_model_inputs(params)
        
        st.markdown("###")
        btn_analyze = st.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=not stable)

    # 3. Logic Control with Session State
    if btn_analyze:
        with st.spinner("Analyzing structure..."):
            try:
                # Run Analysis (Assume constant EI for force finding)
                engine = beam_analysis.BeamAnalysisEngine(spans, sup_df, st.session_state.loads)
                df_res, reactions = engine.solve()
                
                if df_res is not None:
                    # Save results to session state
                    st.session_state['analysis_done'] = True
                    st.session_state['df_res'] = df_res
                    st.session_state['reactions'] = reactions
                    st.session_state['spans'] = spans
                    st.session_state['sup_df'] = sup_df
                else:
                    st.error("Unstable Structure!")
            except Exception as e:
                st.error(f"Analysis Error: {e}")

    # 4. Display Results (Only if analysis exists)
    if st.session_state.get('analysis_done'):
        df = st.session_state['df_res']
        
        with col_preview:
            # Show Diagrams immediately next to inputs
            st.subheader("📊 Analysis Results")
            design_view.draw_diagrams(df, st.session_state['spans'], st.session_state['sup_df'], 
                                      st.session_state.loads, params['u_force'], params['u_len'])
        
        # --- NEW SECTION: Design Parameters (Bottom) ---
        st.markdown("---")
        st.header("✨ Section Design & Detailing")
        
        # 4.1 Define Sections (Interactive)
        st.info("👇 Adjust beam sizes here. Reinforcement will update automatically.")
        
        # Create interactive columns for inputs
        n_spans = len(st.session_state['spans'])
        cols = st.columns(n_spans)
        span_props = []
        
        for i in range(n_spans):
            with cols[i]:
                st.markdown(f"**Span {i+1}**")
                with st.container(border=True):
                    b = st.number_input(f"Width b (cm)", value=25.0, step=5.0, key=f"des_b_{i}")
                    h = st.number_input(f"Depth h (cm)", value=50.0, step=5.0, key=f"des_h_{i}")
                    cv = st.number_input(f"Cover (cm)", value=3.0, step=0.5, key=f"des_c_{i}")
                    span_props.append({"b": b, "h": h, "cv": cv})

        # 4.2 Show Design Tables & Details
        design_view.render_design_results(df, params, st.session_state['spans'], span_props, st.session_state['sup_df'])
        
    else:
        with col_preview:
            st.info("👈 Please define geometry and loads, then click 'Run Analysis'.")

if __name__ == "__main__":
    main()
