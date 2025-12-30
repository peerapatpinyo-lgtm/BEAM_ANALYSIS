import streamlit as st
import pandas as pd
import numpy as np
import solver
import input_handler
import design_view 
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Structural Beam Analysis", layout="wide", page_icon="📐")

# --- CLEAN CSS (Report Style) ---
st.markdown("""
<style>
    /* Clean Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; color: #1F2937; }
    
    /* Summary Cards (Clean White with Border) */
    .metric-card {
        background-color: white;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-title { font-size: 0.85rem; color: #6B7280; font-weight: 500; margin-bottom: 5px; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #111827; }
    .metric-unit { font-size: 0.8rem; color: #9CA3AF; margin-left: 3px; }
    
    /* Highlight Colors for Cards */
    .c-blue .metric-value { color: #2563EB; }
    .c-orange .metric-value { color: #F59E0B; }
    .c-red .metric-value { color: #DC2626; }
</style>
""", unsafe_allow_html=True)

def card(title, value, unit, color_class=""):
    return f"""
    <div class="metric-card {color_class}">
        <div class="metric-title">{title}</div>
        <div><span class="metric-value">{value}</span><span class="metric-unit">{unit}</span></div>
    </div>
    """

def main():
    st.title("📐 Structural Analysis & Design")
    st.caption(f"Standard Analysis Report | ACI 318 | Date: {datetime.now().strftime('%d/%m/%Y')}")
    
    with st.sidebar:
        params = input_handler.render_sidebar()

    col_input, col_output = st.columns([35, 65], gap="large")

    with col_input:
        st.subheader("1. Geometry & Load")
        with st.container(border=True):
            n, spans, sup_df, stable = input_handler.render_model_inputs(params)
        st.markdown("###")
        with st.container(border=True):
            loads = input_handler.render_loads(n, spans, params)
        st.markdown("###")
        run_btn = st.button("Calculate", type="primary", use_container_width=True, disabled=not stable)

    with col_output:
        if run_btn or st.session_state.get('analysis_done'):
            if run_btn:
                try:
                    engine = solver.BeamSolver(spans, sup_df, loads)
                    df_res, reactions = engine.solve()
                    st.session_state.update({'analysis_done': True, 'df_res': df_res, 'reactions': reactions, 
                                           'spans': spans, 'sup_df': sup_df, 'loads': loads})
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            df = st.session_state['df_res']
            reac = st.session_state['reactions']
            total_len = sum(st.session_state['spans'])

            # --- CLEAN SUMMARY CARDS ---
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(card("Max Moment (+)", f"{df['moment'].max():.2f}", "kg-m", "c-blue"), unsafe_allow_html=True)
            with c2: st.markdown(card("Max Moment (-)", f"{df['moment'].min():.2f}", "kg-m", "c-blue"), unsafe_allow_html=True)
            with c3: st.markdown(card("Max Shear", f"{df['shear'].abs().max():.2f}", "kg", "c-orange"), unsafe_allow_html=True)
            with c4: st.markdown(card("Max Reaction", f"{np.max(np.abs(reac[::2])):.2f}", "kg", "c-red"), unsafe_allow_html=True)
            
            st.markdown("###")

            # --- TABS ---
            t1, t2, t3 = st.tabs(["📊 Diagrams", "🏗️ RC Design", "📋 Reactions"])
            
            with t1:
                # Probe Slider
                c_lbl, c_sli = st.columns([2, 8])
                with c_lbl: st.markdown("**🔍 Probe (x):**")
                with c_sli: probe_val = st.slider("p", 0.0, total_len, 0.0, 0.1, label_visibility="collapsed")
                
                design_view.draw_diagrams(df, st.session_state['spans'], st.session_state['sup_df'], 
                                          st.session_state['loads'], params['u_force'], params['u_len'], probe_x=probe_val)
            
            with t2:
                # Design Input Loop
                st.info("Edit section properties to check capacity:")
                n_spans = len(st.session_state['spans'])
                span_props = []
                cols = st.columns(n_spans)
                for i in range(n_spans):
                    with cols[i]:
                        st.markdown(f"**Span {i+1}**")
                        b = st.number_input(f"b (cm)", value=30.0, key=f"b{i}")
                        h = st.number_input(f"h (cm)", value=60.0, key=f"h{i}")
                        mb = st.selectbox("Main", [12,16,20,25], index=1, key=f"m{i}", format_func=lambda x: f"DB{x}")
                        st = st.selectbox("Stirrup", [6,9], index=0, key=f"s{i}", format_func=lambda x: f"RB{x}")
                        span_props.append({"b":b, "h":h, "cv":3.5, "main_bar_dia":mb, "stirrup_dia":st})
                
                st.markdown("---")
                design_view.render_design_results(df, params, st.session_state['spans'], span_props, st.session_state['sup_df'])
                
            with t3:
                r_data = [{"Node": i+1, "Ry (kg)": f"{reac[2*i]:.2f}", "Mz (kg-m)": f"{reac[2*i+1]:.2f}"} 
                          for i in range(len(st.session_state['spans']) + 1)]
                st.table(pd.DataFrame(r_data))

        else:
            st.info("👈 Enter parameters and Calculate.")

if __name__ == "__main__":
    main()
