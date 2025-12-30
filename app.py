import streamlit as st
import pandas as pd
import numpy as np
import solver
import input_handler
import design_view
from datetime import datetime

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Pro RC Beam Designer", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Theme Overrides (aiming for a professional dark-mode-compatible look) */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 { font-weight: 700; letter-spacing: -0.5px; }
    h1 { color: #111827; }

    /* Premium Metric Cards with Gradients */
    .metric-container {
        display: flex; gap: 20px; margin-bottom: 25px;
    }
    .metric-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 24px;
        flex: 1;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        border: 1px solid #E5E7EB;
        position: relative;
        overflow: hidden;
    }
    /* Gradient Accents */
    .card-blue::before { content:''; position:absolute; top:0; left:0; width:100%; height:6px; background: linear-gradient(90deg, #3B82F6, #60A5FA); }
    .card-red::before { content:''; position:absolute; top:0; left:0; width:100%; height:6px; background: linear-gradient(90deg, #EF4444, #F87171); }
    .card-cyan::before { content:''; position:absolute; top:0; left:0; width:100%; height:6px; background: linear-gradient(90deg, #06B6D4, #67E8F9); }
    .card-purple::before { content:''; position:absolute; top:0; left:0; width:100%; height:6px; background: linear-gradient(90deg, #8B5CF6, #C4B5FD); }

    .metric-label { font-size: 0.9rem; color: #6B7280; font-weight: 600; margin-bottom: 8px; }
    .metric-value { font-size: 2rem; font-weight: 800; color: #111827; }
    .metric-unit { font-size: 1rem; color: #9CA3AF; font-weight: 500; margin-left: 4px;}

    /* Tabs & Inputs */
    .stTabs [data-baseweb="tab"] {
        height: 55px; border-radius: 8px; font-weight: 600; font-size: 1rem;
    }
    .span-box-header {
        background: #F3F4F6; padding: 12px 15px; border-radius: 10px 10px 0 0;
        font-weight: 700; color: #374151; border: 1px solid #E5E7EB; border-bottom: none;
    }
    .span-box-body {
        border: 1px solid #E5E7EB; border-radius: 0 0 10px 10px; padding: 20px;
        background: #FFFFFF; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

def render_premium_card(label, value, unit, accent_class):
    return f"""
    <div class="metric-card {accent_class}">
        <div class="metric-label">{label}</div>
        <div><span class="metric-value">{value}</span><span class="metric-unit">{unit}</span></div>
    </div>
    """

def main():
    # --- HEADER ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("🏗️ Professional RC Beam Designer")
        st.markdown(f"**Standard:** ACI 318 Strength Design | **Date:** {datetime.now().strftime('%Y-%m-%d')}")
    with c2:
        st.success("**Engine Status:** ● Online (Matrix Stiffness)")

    st.markdown("---")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Global Settings")
        params = input_handler.render_sidebar()
    
    # --- MAIN LAYOUT ---
    col_input, col_output = st.columns([35, 65], gap="large")
    
    # === LEFT: MODEL INPUT ===
    with col_input:
        st.subheader("1. Structure Definition")
        with st.container(border=True):
            n, spans, sup_df, stable = input_handler.render_model_inputs(params)
        st.markdown("###") 
        with st.container(border=True):
             loads = input_handler.render_loads(n, spans, params)
        st.markdown("###")
        run_btn = st.button("🚀 Run Analysis & Design", type="primary", use_container_width=True, disabled=not stable)
        if not stable: st.error("⚠️ Structure is unstable.")

    # === RIGHT: RESULTS ===
    with col_output:
        if run_btn or st.session_state.get('analysis_done'):
            if run_btn:
                try:
                    with st.spinner("Analyzing and Performing Design Checks..."):
                        engine = solver.BeamSolver(spans, sup_df, loads)
                        df_res, reactions = engine.solve()
                        st.session_state.update({'analysis_done': True, 'df_res': df_res, 'reactions': reactions, 
                                                 'spans': spans, 'sup_df': sup_df, 'loads': loads})
                except Exception as e:
                    st.error(f"Solver Error: {e}")
                    st.stop()

            df = st.session_state['df_res']
            reac = st.session_state['reactions']
            total_len = sum(st.session_state['spans'])

            # --- PREMIUM SUMMARY CARDS ---
            st.subheader("2. Analysis Summary")
            card_html = f"""
            <div class="metric-container">
                {render_premium_card("Max Moment (+)", f"{df['moment'].max():.2f}", "kg-m", "card-blue")}
                {render_premium_card("Max Moment (-)", f"{df['moment'].min():.2f}", "kg-m", "card-red")}
                {render_premium_card("Max Shear", f"{df['shear'].abs().max():.2f}", "kg", "card-cyan")}
                {render_premium_card("Max Reaction", f"{np.max(np.abs(reac[::2])):.2f}", "kg", "card-purple")}
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            # --- PROBE & TABS ---
            c_probe_label, c_probe_slider = st.columns([2, 8])
            with c_probe_label: st.markdown("**🔍 Interactive Probe (x):**")
            with c_probe_slider: probe_val = st.slider("Probe", 0.0, total_len, total_len/2, 0.1, label_visibility="collapsed")

            tab_diag, tab_design, tab_data = st.tabs(["📈 Force Diagrams", "🏗️ Detailed RC Design", "📋 Data Tables"])
            
            with tab_diag:
                design_view.draw_diagrams(df, st.session_state['spans'], st.session_state['sup_df'], 
                                          st.session_state['loads'], params['u_force'], params['u_len'], probe_x=probe_val)
            
            with tab_design:
                st.info("👇 **Design Mode:** Specify section & reinforcement to check capacity (D/C Ratio).")
                n_spans = len(st.session_state['spans'])
                span_props = []
                design_cols = st.columns(n_spans)
                
                for i in range(n_spans):
                    with design_cols[i]:
                        st.markdown(f'<div class="span-box-header">SPAN {i+1}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="span-box-body">', unsafe_allow_html=True)
                        c_b, c_h = st.columns(2)
                        b = c_b.number_input("b (cm)", 15.0, 100.0, 30.0, step=5.0, key=f"b{i}")
                        h = c_h.number_input("h (cm)", 20.0, 200.0, 60.0, step=5.0, key=f"h{i}")
                        st.markdown("---")
                        main_bar = st.selectbox("Bottom Bars", [12, 16, 20, 25, 28], index=2, format_func=lambda x: f"DB{x}", key=f"mb{i}")
                        stirrup = st.selectbox("Stirrups", [6, 9, 10, 12], index=1, format_func=lambda x: f"RB{x}" if x<10 else f"DB{x}", key=f"st{i}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        span_props.append({"b": b, "h": h, "cv": 3.5, "main_bar_dia": main_bar, "stirrup_dia": stirrup})
                
                # Render the new Senior-Level Table
                design_view.render_design_results(df, params, st.session_state['spans'], span_props, st.session_state['sup_df'])

            with tab_data:
                st.dataframe(df, use_container_width=True, height=400)

        else:
            st.info("👈 Define model and click Run to start.")

if __name__ == "__main__":
    main()
