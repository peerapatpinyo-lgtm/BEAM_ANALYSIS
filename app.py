import streamlit as st
import pandas as pd
import numpy as np
import solver
import input_handler
import design_view
from datetime import datetime

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="RC Beam Pro", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
        color: #212529;
    }
    
    /* Header Styles */
    h1 { color: #0D47A1; font-weight: 700; border-bottom: 2px solid #0D47A1; padding-bottom: 10px; }
    h2, h3 { color: #1565C0; font-weight: 600; }
    
    /* Input Highlights */
    .stNumberInput label, .stSelectbox label { color: #1565C0 !important; font-weight: bold; }
    
    /* Card Styling */
    .metric-card {
        background: white; border: 1px solid #E0E0E0; border-radius: 8px; padding: 15px;
        text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #0D47A1; }
    
    /* Custom Container for Span Inputs */
    .span-box {
        background-color: #F1F8E9;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #C5E1A5;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

def display_metric_card(col, label, value, unit, color="#0D47A1"):
    col.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.9rem; color: #616161;">{label}</div>
        <div class="metric-value" style="color: {color};">{value} <span style="font-size: 1rem;">{unit}</span></div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # --- HEADER ---
    with st.container():
        c1, c2 = st.columns([3, 1])
        with c1:
            st.title("🏗️ Professional RC Beam Design")
            st.markdown(f"**Date:** {datetime.now().strftime('%d %B %Y')} | **Engineer:** Senior Eng. Team")
        with c2:
            st.success("✅ **System Status:** Ready\n\nEngine: Matrix Stiffness Method")

    st.markdown("---")

    # --- SIDEBAR (Global Material Properties) ---
    with st.sidebar:
        st.header("⚙️ Project & Materials")
        # เรียก Input เดิม แต่เราจะดึงแค่ parameters ที่เป็น Global (fc, fy)
        # ส่วนเรื่อง Geometry เราจะจัดการเองใน Main Area
        params = input_handler.render_sidebar()
        
    
    # --- MAIN LAYOUT ---
    col_input, col_output = st.columns([1, 2], gap="large")
    
    # === LEFT: MODEL INPUT ===
    with col_input:
        st.subheader("1. Structural Model")
        with st.container(border=True):
            n, spans, sup_df, stable = input_handler.render_model_inputs(params)
            loads = input_handler.render_loads(n, spans, params)
            
            st.markdown("###")
            run_btn = st.button("🚀 Analyze Structure", type="primary", use_container_width=True, disabled=not stable)

            if not stable:
                st.error("⚠️ Structure Unstable")

    # === RIGHT: RESULTS ===
    with col_output:
        # Check if analysis exists
        if run_btn or st.session_state.get('analysis_done'):
            if run_btn:
                try:
                    with st.spinner("Calculating..."):
                        engine = solver.BeamSolver(spans, sup_df, loads)
                        df_res, reactions = engine.solve()
                        
                        st.session_state['analysis_done'] = True
                        st.session_state['df_res'] = df_res
                        st.session_state['reactions'] = reactions
                        st.session_state['spans'] = spans
                        st.session_state['sup_df'] = sup_df
                        st.session_state['loads'] = loads
                except Exception as e:
                    st.error(f"Solver Error: {e}")
                    st.stop()

            # Retrieve Data
            df = st.session_state['df_res']
            reac = st.session_state['reactions']

            # --- EXECUTIVE SUMMARY ---
            st.subheader("2. Analysis Summary")
            m1, m2, m3, m4 = st.columns(4)
            
            # Helper to find absolute max values
            display_metric_card(m1, "Max +Moment", f"{df['moment'].max():.2f}", "kg-m", "#2E7D32")
            display_metric_card(m2, "Max -Moment", f"{df['moment'].min():.2f}", "kg-m", "#C62828")
            display_metric_card(m3, "Max Shear", f"{df['shear'].abs().max():.2f}", "kg", "#EF6C00")
            display_metric_card(m4, "Max Reaction", f"{np.max(np.abs(reac[::2])):.2f}", "kg", "#1565C0")
            
            st.markdown("###")

            # --- TABS ---
            tab_diag, tab_reac, tab_design = st.tabs(["📈 Diagrams", "📋 Reactions", "🏗️ RC Design"])
            
            with tab_diag:
                design_view.draw_diagrams(df, st.session_state['spans'], st.session_state['sup_df'], 
                                          st.session_state['loads'], params['u_force'], params['u_len'])
            
            with tab_reac:
                r_data = []
                for i in range(len(st.session_state['spans']) + 1):
                    r_data.append({
                        "Support Node": f"#{i+1}",
                        "Vertical (Ry) [kg]": f"{reac[2*i]:.2f}",
                        "Moment (Mz) [kg-m]": f"{reac[2*i+1]:.2f}"
                    })
                st.dataframe(pd.DataFrame(r_data), use_container_width=True, hide_index=True)

            # --- RC DESIGN TAB (UPDATED) ---
            with tab_design:
                st.info("👇 **Design Specification:** Adjust beam size & rebar for each span independently.")
                
                n_spans = len(st.session_state['spans'])
                span_props = []
                
                # Create Columns for each span
                d_cols = st.columns(n_spans)
                
                for i in range(n_spans):
                    with d_cols[i]:
                        # Styling for distinct look
                        st.markdown(f"""<div style="background:#E3F2FD; padding:10px; border-radius:5px; border-left: 5px solid #1565C0;">
                            <b>SPAN {i+1}</b> (L = {st.session_state['spans'][i]} m)</div>""", unsafe_allow_html=True)
                        
                        st.markdown("##### 📏 Section")
                        b = st.number_input(f"Width b (cm)", 15.0, 100.0, 25.0, step=5.0, key=f"b_{i}")
                        h = st.number_input(f"Depth h (cm)", 20.0, 200.0, 50.0, step=5.0, key=f"h_{i}")
                        
                        st.markdown("##### ⛓️ Rebar")
                        # Select Box for Main Bar
                        main_bar = st.selectbox(f"Main Bar", options=[12, 16, 20, 25, 28], index=1, 
                                              format_func=lambda x: f"DB{x}", key=f"mb_{i}")
                        
                        # Select Box for Stirrup
                        stirrup = st.selectbox(f"Stirrup", options=[6, 9], index=0, 
                                             format_func=lambda x: f"RB{x}" if x<10 else f"DB{x}", key=f"st_{i}")
                        
                        span_props.append({
                            "b": b, "h": h, "cv": 3.0, 
                            "main_bar_dia": main_bar, 
                            "stirrup_dia": stirrup
                        })
                
                st.markdown("---")
                st.subheader("📝 Design Calculation Results")
                
                # Pass data to design view
                design_view.render_design_results(df, params, st.session_state['spans'], span_props, st.session_state['sup_df'])

        else:
            st.info("👈 Please define model and Click 'Analyze Structure'")

if __name__ == "__main__":
    main()
