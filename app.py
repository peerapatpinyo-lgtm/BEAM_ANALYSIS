import streamlit as st
import pandas as pd
import numpy as np
import solver
import input_handler
import design_view
from datetime import datetime

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Pro RC Beam Designer", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&family=Roboto+Mono:wght@400;700&display=swap');
    
    /* General Theme */
    html, body, [class*="css"] {
        font-family: 'Kanit', sans-serif;
        color: #1F2937;
    }
    
    h1, h2, h3 { color: #111827; font-weight: 600; letter-spacing: -0.5px; }
    
    /* Custom Card Style */
    .metric-container {
        display: flex;
        gap: 15px;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 20px 15px;
        flex: 1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        border-top: 4px solid #E5E7EB;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .metric-label { font-size: 0.85rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
    .metric-value { font-size: 1.6rem; font-weight: 700; margin-top: 5px; font-family: 'Roboto Mono', monospace; }
    .metric-unit { font-size: 0.9rem; color: #9CA3AF; font-weight: 400; }
    
    /* Highlight Colors for Cards */
    .border-blue { border-top-color: #3B82F6 !important; }   /* Reaction */
    .border-green { border-top-color: #10B981 !important; }  /* Positive M */
    .border-red { border-top-color: #EF4444 !important; }    /* Negative M */
    .border-orange { border-top-color: #F59E0B !important; } /* Shear */

    /* Input & Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px;
        background-color: #F3F4F6;
        border: none;
        color: #4B5563;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #EFF6FF;
        color: #2563EB;
    }
    
    /* Span Box Styling */
    .span-box-header {
        background: #F8FAFC;
        padding: 10px 15px;
        border-radius: 8px 8px 0 0;
        border: 1px solid #E2E8F0;
        border-bottom: none;
        font-weight: 600;
        color: #334155;
    }
    .span-box-body {
        border: 1px solid #E2E8F0;
        border-radius: 0 0 8px 8px;
        padding: 15px;
        margin-bottom: 15px;
        background: white;
    }
    
</style>
""", unsafe_allow_html=True)

def render_metric_card(label, value, unit, color_class):
    return f"""
    <div class="metric-card {color_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-unit">{unit}</div>
    </div>
    """

def main():
    # --- HEADER ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("🏗️ Structural Beam Analysis")
        st.markdown(f"<span style='color:#6B7280'>Professional RC Design Suite | Date: {datetime.now().strftime('%d %b %Y')}</span>", unsafe_allow_html=True)
    with c2:
        st.caption("Engine Status")
        st.success("● Matrix Solver Ready")

    st.markdown("---")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Settings")
        params = input_handler.render_sidebar()
    
    # --- MAIN LAYOUT ---
    col_input, col_output = st.columns([35, 65], gap="large")
    
    # === LEFT: MODEL INPUT ===
    with col_input:
        st.subheader("1. Model Definition")
        
        with st.container(border=True):
            st.markdown("##### 📏 Geometry")
            n, spans, sup_df, stable = input_handler.render_model_inputs(params)
        
        st.markdown("###") # Spacing
        
        with st.container(border=True):
            st.markdown("##### ⬇️ Loading")
            loads = input_handler.render_loads(n, spans, params)
            
        st.markdown("###")
        
        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=not stable)
        if not stable:
            st.warning("⚠️ Structure is unstable. Please check supports.")

    # === RIGHT: RESULTS ===
    with col_output:
        if run_btn or st.session_state.get('analysis_done'):
            if run_btn:
                try:
                    with st.spinner("Solving Matrix Equations..."):
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

            # --- EXECUTIVE SUMMARY CARDS ---
            st.subheader("2. Analysis Results")
            
            # HTML Injection for Custom Cards
            card_html = f"""
            <div class="metric-container">
                {render_metric_card("Max Moment (+)", f"{df['moment'].max():.2f}", "kg-m", "border-green")}
                {render_metric_card("Max Moment (-)", f"{df['moment'].min():.2f}", "kg-m", "border-red")}
                {render_metric_card("Max Shear", f"{df['shear'].abs().max():.2f}", "kg", "border-orange")}
                {render_metric_card("Max Reaction", f"{np.max(np.abs(reac[::2])):.2f}", "kg", "border-blue")}
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            st.markdown("###")

            # --- TABS ---
            tab_diag, tab_design, tab_data = st.tabs(["📈 Force Diagrams", "🏗️ RC Design", "📋 Data & React."])
            
            with tab_diag:
                # เรียกกราฟที่ปรับปรุงความสวยงามแล้ว
                design_view.draw_diagrams(df, st.session_state['spans'], st.session_state['sup_df'], 
                                          st.session_state['loads'], params['u_force'], params['u_len'])
            
            with tab_design:
                st.info("💡 **Design Mode:** Configure section and reinforcement for each span.")
                
                n_spans = len(st.session_state['spans'])
                span_props = []
                
                # Use a cleaner loop with custom HTML/CSS containers
                design_cols = st.columns(n_spans)
                
                for i in range(n_spans):
                    with design_cols[i]:
                        st.markdown(f'<div class="span-box-header">SPAN {i+1} <span style="font-weight:normal; font-size:0.8em; color:#64748B;">(L={st.session_state["spans"][i]}m)</span></div>', unsafe_allow_html=True)
                        
                        # Container Start
                        st.markdown('<div class="span-box-body">', unsafe_allow_html=True)
                        
                        st.caption("Concrete Section (cm)")
                        c_b, c_h = st.columns(2)
                        b = c_b.number_input("b", 15.0, 100.0, 25.0, step=5.0, key=f"b_{i}", label_visibility="collapsed")
                        h = c_h.number_input("h", 20.0, 200.0, 50.0, step=5.0, key=f"h_{i}", label_visibility="collapsed")
                        
                        st.caption("Reinforcement")
                        main_bar = st.selectbox("Main", [12, 16, 20, 25, 28], index=1, format_func=lambda x: f"DB{x}", key=f"mb_{i}")
                        stirrup = st.selectbox("Stirrup", [6, 9], index=0, format_func=lambda x: f"RB{x}" if x<10 else f"DB{x}", key=f"st_{i}")
                        
                        # Container End
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        span_props.append({
                            "b": b, "h": h, "cv": 3.0, 
                            "main_bar_dia": main_bar, 
                            "stirrup_dia": stirrup
                        })
                
                # Render Design Table
                design_view.render_design_results(df, params, st.session_state['spans'], span_props, st.session_state['sup_df'])

            with tab_data:
                st.markdown("#### Reaction Forces")
                r_data = []
                for i in range(len(st.session_state['spans']) + 1):
                    r_data.append({
                        "Node ID": i+1,
                        "Vertical (Ry)": f"{reac[2*i]:.2f}",
                        "Moment (Mz)": f"{reac[2*i+1]:.2f}"
                    })
                st.dataframe(pd.DataFrame(r_data), use_container_width=True, hide_index=True)

        else:
            # Empty State
            st.markdown("""
            <div style="text-align: center; padding: 60px; background: #F9FAFB; border-radius: 10px; color: #9CA3AF;">
                <h3>Waiting for Input</h3>
                <p>Define structure geometry and loads on the left panel to begin.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
