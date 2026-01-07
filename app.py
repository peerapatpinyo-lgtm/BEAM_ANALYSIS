
import streamlit as st
import pandas as pd
import numpy as np

# Import Local Modules
import input_handler
import solver
import design_view
import rc_design
import section_plotter

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Pro Beam Studio",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Textbook look
st.markdown("""
<style>
    .block-container { max-width: 1200px; padding-top: 2rem; }
    h1, h2, h3, h4 { font-family: 'Helvetica', sans-serif; color: #2C3E50; }
    .stAlert { padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ Professional RC Beam Studio")
st.markdown("---")

# ==========================================
# 2. SIDEBAR & INPUTS
# ==========================================
params = input_handler.render_sidebar()

if 'fc' not in params:
    est_fc = (params['E'] / 4700)**2
    params['fc'] = float(np.clip(est_fc, 20, 35)) # Clip to normal range

with st.sidebar:
    st.markdown("---")
    st.subheader("4. Rebar Strength")
    params['fy'] = st.number_input("Yield Strength (fy)", value=400.0, step=10.0, format="%.1f")
    params['cover'] = st.slider("Cover (mm)", 20, 75, 25)
    params['db_main'] = st.selectbox("Main Bar DB (mm)", [12, 16, 20, 25, 28, 32], index=1)

n_spans, spans, sup_df, stable = input_handler.render_model_inputs(params)
loads_df = input_handler.render_loads(n_spans, spans, params, sup_df)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if not stable:
    st.warning("⚠️ Structure Unstable. Please add at least 2 supports (or 1 Fixed).")
else:
    if st.button("🚀 Run Analysis & Design", type="primary"):
        with st.spinner("Processing..."):
            # Prepare & Solve
            sup_list = sup_df.to_dict('records') if not sup_df.empty else []
            load_list = loads_df.to_dict('records') if loads_df is not None else []
            
            # Factored Loads
            factored_loads = []
            for l in load_list:
                f = params['gamma_dead'] if l['case'] == 'DL' else params['gamma_live']
                new_l = l.copy()
                new_l['mag'] = l['mag'] * f
                factored_loads.append(new_l)

            beam_solver = solver.BeamSolver(spans, sup_list, factored_loads, params['E'], params['b'], params['h'], params['I'])
            df_res, reac, eq_check = beam_solver.solve()
            
            if df_res.empty:
                st.error("Solver Error.")
                st.stop()

        st.success("Analysis Complete!")
        
        # --- TABS ---
        tab1, tab2 = st.tabs(["📊 Analysis Results", "🏗️ RC Design Detailing"])
        
        # --- TAB 1: Analysis ---
        with tab1:
            st.markdown("#### Internal Forces Diagrams")
            fig_diagram = design_view.draw_interactive_diagrams(df_res, reac, spans, sup_df, load_list)
            st.plotly_chart(fig_diagram, use_container_width=True)
            
            st.markdown("#### Support Reactions")
            reac_data = [{"Node": k, "Rx (kN)": 0, "Ry (kN)": v/1000.0, "Mz (kNm)": 0} for k, v in reac.items()]
            st.dataframe(pd.DataFrame(reac_data).set_index("Node").T)

        # --- TAB 2: Design ---
        with tab2:
            st.markdown("### Reinforced Concrete Design (ACI 318 / EIT)")
            
            # 1. Collect Data
            cum_dist = [0] + list(np.cumsum(spans))
            all_span_designs = []
            
            for i in range(n_spans):
                start_x = cum_dist[i]
                end_x = cum_dist[i+1]
                mask = (df_res['x'] >= start_x) & (df_res['x'] <= end_x)
                span_res = df_res[mask]
                
                m_pos = max(0, span_res['moment'].max() / 1000.0)
                m_neg = span_res['moment'].min() / 1000.0
                v_u = span_res['shear'].abs().max() / 1000.0
                
                design_res = rc_design.design_span_expert(
                    m_pos, m_neg, v_u, 
                    params['b'], params['h'], params['fc'], params['fy'], 
                    params['cover'], params['db_main']
                )
                design_res['db'] = params['db_main']
                all_span_designs.append(design_res)

            # 2. LONGITUDINAL SECTION (Full Width)
            st.markdown("#### 📐 Longitudinal Elevation")
            fig_long = section_plotter.plot_longitudinal_section(
                spans, sup_df, all_span_designs, params['h'], params['cover']
            )
            st.pyplot(fig_long)
            
            st.markdown("---")
            
            # 3. CROSS SECTIONS (Span by Span)
            st.markdown("#### 🔍 Span Details")
            
            for i in range(n_spans):
                d = all_span_designs[i]
                with st.container():
                    col_info, col_img = st.columns([1.2, 1])
                    
                    with col_info:
                        st.markdown(f"**SPAN {i+1}** (L={spans[i]}m)")
                        
                        # Data Table
                        res_data = {
                            "Location": ["Midspan (+)", "Support (-)"],
                            "Design Moment": [f"{d['pos']['capacity']:.1f} kNm", f"{d['neg']['capacity']:.1f} kNm"],
                            "Rebar": [f"{d['pos']['n']}-DB{params['db_main']}", f"{d['neg']['n']}-DB{params['db_main']}"],
                            "Status": ["✅ OK" if "OK" in d['pos']['note'] else "⚠️ Check", "✅ OK" if "OK" in d['neg']['note'] else "⚠️ Check"]
                        }
                        st.table(pd.DataFrame(res_data))
                        st.info(f"🧱 Shear Design: **{d['shear_stirrups']}**")

                    with col_img:
                        # Cross Section
                        fig_sec = section_plotter.plot_section(
                            params['b'], params['h'], params['cover'], params['db_main'],
                            n_top=d['neg']['n'], # Representative Top
                            n_bot=d['pos']['n'], # Representative Bot
                            stirrup_info=d['shear_stirrups'],
                            fc=params['fc'], fy=params['fy']
                        )
                        # จัดกลางและไม่ขยายจนแตก
                        st.pyplot(fig_sec, use_container_width=False)
                
                st.divider()
