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

st.title("🏗️ Professional RC Beam Studio")
st.markdown("---")

# ==========================================
# 2. SIDEBAR & INPUTS
# ==========================================
params = input_handler.render_sidebar()

# Auto-calc fc from E if not present (Simple estimation)
if 'fc' not in params:
    est_fc = (params['E'] / 4700)**2
    if est_fc < 15: est_fc = 20
    if est_fc > 50: est_fc = 35
    params['fc'] = float(int(est_fc))

with st.sidebar:
    st.markdown("---")
    st.subheader("4. Rebar Strength")
    params['fy'] = st.number_input("Yield Strength (fy)", value=400.0, step=10.0, format="%.1f")
    st.caption(f"Using fc' ≈ {params['fc']:.1f} MPa")
    params['cover'] = st.slider("Cover (mm)", 20, 75, 25)
    params['db_main'] = st.selectbox("Main Bar DB (mm)", [12, 16, 20, 25, 28, 32], index=1)

# Render Geometry & Loads
n_spans, spans, sup_df, stable = input_handler.render_model_inputs(params)
loads_df = input_handler.render_loads(n_spans, spans, params, sup_df)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if not stable:
    st.error("⚠️ Structure is Unstable! Please add supports.")
else:
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Analyzing..."):
            # Prepare Solver Data
            sup_list = sup_df.to_dict('records') if not sup_df.empty else []
            load_list = loads_df.to_dict('records') if loads_df is not None else []
            
            # Factored Loads
            factored_loads = []
            for l in load_list:
                f = params['gamma_dead'] if l['case'] == 'DL' else params['gamma_live']
                new_l = l.copy()
                new_l['mag'] = l['mag'] * f
                factored_loads.append(new_l)

            # Solve
            beam_solver = solver.BeamSolver(spans, sup_list, factored_loads, params['E'], params['b'], params['h'], params['I'])
            df_res, reac, eq_check = beam_solver.solve()
            
            if df_res.empty:
                st.error("Solver Error.")
                st.stop()

        st.success("✅ Analysis Complete!")
        
        # --- TABS ---
        tab1, tab2 = st.tabs(["📊 Analysis Diagrams", "🏗️ RC Design & Detailing"])
        
        with tab1:
            st.subheader("Shear, Moment & Deflection")
            fig_diagram = design_view.draw_interactive_diagrams(df_res, reac, spans, sup_df, load_list)
            st.plotly_chart(fig_diagram, use_container_width=True)
            
            st.markdown("#### Support Reactions (Factored)")
            reac_data = [{"Node": k, "R (kN)": v/1000.0} for k, v in reac.items()]
            st.dataframe(pd.DataFrame(reac_data).set_index("Node").T)

        with tab2:
            st.subheader("Reinforced Concrete Design (ACI 318)")
            
            # --- 1. COLLECT DESIGN DATA FOR LONGITUDINAL PLOT ---
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
                design_res['db'] = params['db_main'] # Add DB info for plotter
                all_span_designs.append(design_res)

            # --- 2. DRAW LONGITUDINAL SECTION (TOP) ---
            st.markdown("### 📐 Overall Beam Reinforcement")
            fig_long = section_plotter.plot_longitudinal_section(
                spans, sup_df, all_span_designs, params['h'], params['cover']
            )
            st.pyplot(fig_long)
            st.markdown("---")

            # --- 3. DRAW CROSS SECTIONS PER SPAN ---
            for i in range(n_spans):
                d_res = all_span_designs[i]
                
                col_info, col_plot = st.columns([1, 1])
                
                with col_info:
                    st.markdown(f"#### Span {i+1} Detail")
                    st.info(f"**Positive Zone (Midspan)**\n- Moment: {d_res['pos']['capacity']:.2f} kNm\n- Rebar: {d_res['pos']['n']} - DB{params['db_main']}")
                    st.warning(f"**Negative Zone (Support)**\n- Moment: {d_res['neg']['capacity']:.2f} kNm\n- Rebar: {d_res['neg']['n']} - DB{params['db_main']}")
                    st.error(f"**Shear Stirrups**\n- {d_res['shear_stirrups']}")

                with col_plot:
                    # Plot Section
                    fig_sec = section_plotter.plot_section(
                        params['b'], params['h'], params['cover'], params['db_main'],
                        n_top=d_res['neg']['n'], # Show Max Top Bars
                        n_bot=d_res['pos']['n'], # Show Max Bot Bars
                        stirrup_info=d_res['shear_stirrups'],
                        fc=params['fc'], fy=params['fy']
                    )
                    st.pyplot(fig_sec)
                
                st.divider()
