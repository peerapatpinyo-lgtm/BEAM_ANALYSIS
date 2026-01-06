import streamlit as st
import pandas as pd
import numpy as np

# --- Import Custom Modules ---
import input_handler
import design_view
from solver import BeamSolver
import rc_design
import section_plotter 
import file_manager

st.set_page_config(page_title="Professional Beam Studio", layout="wide", page_icon="🏗️")

# ==========================================
# 1. INPUT SECTION
# ==========================================
params = input_handler.render_sidebar()

st.title("🏗️ Professional Beam Studio")
st.caption(f"Section: {params['b']*100:.0f} x {params['h']*100:.0f} cm | I = {params['I']:.2e} m⁴")
st.markdown("---")

n_spans, spans, sup_df, is_stable = input_handler.render_model_inputs(params)
st.markdown("---")
loads_df = input_handler.render_loads(n_spans, spans, params, sup_df)

# ==========================================
# 2. ANALYSIS LOGIC
# ==========================================
st.markdown("### 🚀 Analysis Control")
if st.button("RUN ANALYSIS", type="primary"):
    
    if not is_stable:
        st.error("🚨 Structure is Unstable!")
        st.stop()
    
    if loads_df is None or loads_df.empty:
        st.warning("⚠️ Please add loads.")
        st.stop()

    # --- Load Factoring ---
    raw_loads = loads_df.to_dict('records')
    factored_loads = []
    
    for l in raw_loads:
        factor = params['gamma_dead'] if l['case'] == 'DL' else params['gamma_live']
        f_load = l.copy()
        f_load['mag'] = l['mag'] * factor
        factored_loads.append(f_load)

    # --- Solver ---
    try:
        with st.spinner("Solving Structure..."):
            solver = BeamSolver(
                spans=spans,
                supports_input=sup_df.to_dict('records'),
                loads_input=factored_loads,
                E=params['E'],
                I_custom=params['I']
            )
            
            df_res, reactions, status = solver.solve()
            
            if df_res.empty:
                st.error(f"Analysis Failed: {status.get('error')}")
            else:
                st.session_state.results = {
                    'df': df_res,
                    'reac': reactions,
                    'loads': raw_loads
                }
                st.success("Analysis Complete!")
                
    except Exception as e:
        st.error(f"System Error: {str(e)}")

# ==========================================
# 3. DISPLAY RESULTS
# ==========================================
if 'results' in st.session_state:
    res = st.session_state.results
    
    # 3.1 Deflection Check (Serviceability)
    st.markdown("#### 📏 Serviceability Check (Deflection)")
    cum_dist = [0] + list(np.cumsum(spans))
    
    cols_def = st.columns(n_spans)
    for i in range(n_spans):
        mask = (res['df']['x'] >= cum_dist[i]) & (res['df']['x'] <= cum_dist[i+1])
        span_data = res['df'][mask]
        if not span_data.empty:
            max_def_mm = span_data['deflection'].abs().max() * 1000 
            limit = (spans[i] * 1000) / 240
            status_def = "✅ PASS" if max_def_mm <= limit else "❌ FAIL"
            cols_def[i].metric(
                label=f"Span {i+1} (Limit L/240)",
                value=f"{max_def_mm:.2f} mm",
                delta=status_def,
                delta_color="normal" if "PASS" in status_def else "inverse"
            )
    
    st.divider()

    # 3.2 Diagrams (Fixed: Now Rendering Properly)
    fig_structure = design_view.draw_interactive_diagrams(
        df=res['df'],
        reac=res['reac'],
        spans=spans,
        sup_df=sup_df,
        loads=res['loads'],
        dl_factor=params['gamma_dead'],
        ll_factor=params['gamma_live']
    )
    st.plotly_chart(fig_structure, use_container_width=True)
    
    # 3.3 Reaction Table
    with st.expander("📌 View Reactions"):
        reac_data = []
        for node_idx, val in res['reac'].items():
            val_show = val / 1000 if params['u_force'] == 'kN' else val
            s_type = "Support"
            match = sup_df[sup_df['id'] == node_idx]
            if not match.empty: s_type = match.iloc[0]['type']
            
            reac_data.append({
                "Node": node_idx+1, 
                "Type": s_type,
                f"Ry ({params['u_force']})": f"{val_show:.2f}"
            })
        st.table(pd.DataFrame(reac_data).set_index("Node"))

    # 3.4 RC Design (Professional Detail)
    st.markdown("---")
    st.header("🏗️ Reinforced Concrete Design Details")
    
    with st.expander("🛠️ Design Parameters (Click to Edit)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        fc = c1.number_input("f'c (MPa)", value=24.0)
        fy = c2.number_input("fy (MPa)", value=400.0)
        cover = c3.number_input("Cover (mm)", value=30.0)
        db = c4.selectbox("Main Bar (mm)", [12, 16, 20, 25, 28], index=2)

    df = res['df']
    
    for i in range(n_spans):
        st.markdown(f"### 🌉 Span {i+1}")
        
        # Filter Data
        mask = (df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])
        span_data = df[mask]
        
        if span_data.empty: continue
        
        # Get Forces (Conversion handled inside design functions)
        # Solver gives N-m, we pass raw or convert as needed. 
        # Here we pass raw to rc_design, assume it handles units OR convert before
        # NOTE: design_view handles conversion internally now.
        # But rc_design.design_span_expert typically expects kNm/kN
        
        m_max_pos = span_data['moment'].max() / 1000 # kNm
        m_max_neg = span_data['moment'].min() / 1000 # kNm
        v_max_abs = span_data['shear'].abs().max() / 1000 # kN
        
        # Call Design Function
        design_res = rc_design.design_span_expert(
            m_pos=m_max_pos, m_neg=m_max_neg, v_u=v_max_abs,
            b=params['b'], h=params['h'],
            fc=fc, fy=fy, cover=cover, db=db
        )
        
        # --- Visualization Section ---
        with st.container(border=True):
            col_viz, col_data = st.columns([1, 1])
            
            with col_viz:
                st.write("**📊 Moment Capacity Check**")
                # เรียกใช้กราฟ Capacity (Updated in design_view)
                fig_cap = design_view.plot_capacity_vs_demand(
                    df_span=span_data,
                    phi_Mn_pos=design_res['pos']['capacity'],
                    phi_Mn_neg=design_res['neg']['capacity']
                )
                st.plotly_chart(fig_cap, use_container_width=True)

            with col_data:
                st.write("**Cross Section**")
                # ส่งค่า fc, fy เข้าไปใน section_plotter
                fig_sec = section_plotter.plot_section(
                    b=params['b'], h=params['h'], 
                    cover_mm=cover, db_mm=db, 
                    n_top=design_res['neg']['n'], n_bot=design_res['pos']['n'],
                    stirrup_info=design_res['shear_stirrups'],
                    fc=fc, fy=fy
                )
                st.pyplot(fig_sec, use_container_width=True)

        # --- Technical Data Tabs ---
        tab1, tab2, tab3 = st.tabs(["💪 Strength", "⚓ Detailing", "🔍 Serviceability"])
        
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                 st.info(f"**Top Steel (-M):**\n\n{design_res['neg']['n']} - DB{db}")
                 st.caption(f"Capacity: {design_res['neg']['capacity']:.2f} kNm\nDemand: {abs(m_max_neg):.2f} kNm")
            with c2:
                 st.success(f"**Bot Steel (+M):**\n\n{design_res['pos']['n']} - DB{db}")
                 st.caption(f"Capacity: {design_res['pos']['capacity']:.2f} kNm\nDemand: {m_max_pos:.2f} kNm")
            
            st.write("---")
            st.write(f"**Shear Reinforcement:** {design_res['shear_stirrups']}")
            st.caption(f"Status: {design_res['shear_status']}")

        with tab2:
            st.write("##### Development Length & Splices")
            st.write(f"**Top Bars (Zone -M):** $L_d$ = {design_res['neg']['Ld']/1000:.2f} m, Lap Splice = {design_res['neg']['Ls']/1000:.2f} m")
            st.write(f"**Bot Bars (Zone +M):** $L_d$ = {design_res['pos']['Ld']/1000:.2f} m, Lap Splice = {design_res['pos']['Ls']/1000:.2f} m")

        with tab3:
            st.write("##### Crack Width Control (ACI 318)")
            st.write(f"**Top Surface:** {'✅ OK' if design_res['neg']['crack_ok'] else '⚠️ Check Spacing'}")
            st.write(f"**Bottom Surface:** {'✅ OK' if design_res['pos']['crack_ok'] else '⚠️ Check Spacing'}")
