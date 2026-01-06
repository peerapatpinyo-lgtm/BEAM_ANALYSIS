import streamlit as st
import pandas as pd
import numpy as np

# --- Import Custom Modules ---
import input_handler
import design_view
from solver import BeamSolver
import rc_design
import section_plotter  # <--- เพิ่มบรรทัดนี้ เพื่อเรียกใช้ไฟล์วาดรูป

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
                I_custom=params['I'] # Uses Calculated I from Sidebar
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
        # Find Max Deflection in Span
        mask = (res['df']['x'] >= cum_dist[i]) & (res['df']['x'] <= cum_dist[i+1])
        span_data = res['df'][mask]
        if not span_data.empty:
            max_def_mm = span_data['deflection'].abs().max() * 1000 # convert to mm
            
            # Limit L/240
            limit = (spans[i] * 1000) / 240
            
            status_def = "✅ PASS" if max_def_mm <= limit else "❌ FAIL"
            cols_def[i].metric(
                label=f"Span {i+1} (Limit L/240 = {limit:.1f}mm)",
                value=f"{max_def_mm:.2f} mm",
                delta=status_def,
                delta_color="normal" if "PASS" in status_def else "inverse"
            )
    
    st.divider()

    # 3.2 Diagrams
    design_view.draw_interactive_diagrams(
        df=res['df'],
        reac=res['reac'],
        spans=spans,
        sup_df=sup_df,
        loads=res['loads'],
        dl_factor=params['gamma_dead'],
        ll_factor=params['gamma_live']
    )
    
    # 3.3 Reaction Table
    st.subheader("📌 Reactions")
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

    # 3.4 RC Design (Detailed with Visualization)
    st.markdown("---")
    st.header("🏗️ Reinforced Concrete Design")
    
    with st.expander("🛠️ RC Parameters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        fc = c1.number_input("f'c (MPa)", value=24.0)
        fy = c2.number_input("fy (MPa)", value=400.0)
        cover = c3.number_input("Cover (mm)", value=30.0)
        db = c4.selectbox("Main Bar (mm)", [12, 16, 20, 25, 28], index=2)

    # Loop Design per Span
    df = res['df']
    
    for i in range(n_spans):
        st.markdown(f"#### Span {i+1}")
        mask = (df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])
        span_data = df[mask]
        
        if span_data.empty: continue
        
        m_max_pos = span_data['moment'].max() / 1000
        m_max_neg = span_data['moment'].min() / 1000
        v_max_abs = span_data['shear'].abs().max() / 1000
        
        # Call Improved RC Design Function
        design_res = rc_design.design_span_expert(
            m_pos=m_max_pos, m_neg=m_max_neg, v_u=v_max_abs,
            b=params['b'], h=params['h'],
            fc=fc, fy=fy, cover=cover, db=db
        )
        
        # --- Display Results Card ---
        with st.container(border=True):
            col_txt, col_img = st.columns([2, 1])
            
            with col_txt:
                c_bot, c_top = st.columns(2)
                with c_bot:
                    st.caption("Bottom Steel (+M)")
                    st.info(f"🔵 {design_res['pos']['n']} - DB{db}")
                    st.markdown(f"*{design_res['pos']['note']}*")
                    
                with c_top:
                    st.caption("Top Steel (-M)")
                    st.warning(f"🔴 {design_res['neg']['n']} - DB{db}")
                    st.markdown(f"*{design_res['neg']['note']}*")
                
                st.divider()
                st.caption("Shear Reinforcement")
                st.success(f"⛓️ {design_res['shear_stirrups']}")
                st.write(f"Status: **{design_res['shear_status']}**")

            # --- Visual Section (ส่วนแสดงรูปภาพ) ---
            with col_img:
                st.caption("Cross Section Preview")
                fig_sec = section_plotter.plot_section(
                    b=params['b'], 
                    h=params['h'], 
                    cover_mm=cover, 
                    db_mm=db, 
                    n_top=design_res['neg']['n'], 
                    n_bot=design_res['pos']['n'],
                    stirrup_info=design_res['shear_stirrups']
                )
                st.pyplot(fig_sec, use_container_width=True)
