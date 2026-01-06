import streamlit as st
import pandas as pd
import numpy as np

# --- Import Custom Modules ---
import input_handler
import design_view
from solver import BeamSolver
import rc_design

# --- Page Config ---
st.set_page_config(page_title="Professional Beam Studio", layout="wide", page_icon="🏗️")

# --- CSS Styling (Optional) ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    .reportview-container .main .block-container { max-width: 1200px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. INPUT SECTION (via input_handler)
# ==========================================
# 1.1 Sidebar Config
params = input_handler.render_sidebar()

st.title("🏗️ Professional Beam Studio")
st.caption("Advanced Structural Analysis & RC Design System")
st.markdown("---")

# 1.2 Geometry Inputs
n_spans, spans, sup_df, is_stable = input_handler.render_model_inputs(params)

# 1.3 Load Inputs
st.markdown("---")
loads_df = input_handler.render_loads(n_spans, spans, params, sup_df)

# ==========================================
# 2. ANALYSIS LOGIC
# ==========================================
st.markdown("### 🚀 Analysis Control")
col_act, col_info = st.columns([1, 4])

if col_act.button("RUN ANALYSIS", type="primary"):
    
    # Validation
    if not is_stable:
        st.error("🚨 Structure is Unstable! Please check supports.")
        st.stop()
    
    if loads_df is None or loads_df.empty:
        st.warning("⚠️ Please add at least one load.")
        st.stop()

    # --- Step A: Load Factoring (DL/LL) ---
    # เตรียมข้อมูลโหลดส่งให้ Solver โดยคูณ Load Factors
    raw_loads = loads_df.to_dict('records')
    factored_loads = []
    
    for l in raw_loads:
        factor = params['gamma_dead'] if l['case'] == 'DL' else params['gamma_live']
        f_load = l.copy()
        f_load['mag'] = l['mag'] * factor # Apply Factor Here
        factored_loads.append(f_load)

    # --- Step B: Solve Structure ---
    try:
        with st.spinner("Solving Stiffness Matrix..."):
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
                # Save to Session State
                st.session_state.results = {
                    'df': df_res,
                    'reac': reactions,
                    'loads': raw_loads,
                    'factored_loads': factored_loads
                }
                st.success("Analysis Complete!")
                
    except Exception as e:
        st.error(f"System Error: {str(e)}")

# ==========================================
# 3. DISPLAY RESULTS (via design_view)
# ==========================================
if 'results' in st.session_state:
    res = st.session_state.results
    
    st.markdown("---")
    
    # 3.1 Reaction Table
    st.subheader("📌 Support Reactions (Factored)")
    
    reac_data = []
    for node_idx, val in res['reac'].items():
        # Find support type name
        s_type = "Support"
        match = sup_df[sup_df['id'] == node_idx]
        if not match.empty: s_type = match.iloc[0]['type']
        
        # Unit conversion
        val_show = val / 1000 if params['u_force'] == 'kN' else val
        
        reac_data.append({
            "Node": node_idx + 1,
            "Type": s_type,
            f"Ry ({params['u_force']})": f"{val_show:.2f}"
        })
    
    st.table(pd.DataFrame(reac_data).set_index("Node"))

    # 3.2 Diagrams (SFD, BMD, Deflection)
    design_view.draw_interactive_diagrams(
        df=res['df'],
        reac=res['reac'],
        spans=spans,
        sup_df=sup_df,
        loads=res['loads'], # Show Raw loads in diagram annotation
        dl_factor=params['gamma_dead'],
        ll_factor=params['gamma_live']
    )

    # ==========================================
    # 4. RC DESIGN MODULE (via rc_design)
    # ==========================================
    st.markdown("---")
    st.header("🏗️ Reinforced Concrete Design")
    
    with st.expander("🛠️ RC Parameters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        fc = c1.number_input("f'c (MPa)", value=24.0)
        fy = c2.number_input("fy (MPa)", value=400.0)
        cover = c3.number_input("Cover (mm)", value=30.0)
        db = c4.selectbox("Main Bar (mm)", [12, 16, 20, 25, 28], index=2)
        
        c5, c6 = st.columns(2)
        b_section = c5.number_input("Width b (cm)", value=25.0) / 100 # m
        h_section = c6.number_input("Depth h (cm)", value=50.0) / 100 # m

    # Loop Design per Span
    df = res['df']
    cum_dist = [0] + list(np.cumsum(spans))
    
    for i in range(n_spans):
        st.markdown(f"#### Span {i+1} Design")
        
        # Filter forces in this span
        mask = (df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])
        span_data = df[mask]
        
        if span_data.empty: continue
        
        # Get Envelope Forces (Unit: N, Nm)
        m_max_pos = span_data['moment'].max() # +Moment (Bottom Steel)
        m_max_neg = span_data['moment'].min() # -Moment (Top Steel)
        v_max_abs = span_data['shear'].abs().max()
        
        # Convert to kN, kNm for Design Function
        design_res = rc_design.design_span_expert(
            m_pos=m_max_pos / 1000,
            m_neg=m_max_neg / 1000,
            v_u=v_max_abs / 1000,
            b=b_section, h=h_section,
            fc=fc, fy=fy, cover=cover, db=db
        )
        
        # Display Card
        with st.container(border=True):
            col_res1, col_res2, col_res3 = st.columns([1, 1, 1])
            
            with col_res1:
                st.caption("Positive Moment (Mid-Span)")
                st.markdown(f"**Mu+ :** {m_max_pos/1000:.2f} kNm")
                st.success(f"bot: **{design_res['pos']['n']} - DB{db}**")
            
            with col_res2:
                st.caption("Negative Moment (Support)")
                st.markdown(f"**Mu- :** {m_max_neg/1000:.2f} kNm")
                st.error(f"top: **{design_res['neg']['n']} - DB{db}**")
                
            with col_res3:
                st.caption("Shear Check")
                st.markdown(f"**Vu :** {v_max_abs/1000:.2f} kN")
                status = design_res['shear_status']
                if "OK" in status:
                    st.info(f"Shear: {status}")
                else:
                    st.warning(f"Shear: {status}")
