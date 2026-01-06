import streamlit as st
import pandas as pd
from solver import BeamSolver
import rc_design
import design_view # กราฟที่คุณเขียนไว้เดิม

st.set_page_config(page_title="Professional Beam Structural Analyzer", layout="wide")

if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state.loads = []

with st.sidebar:
    st.header("⚙️ Design Parameters")
    fc = st.number_input("fc' (Concrete Strength, MPa)", 25.0)
    fy = st.number_input("fy (Steel Yield, MPa)", 400.0) # เพิ่มต่อจาก fc' ตามสั่ง
    
    st.divider()
    b = st.number_input("Width b (m)", 0.3)
    h = st.number_input("Height h (m)", 0.5)
    
    st.write("**Moment of Inertia (I)**")
    i_mode = st.radio("I Calc", ["Auto", "Manual"], horizontal=True)
    I_val = st.number_input("I (m⁴)", value=(b*h**3)/12, format="%.6e") if i_mode == "Manual" else (b*h**3)/12
    
    if st.button("🗑️ Reset All", use_container_width=True):
        st.session_state.loads = []
        st.rerun()

st.title("🏗️ Beam Analysis & Design Master")

t1, t2 = st.tabs(["📐 Geometry & Supports", "📥 Load Definition"])

with t1:
    c1, c2 = st.columns([1, 2])
    with c1:
        n_spans = st.number_input("Spans", 1, 10, len(st.session_state.spans))
        if n_spans != len(st.session_state.spans):
            st.session_state.spans = [5.0] * n_spans
            st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'None'} for i in range(n_spans+1)]
            st.rerun()
        for i in range(n_spans):
            st.session_state.spans[i] = st.number_input(f"L{i+1} (m)", 0.1, 30.0, float(st.session_state.spans[i]))
    with c2:
        df_sup = pd.DataFrame([{'id': i, 'type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')} for i in range(n_spans+1)])
        ed_sup = st.data_editor(df_sup, column_config={"type": st.column_config.SelectboxColumn("Type", options=["None", "Pin", "Roller", "Fixed"])}, hide_index=True, use_container_width=True)
        st.session_state.supports = ed_sup.to_dict('records')

with t2:
    with st.form("load_form"):
        l_idx = st.selectbox("Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
        l_type = st.selectbox("Type", ["P (Point)", "U (Uniform)", "M (Moment)"])
        l_mag = st.number_input("Magnitude (kN or kNm)", 10.0)
        x_pos = st.number_input("x (m from left)", 0.0)
        dist = st.number_input("Length (m)", 0.0) if "U" in l_type else 0.0
        if st.form_submit_button("➕ Add Load"):
            st.session_state.loads.append({'span_index': l_idx, 'type': l_type[0], 'mag': l_mag*1000, 'x': x_pos, 'dist': dist, 'case': 'DL'})
            st.rerun()

st.divider()
if st.button("🚀 EXECUTE FULL ANALYSIS", type="primary", use_container_width=True):
    sol = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b, h, I_val)
    df, reac, eq = sol.solve()
    
    if not df.empty:
        # Diagrams
        design_view.draw_interactive_diagrams(df, reac, st.session_state.spans, pd.DataFrame(st.session_state.supports), st.session_state.loads)

        # Equilibrium Check
        st.header("⚖️ Equilibrium & Reactions")
        st.dataframe(reac, hide_index=True, use_container_width=True)
        e1, e2 = st.columns(2)
        e1.metric("Vertical Balance (kN)", f"Load: {eq['l_fy']/1000:.2f} | Reac: {eq['r_fy']/1000:.2f}")
        e2.metric("Moment Balance (kNm)", f"Load: {eq['l_m0']/1000:.2f} | Reac: {eq['r_m0']/1000:.2f}")

        # RC Design (Advanced)
        st.header("🧱 Reinforced Concrete Design Report")
        mu_pos, mu_neg = df['moment'].max()/1000, df['moment'].min()/1000
        vu_max = df['shear'].abs().max()/1000
        rc = rc_design.calculate_advanced_rc(mu_pos, mu_neg, vu_max, b, h, fc, fy)
        
        c_note, c_sum = st.columns([1.5, 1])
        with c_note:
            st.subheader("Calculation Note")
            for line in rc['report']: st.markdown(line)
        with c_sum:
            st.subheader("Final Reinforcement")
            st.success(f"**Bottom Steel:** {max(2, int(rc['as_bot']/314+1))}xDB20")
            st.error(f"**Top Steel:** {max(2, int(rc['as_top']/314+1))}xDB20")
            st.warning(f"**Stirrups:** RB9 @ {int(rc['spacing'])} mm")
