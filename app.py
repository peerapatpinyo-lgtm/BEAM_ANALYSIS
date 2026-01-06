import streamlit as st
import pandas as pd
from solver import BeamSolver
import rc_design
import design_view # ไฟล์ที่คุณส่งมา

st.set_page_config(page_title="Pro Beam Studio", layout="wide")

if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state.loads = []

with st.sidebar:
    st.header("Materials & Section")
    fc = st.number_input("fc' (MPa)", 25.0)
    b, h = st.number_input("Width (m)", 0.3), st.number_input("Height (m)", 0.5)
    st.divider()
    I_val = st.number_input("I (m4)", value=(b*h**3)/12, format="%.6e")

st.title("🏗️ Beam Structural Master")

# 1. Geometry & Loads Setup (Tabs)
t1, t2 = st.tabs(["📐 Structure", "加 Loads"])
with t1:
    ns = st.number_input("Spans", 1, 10, len(st.session_state.spans))
    if ns != len(st.session_state.spans):
        st.session_state.spans = [5.0] * ns
        st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'None'} for i in range(ns+1)]
        st.rerun()
    for i in range(ns): st.session_state.spans[i] = st.number_input(f"L{i+1}", 0.1, 30.0, float(st.session_state.spans[i]))
    
    st.write("⚓ Supports")
    df_s = pd.DataFrame([{'id': i, 'type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')} for i in range(ns+1)])
    ed_s = st.data_editor(df_s, column_config={"type": st.column_config.SelectboxColumn("Type", options=["None", "Pin", "Roller", "Fixed"])}, hide_index=True)
    st.session_state.supports = ed_s.to_dict('records')

with t2:
    with st.expander("Add Load", expanded=True):
        la, lb, lc = st.columns(3); l_idx = la.selectbox("Span", range(ns)); l_t = lb.selectbox("Type", ["P", "U", "M"]); l_m = lc.number_input("Mag (kN)", 10.0)
        x_p = st.number_input("x (m)", 0.0, float(st.session_state.spans[l_idx]))
        dist = st.number_input("Dist (m)", 0.0, float(st.session_state.spans[l_idx])) if l_t == "U" else 0.0
        if st.button("Add"):
            st.session_state.loads.append({'span_index': l_idx, 'type': l_t, 'mag': l_m*1000, 'x': x_p, 'dist': dist, 'case': 'DL'})
            st.rerun()

# 2. Execution
if st.button("🚀 ANALYZE & DESIGN", type="primary", use_container_width=True):
    solver = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b, h, I_val)
    df, reac, eq = solver.solve()
    
    if not df.empty:
        # --- ใช้ไฟล์ design_view.py ของคุณ ---
        design_view.draw_interactive_diagrams(df, reac, st.session_state.spans, pd.DataFrame(st.session_state.supports), st.session_state.loads)

        # --- RC Design Result ---
        st.divider()
        mu_max = df['moment'].abs().max() / 1000
        vu_max = df['shear'].abs().max() / 1000
        rc = rc_design.calculate_rc_details(mu_max, vu_max, b, h, fc)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Required As", f"{rc['as_mm2']:.0f} mm²")
        c2.metric("Max Moment", f"{mu_max:.2f} kNm")
        c3.metric("Max Shear", f"{vu_max:.2f} kN")
        
        # Statics Balance Check
        err_f = abs(eq['l_fy'] - eq['r_fy'])
        if err_f < 1e-7: st.success("✅ Equilibrium Balanced (Error ≈ 0)")
        else: st.warning(f"⚠️ Balance Error: {err_f:.4f} N")
