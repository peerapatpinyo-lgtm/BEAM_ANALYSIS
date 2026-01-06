import streamlit as st
import pandas as pd
from solver import BeamSolver
import rc_design
import design_view

st.set_page_config(page_title="World-Class Beam Studio", layout="wide")

# Session State Init
if 'loads' not in st.session_state: st.session_state.loads = []
if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]

with st.sidebar:
    st.title("🏆 Structural Specs")
    fc = st.number_input("fc' (Concrete MPa)", 28.0)
    fy = st.number_input("fy (Steel MPa)", 400.0)
    st.divider()
    b = st.number_input("Width (m)", 0.3)
    h = st.number_input("Height (m)", 0.5)
    i_mode = st.radio("I-Stiffness", ["Gross (Ig)", "Manual"])
    I_val = st.number_input("I (m⁴)", value=(b*h**3)/12, format="%.6e") if i_mode == "Manual" else (b*h**3)/12
    if st.button("🔥 Clear All Data"):
        st.session_state.loads = []; st.rerun()

st.title("🚀 Professional Beam Structural Lab")

col_input, col_load = st.columns([1, 1])

with col_input:
    st.subheader("📏 Geometry")
    n_spans = st.number_input("Number of Spans", 1, 10, len(st.session_state.spans))
    if n_spans != len(st.session_state.spans):
        st.session_state.spans = [5.0] * n_spans
        st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'None'} for i in range(n_spans+1)]
        st.rerun()
    for i in range(n_spans):
        st.session_state.spans[i] = st.number_input(f"L{i+1} (m)", 0.1, 20.0, float(st.session_state.spans[i]))
    
    st.write("⚓ Supports")
    df_s = pd.DataFrame([{'id': i, 'type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')} for i in range(n_spans+1)])
    ed_s = st.data_editor(df_s, column_config={"type": st.column_config.SelectboxColumn("Type", options=["None", "Pin", "Roller", "Fixed"])}, hide_index=True)
    st.session_state.supports = ed_s.to_dict('records')

with col_load:
    st.subheader("📥 Load Manager")
    with st.container(border=True):
        la, lb, lc = st.columns(3)
        l_idx = la.selectbox("Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
        l_t = lb.selectbox("Type", ["P (Point)", "U (Uniform)", "M (Moment)"])
        l_m = lc.number_input("Mag (kN)", 20.0)
        
        lx, ld = st.columns(2)
        x_p = lx.number_input("Position x (m)", 0.0)
        dist = ld.number_input("Dist (m)", 0.0) if "U" in l_t else 0.0
        
        # แก้ปัญหา Add Load: ใช้ปุ่มปกติ ไม่ต้องใช้ Form ถ้ามีปัญหา Session
        if st.button("➕ Add Load to List", use_container_width=True):
            st.session_state.loads.append({'span_index': l_idx, 'type': l_t[0], 'mag': l_m*1000, 'x': x_p, 'dist': dist})
            st.rerun()

    # ตารางแสดง Load ที่เพิ่มเข้าไปแล้ว
    if st.session_state.loads:
        st.write("**Current Loads:**")
        load_df = pd.DataFrame(st.session_state.loads)
        st.dataframe(load_df, height=150, use_container_width=True)

st.divider()

if st.button("🏗️ RUN FULL WORLD-CLASS ANALYSIS", type="primary", use_container_width=True):
    sol = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b, h, I_val)
    df, reac, eq = sol.solve()
    
    if not df.empty:
        design_view.draw_interactive_diagrams(df, reac, st.session_state.spans, pd.DataFrame(st.session_state.supports), st.session_state.loads)

        # Output Results
        st.header("🏁 Engineering Results")
        m1, m2, m3 = st.columns(3)
        mu_pos, mu_neg = df['moment'].max()/1000, df['moment'].min()/1000
        vu_max = df['shear'].abs().max()/1000
        
        m1.metric("Max Moment (+)", f"{mu_pos:.2f} kNm")
        m2.metric("Max Moment (-)", f"{mu_neg:.2f} kNm")
        m3.metric("Max Shear", f"{vu_max:.2f} kN")

        # RC Design Section
        rc = rc_design.calculate_advanced_rc(mu_pos, mu_neg, vu_max, b, h, fc, fy)
        
        c_note, c_det = st.columns([1.5, 1])
        with c_note:
            st.info("📑 **Design Calculation Note**")
            for line in rc['report']: st.markdown(line)
        
        with c_det:
            st.subheader("👷 Reinforcement Detail")
            st.success(f"**Bottom:** {max(2, int(rc['as_bot']/314+1))}xDB20")
            st.error(f"**Top:** {max(2, int(rc['as_top']/314+1))}xDB20")
            st.warning(f"**Stirrups:** RB9 @ {int(rc['spacing'])} mm")

        # Equilibrium Check
        if abs(eq['l_fy'] - eq['r_fy']) < 1e-3:
            st.success(f"⚖️ Statics Verified: Sum Fy = 0 (Error: {abs(eq['l_fy'] - eq['r_fy']):.4f} N)")
