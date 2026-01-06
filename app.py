import streamlit as st
import pandas as pd
from solver import BeamSolver
import rc_design
import design_view # ไฟล์ Interactive Diagram ของคุณ

st.set_page_config(page_title="Professional Structural Beam Lab", layout="wide")

# Session States
if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state.loads = []

# --- Sidebar: Materials & Section ---
with st.sidebar:
    st.header("🛠️ Properties")
    fc = st.number_input("fc' (MPa)", 25.0)
    b = st.number_input("Width (m)", 0.3)
    h = st.number_input("Height (m)", 0.5)
    
    st.divider()
    st.write("**Moment of Inertia (I)**")
    i_mode = st.radio("Calculation Mode", ["Auto (bh³/12)", "Manual Input"], horizontal=True)
    if i_mode == "Manual Input":
        I_val = st.number_input("I (m⁴)", value=(b*h**3)/12, format="%.6e")
    else:
        I_val = (b*h**3)/12
        st.caption(f"Calculated I: {I_val:.6e} m⁴")
    
    if st.button("🗑️ Clear Loads"): 
        st.session_state.loads = []
        st.rerun()

st.title("🏗️ Beam Structural Analyzer & Designer")

# --- Geometry & Loads ---
tab1, tab2 = st.tabs(["📐 Geometry & Supports", "📥 Applied Loads"])

with tab1:
    col_ns, col_ed = st.columns([1, 2])
    with col_ns:
        n_spans = st.number_input("Number of Spans", 1, 10, len(st.session_state.spans))
        if n_spans != len(st.session_state.spans):
            st.session_state.spans = [5.0] * n_spans
            st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'None'} for i in range(n_spans+1)]
            st.rerun()
        for i in range(n_spans):
            st.session_state.spans[i] = st.number_input(f"L{i+1} (m)", 0.1, 30.0, float(st.session_state.spans[i]))
    with col_ed:
        df_sup = pd.DataFrame([{'id': i, 'type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')} for i in range(n_spans+1)])
        ed_sup = st.data_editor(df_sup, column_config={"type": st.column_config.SelectboxColumn("Support", options=["None", "Pin", "Roller", "Fixed"])}, hide_index=True)
        st.session_state.supports = ed_sup.to_dict('records')

with tab2:
    with st.form("load_entry", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        l_idx = c1.selectbox("Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
        l_type = c2.selectbox("Type", ["P (Point)", "U (Uniform)", "M (Moment)"])
        l_mag = c3.number_input("Mag (kN or kNm)", 10.0)
        
        c4, c5 = st.columns(2)
        x_p = c4.number_input("x (m from left)", 0.0, float(st.session_state.spans[l_idx]))
        dist = c5.number_input("Dist (m)", 0.0, float(st.session_state.spans[l_idx])) if "U" in l_type else 0.0
        if st.form_submit_button("➕ Add Load"):
            st.session_state.loads.append({'span_index': l_idx, 'type': l_type[0], 'mag': l_mag*1000, 'x': x_p, 'dist': dist, 'case': 'DL'})
            st.rerun()
    
    if st.session_state.loads:
        for i, ld in enumerate(st.session_state.loads):
            st.text(f"Load {i+1}: {ld['type']} | {ld['mag']/1000}kN | Span {ld['span_index']+1}")

# --- Execution & Results ---
st.divider()
if st.button("🚀 EXECUTE FULL ANALYSIS", type="primary", use_container_width=True):
    sol = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b, h, I_val)
    df, reac, eq = sol.solve()
    
    if not df.empty:
        # 1. Diagrams (Your design_view.py)
        design_view.draw_interactive_diagrams(df, reac, st.session_state.spans, pd.DataFrame(st.session_state.supports), st.session_state.loads)

        # 2. Reactions & Equilibrium
        st.header("⚓ Support Reactions & Equilibrium")
        st.dataframe(reac, use_container_width=True, hide_index=True)
        
        e1, e2 = st.columns(2)
        e1.metric("Sum Fy (Load vs Reac)", f"{eq['l_fy']/1000:.2f} kN / {eq['r_fy']/1000:.2f} kN")
        if abs(eq['l_fy'] - eq['r_fy']) < 1e-5: e1.success("✅ Fy Balanced")
        
        e2.metric("Sum M@0 (Load vs Reac)", f"{eq['l_m0']/1000:.2f} kNm / {eq['r_m0']/1000:.2f} kNm")
        if abs(eq['l_m0'] - eq['r_m0']) < 1e-5: e2.success("✅ M Balanced")

        # 3. Advanced RC Design (The core of your request)
        st.header("🧱 Advanced RC Design & Calculation Note")
        mu_pos, mu_neg = df['moment'].max()/1000, df['moment'].min()/1000
        vu_max = df['shear'].abs().max()/1000
        
        rc = rc_design.calculate_advanced_rc(mu_pos, mu_neg, vu_max, b, h, fc)
        
        rep_col, bar_col = st.columns([1.5, 1])
        with rep_col:
            st.info("📜 **Step-by-Step Calculation**")
            for line in rc['report']: st.write(line)
        
        with bar_col:
            st.subheader("Summary Detailing")
            st.success(f"**Bottom Steel:** {max(2, int(rc['as_bot']/314+1))}xDB20")
            st.error(f"**Top Steel:** {max(2, int(rc['as_top']/314+1))}xDB20")
            st.warning(f"**Stirrups:** RB9 @ {int(rc['spacing'])} mm")

    else:
        st.error("Analysis failed. Please check your support and load configuration.")
