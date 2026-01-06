import streamlit as st
import pandas as pd
from solver import BeamSolver
import design_view

st.set_page_config(page_title="Professional Beam Structural System", layout="wide")

# Session State Initialization
if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state.loads = []

st.title("🏗️ Professional Beam Analysis (Full Verification)")

# --- 1. Materials & Section (fc', fy, I กลับมาแล้ว) ---
st.sidebar.header("Section & Materials")
fc = st.sidebar.number_input("Concrete Strength (fc') [MPa]", 20.0, 50.0, 25.0)
fy = st.sidebar.number_input("Steel Strength (fy) [MPa]", 240.0, 500.0, 400.0)
b = st.sidebar.number_input("Width (b) [m]", 0.1, 1.0, 0.3)
h = st.sidebar.number_input("Height (h) [m]", 0.1, 2.0, 0.5)

use_custom_i = st.sidebar.checkbox("Custom I (Moment of Inertia)")
if use_custom_i:
    I_val = st.sidebar.number_input("I (m4)", value=(b*h**3)/12, format="%.6e")
else:
    I_val = (b * h**3) / 12
    st.sidebar.info(f"Auto Calculated I: {I_val:.6e} m⁴")

# --- 2 & 6. Spans & Supports Input ---
st.header("1. Structure Configuration")
col_g1, col_g2 = st.columns([1, 2])

with col_g1:
    n_spans = st.number_input("Number of Spans", 1, 10, len(st.session_state.spans))
    if n_spans != len(st.session_state.spans):
        st.session_state.spans = [5.0] * n_spans
        st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'None'} for i in range(n_spans+1)]
        st.rerun()
    for i in range(n_spans):
        st.session_state.spans[i] = st.number_input(f"Span {i+1} (m)", 0.1, 25.0, float(st.session_state.spans[i]))

with col_g2:
    st.write("⚓ Support Settings")
    n_nodes = n_spans + 1
    sup_df_input = pd.DataFrame([
        {'Node': i, 'Type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')}
        for i in range(n_nodes)
    ])
    edited_sup = st.data_editor(
        sup_df_input,
        column_config={"Type": st.column_config.SelectboxColumn("Type", options=["None", "Pin", "Roller", "Fixed"])},
        hide_index=True, use_container_width=True
    )
    st.session_state.supports = [{'id': r['Node'], 'type': r['Type']} for _, r in edited_sup.iterrows()]

# --- 3. Loads Management (Delete Item by Item) ---
st.header("2. Loads Management")
with st.expander("Add New Load", expanded=True):
    cl1, cl2, cl3, cl4 = st.columns([1, 1, 1, 2])
    l_span = cl1.selectbox("Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
    l_type = cl2.selectbox("Type", ["Point (P)", "Uniform (U)", "Moment (M)"])
    l_case = cl3.selectbox("Case", ["DL", "LL"])
    l_mag = cl4.number_input("Magnitude (kN or kNm)", value=10.0)
    
    x_pos = st.number_input("Position x from left (m)", 0.0, float(st.session_state.spans[l_span]), 0.0)
    l_dist = st.number_input("Load Length (m)", 0.0, float(st.session_state.spans[l_span]), 0.0) if "U" in l_type else 0.0

    if st.button("➕ Add Load"):
        t_code = "P" if "P" in l_type else ("U" if "U" in l_type else "M")
        st.session_state.loads.append({'span_index': l_span, 'type': t_code, 'mag': l_mag*1000, 'x': x_pos, 'dist': l_dist, 'case': l_case})
        st.rerun()

if st.session_state.loads:
    st.write("Current Loads:")
    for i, ld in enumerate(st.session_state.loads):
        cols = st.columns([5, 1])
        cols[0].info(f"#{i+1}: Span {ld['span_index']+1} | {ld['type']} Load: {ld['mag']/1000} | x={ld['x']}m")
        if cols[1].button("🗑️", key=f"del_{i}"):
            st.session_state.loads.pop(i)
            st.rerun()

# --- Analysis & All Checks (Reactions, Equation Check, Deep Beam) ---
if st.button("🚀 EXECUTE ANALYSIS", type="primary", use_container_width=True):
    solver = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b, h, I_val)
    df, reac_df, eq = solver.solve()
    
    if not df.empty:
        # Visualization
        design_view.draw_interactive_diagrams(df, None, st.session_state.spans, st.session_state.supports, st.session_state.loads)

        # 1. Reaction Table
        st.subheader("⚓ Support Reactions")
        st.table(reac_df)

        # 2. Equation Check (Statics Check)
        st.subheader("⚖️ Equation Check (Statics Verification)")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**ΣFy = 0 (kN)**")
            st.write(f"Total Load: {eq['sum_fy_load']/1000:.3f}")
            st.write(f"Total Reaction: {eq['sum_fy_reac']/1000:.3f}")
            diff_y = abs(eq['sum_fy_load'] - eq['sum_fy_reac'])
            st.markdown(f"**Error Fy:** `{diff_y:.6f} N` " + ("✅ Balanced" if diff_y < 0.1 else "❌ Check Stability"))
        with c2:
            st.write("**ΣM @ Node 0 = 0 (kNm)**")
            st.write(f"Load Moment: {eq['sum_m0_load']/1000:.3f}")
            st.write(f"Reaction Moment: {eq['sum_m0_reac']/1000:.3f}")
            diff_m = abs(eq['sum_m0_load'] - eq['sum_m0_reac'])
            st.markdown(f"**Error M:** `{diff_m:.6f} Nm` " + ("✅ Balanced" if diff_m < 0.1 else "❌ Check Stability"))

        # 4 & 5. Analysis Summary & Deep Beam Check
        st.subheader("📊 Results Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Max Shear", f"{df['shear'].abs().max()/1000:.2f} kN")
        m2.metric("Max Moment", f"{df['moment'].abs().max()/1000:.2f} kNm")
        m3.metric("Max Deflection", f"{df['deflection'].abs().max()*1000:.2f} mm")

        d_eff = h - 0.05
        is_deep = any((L / d_eff) < 2.0 for L in st.session_state.spans)
        if is_deep:
            st.warning("⚠️ **DEEP BEAM WARNING:** One or more spans have L/d < 2.0. Use Strut-and-Tie model for design.")
            
        else:
            st.success("✅ Slender Beam Condition (L/d ≥ 2.0)")

        st.info(f"Material Status: fc'={fc} MPa, fy={fy} MPa")
    else:
        st.error("Analysis failed. Ensure at least 2 supports are defined.")
