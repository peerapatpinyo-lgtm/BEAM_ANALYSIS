import streamlit as st
import pandas as pd
from solver import BeamSolver
import design_view

st.set_page_config(page_title="Professional Beam Analysis", layout="wide")

if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state.loads = []

with st.sidebar:
    st.header("Materials & Section")
    fc = st.number_input("fc' (MPa)", 25.0)
    b = st.number_input("Width (m)", 0.3)
    h = st.number_input("Height (m)", 0.5)
    use_i = st.checkbox("Custom I")
    I_val = st.number_input("I (m4)", value=(b*h**3)/12, format="%.6e") if use_i else (b*h**3)/12

st.title("🏗️ Beam Master: Timoshenko Analysis")

# 1. Structure
st.header("1. Geometry")
col1, col2 = st.columns([1, 2])
with col1:
    n_s = st.number_input("Spans", 1, 10, len(st.session_state.spans))
    if n_s != len(st.session_state.spans):
        st.session_state.spans = [5.0] * n_s
        st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'None'} for i in range(n_s+1)]
        st.rerun()
    for i in range(n_s):
        st.session_state.spans[i] = st.number_input(f"L{i+1} (m)", 0.1, 30.0, float(st.session_state.spans[i]))
with col2:
    st.write("⚓ Supports")
    df_s = pd.DataFrame([{'Node': i, 'Type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')} for i in range(n_s+1)])
    ed_s = st.data_editor(df_s, column_config={"Type": st.column_config.SelectboxColumn("Type", options=["None", "Pin", "Roller", "Fixed"])}, hide_index=True)
    st.session_state.supports = [{'id': r['Node'], 'type': r['Type']} for _, r in ed_s.iterrows()]

# 2. Loads
st.header("2. Loads")
with st.expander("Add Load", expanded=True):
    c_a, c_b, c_c = st.columns(3)
    l_idx = c_a.selectbox("Span", range(n_s))
    l_t = c_b.selectbox("Type", ["P (Point)", "U (Uniform)", "M (Moment)"])
    l_m = c_c.number_input("Mag (kN/kNm)", 10.0)
    pos_x = st.number_input("x (m from left)", 0.0, float(st.session_state.spans[l_idx]), 0.0)
    dist_l = st.number_input("Dist (m)", 0.0, float(st.session_state.spans[l_idx])) if "U" in l_t else 0.0
    if st.button("➕ Add"):
        st.session_state.loads.append({'span_index': l_idx, 'type': l_t[0], 'mag': l_m*1000, 'x': pos_x, 'dist': dist_l})
        st.rerun()

for i, ld in enumerate(st.session_state.loads):
    cc1, cc2 = st.columns([5, 1])
    cc1.info(f"#{i+1}: Span {ld['span_index']+1} | {ld['type']} | {ld['mag']/1000} kN")
    if cc2.button("🗑️", key=f"d_{i}"):
        st.session_state.loads.pop(i); st.rerun()

# 3. Analyze
if st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True):
    sol = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b, h, I_val)
    df, reac, eq = sol.solve()
    if not df.empty:
        design_view.draw_interactive_diagrams(df, None, st.session_state.spans, st.session_state.supports, st.session_state.loads)
        
        st.header("⚖️ Statics Verification")
        v1, v2 = st.columns(2)
        with v1:
            st.metric("Total Load (kN)", f"{eq['l_fy']/1000:.3f}")
            st.metric("Total Reaction (kN)", f"{eq['r_fy']/1000:.3f}")
            err_f = abs(eq['l_fy'] - eq['r_fy'])
            if err_f < 0.1: st.success(f"Balanced ({err_f:.2e} N)")
            else: st.error(f"Unbalanced ({err_f:.2f} N)")
        with v2:
            st.metric("Load Moment (kNm)", f"{eq['l_m0']/1000:.3f}")
            st.metric("Reaction Moment (kNm)", f"{eq['r_m0']/1000:.3f}")
            err_m = abs(eq['l_m0'] - eq['r_m0'])
            if err_m < 0.1: st.success(f"Balanced ({err_m:.2e} Nm)")
            else: st.error(f"Unbalanced ({err_m:.2f} Nm)")

        st.header("⚓ Reaction Details")
        st.table(reac)

        st.header("📉 Deflection Check")
        max_d = df['deflection'].abs().max() * 1000
        total_L = sum(st.session_state.spans)
        st.write(f"Max: {max_d:.2f} mm | Limit L/240: {total_L*1000/240:.2f} mm")
        if max_d > total_L*1000/240: st.error("Deflection Exceeds Limit")
        else: st.success("Serviceability OK")
