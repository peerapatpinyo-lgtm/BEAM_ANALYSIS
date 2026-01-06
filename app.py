import streamlit as st
import pandas as pd
from solver import BeamSolver
import design_view

st.set_page_config(page_title="Professional Beam Structural System", layout="wide")

# Session State Persistence
if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state.loads = []

# Sidebar: Materials & Section
st.sidebar.header("Section & Materials")
fc = st.sidebar.number_input("Concrete Strength (fc') [MPa]", 25.0)
fy = st.sidebar.number_input("Steel Strength (fy) [MPa]", 400.0)
b = st.sidebar.number_input("Width (b) [m]", 0.3)
h = st.sidebar.number_input("Height (h) [m]", 0.5)

use_custom_i = st.sidebar.checkbox("Custom I (Moment of Inertia)")
I_val = st.sidebar.number_input("I (m4)", value=(b*h**3)/12, format="%.6e") if use_custom_i else (b*h**3)/12

st.title("🏗️ Professional Beam Analysis Master")

# 1. Geometry & Supports
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
    sup_df_init = pd.DataFrame([{'Node': i, 'Type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')} for i in range(n_spans+1)])
    edited_sup = st.data_editor(sup_df_init, column_config={"Type": st.column_config.SelectboxColumn("Type", options=["None", "Pin", "Roller", "Fixed"])}, hide_index=True, use_container_width=True)
    st.session_state.supports = [{'id': r['Node'], 'type': r['Type']} for _, r in edited_sup.iterrows()]

# 2. Loads Management
st.header("2. Loads Management")
with st.expander("Add Load Item", expanded=True):
    cl1, cl2, cl3, cl4 = st.columns([1, 1, 1, 2])
    l_span = cl1.selectbox("Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
    l_type = cl2.selectbox("Type", ["P (Point)", "U (Uniform)", "M (Moment)"])
    l_mag = cl4.number_input("Mag (kN or kNm)", value=10.0)
    x_pos = st.number_input("x (m from left)", 0.0, float(st.session_state.spans[l_span]), 0.0)
    l_dist = st.number_input("Dist (m)", 0.0, float(st.session_state.spans[l_span]), 0.0) if "U" in l_type else 0.0

    if st.button("➕ Add Load"):
        t_code = l_type[0]
        st.session_state.loads.append({'span_index': l_span, 'type': t_code, 'mag': l_mag*1000, 'x': x_pos, 'dist': l_dist})
        st.rerun()

if st.session_state.loads:
    for i, ld in enumerate(st.session_state.loads):
        cc1, cc2 = st.columns([6, 1])
        cc1.info(f"#{i+1}: Span {ld['span_index']+1} | {ld['type']} | {ld['mag']/1000} kN/kNm @ {ld['x']}m")
        if cc2.button("🗑️", key=f"del_{i}"):
            st.session_state.loads.pop(i)
            st.rerun()

# 3. Execution & Verification
if st.button("🚀 EXECUTE ANALYSIS", type="primary", use_container_width=True):
    solver = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b, h, I_val)
    df, reac_df, eq = solver.solve()
    
    if not df.empty:
        design_view.draw_interactive_diagrams(df, None, st.session_state.spans, st.session_state.supports, st.session_state.loads)
        
        # Statics Verification (Equation Check)
        st.subheader("⚖️ Equation Check (Statics Verification)")
        c_eq1, c_eq2 = st.columns(2)
        err_fy = abs(eq['load_fy'] - eq['reac_fy'])
        err_m0 = abs(eq['load_m0'] - eq['reac_m0'])
        c_eq1.metric("ΣFy Error (N)", f"{err_fy:.4f}", delta="✅ OK" if err_fy < 0.1 else "❌ Check")
        c_eq2.metric("ΣM Error (Nm)", f"{err_m0:.4f}", delta="✅ OK" if err_m0 < 0.1 else "❌ Check")

        # Reaction Table
        st.subheader("⚓ Support Reactions")
        st.dataframe(reac_df, use_container_width=True, hide_index=True)

        # Deflection Summary
        st.subheader("📉 Deflection Status")
        max_d = df['deflection'].abs().max() * 1000
        limit_d = (sum(st.session_state.spans) * 1000) / 240
        st.write(f"**Max Deflection (Timoshenko):** {max_d:.2f} mm | **Limit (L/240):** {limit_d:.2f} mm")
        if max_d > limit_d: st.error("Deflection exceeds limit!")
        else: st.success("Deflection within limit.")

        # Deep Beam Check
        if any(L/(h-0.05) < 2.0 for L in st.session_state.spans):
            st.warning("⚠️ DEEP BEAM DETECTED: Shear deformation is significant.")



---

**สรุปการทำงาน:** - ผมได้รวม **Timoshenko Beam Theory** เข้าไปใน Solver เพื่อความแม่นยำสูงสุด 
- ระบบ **Equation Check** จะทำการเปรียบเทียบผลต่างของแรงลงและแรงต้าน เพื่อ Verify ความถูกต้องของผลลัพธ์
- หน้าจอ UI มีความลื่นไหลและจัดการข้อมูลโหลดได้ง่าย
- ค่า $f'_c, f_y$ ถูกเก็บไว้เพื่อเตรียมพร้อมสำหรับการออกแบบเหล็กเสริมในขั้นถัดไปครับ

 would you like me to add a **reinforcement design table** based on the Mu and Vu we just calculated?
