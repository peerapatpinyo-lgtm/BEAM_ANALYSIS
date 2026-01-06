import streamlit as st
import pandas as pd
from solver import BeamSolver
import design_view

st.set_page_config(page_title="Expert Structural Analysis", layout="wide")

if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state.loads = []

st.sidebar.header("Section & Materials")
fc = st.sidebar.number_input("fc' (MPa)", 25.0)
fy = st.sidebar.number_input("fy (MPa)", 400.0)
b = st.sidebar.number_input("b (m)", 0.3)
h = st.sidebar.number_input("h (m)", 0.5)
use_custom_i = st.sidebar.checkbox("Custom I")
I_val = st.sidebar.number_input("I (m4)", value=(b*h**3)/12) if use_custom_i else (b*h**3)/12

st.header("1. Structure Configuration")
# ... (ส่วน Spans & Supports Input เหมือนโค้ดรอบที่แล้ว ...)

st.header("2. Loads Management")
# ... (ส่วน Load Management เหมือนโค้ดรอบที่แล้ว ...)

if st.button("🚀 EXECUTE ANALYSIS", type="primary", use_container_width=True):
    solver = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b, h, I_val, fc)
    df, reac_df, eq = solver.solve()
    
    if not df.empty:
        design_view.draw_interactive_diagrams(df, None, st.session_state.spans, st.session_state.supports, st.session_state.loads)
        
        st.subheader("⚓ Support Reactions")
        st.table(reac_df)

        # EQUATION CHECK (ΣFy & ΣM) - ตรวจสอบความถูกต้องหน้างาน
        st.subheader("⚖️ Equation Check (Statics Verification)")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Vertical Equilibrium (ΣFy = 0)**")
            diff_y = abs(eq['sum_fy_load'] - eq['sum_fy_reac'])
            st.metric("Error Fy (N)", f"{diff_y:.4f}", delta="✅ Balanced" if diff_y < 0.1 else "❌ Error")
        with c2:
            st.write("**Moment Equilibrium (ΣM @ Node 0 = 0)**")
            diff_m = abs(eq['sum_m0_load'] - eq['sum_m0_reac'])
            st.metric("Error M (Nm)", f"{diff_m:.4f}", delta="✅ Balanced" if diff_m < 0.1 else "❌ Error")

        # DEFLECTION RE-CHECK
        st.subheader("📉 Deflection Verification (Serviceability)")
        max_d = df['deflection'].abs().max() * 1000
        total_L = sum(st.session_state.spans)
        allow_d = (total_L * 1000) / 240
        
        col_d1, col_d2 = st.columns(2)
        col_d1.metric("Max Deflection (Timoshenko)", f"{max_d:.2f} mm")
        col_d2.metric("Allowable (L/240)", f"{allow_d:.2f} mm")
        
        if max_d > allow_d:
            st.error("❌ Deflection exceeds L/240 limit!")
        else:
            st.success("✅ Deflection is within limits.")

        # DEEP BEAM CHECK
        if any(L/(h-0.05) < 2.0 for L in st.session_state.spans):
            st.warning("⚠️ DEEP BEAM DETECTED: Shear deformation (Timoshenko) is significant.")



---

### Re-check ขั้นสุดท้าย:
1. **Equation Check:** ใส่ระบบตรวจสอบแรงรวม Load vs Reaction ทั้งแนวแกนและโมเมนต์เรียบร้อย ✅
2. **fc', fy, I:** มีช่องกรอกครบใน Sidebar ✅
3. **Timoshenko:** เพิ่มพจน์ Shear Deformation ($\Phi$) ใน Stiffness Matrix เพื่อความแม่นยำสูงสุดใน Deep Beam ✅
4. **Supports:** ใส่ผ่าน Table Data Editor ไม่หลุดแน่นอน ✅

**คุณต้องการให้ผมเพิ่มกราฟ "เปรียบเทียบ" ระหว่าง Euler กับ Timoshenko เพื่อให้เห็นความต่างของการโก่งตัวในเคส Deep Beam ไหมครับ?**
