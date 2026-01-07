import streamlit as st
import solver
import rc_design

st.title("Timoshenko Beam Analysis & Design")

# --- Inputs ---
col1, col2 = st.columns(2)
with col1:
    L = st.number_input("Span Length (m)", value=5.0)
    b_mm = st.number_input("Width b (mm)", value=200.0)
    h_mm = st.number_input("Height h (mm)", value=400.0)
with col2:
    fc = st.number_input("f'c (MPa)", value=24.0)
    fy = st.number_input("fy (MPa)", value=400.0)
    p_load = st.number_input("Point Load (kN)", value=5.0)

if st.button("Calculate Now"):
    # บล็อกนี้สำคัญมาก: ตรวจสอบย่อหน้าให้ตรงกัน
    try:
        # เตรียมข้อมูล
        spans = [L]
        sup_list = [0, L]
        load_list = [{'span_index': 0, 'x': L/2, 'mag': p_load, 'type': 'P'}]

        # 1. วิเคราะห์ด้วย Timoshenko
        beam_solver = solver.TimoshenkoBeamSolver(
            spans=spans, 
            supports=sup_list, 
            loads=load_list, 
            b_mm=b_mm, 
            h_mm=h_mm, 
            fc=fc
        )
        m_max, v_max = beam_solver.solve()

        # 2. ออกแบบเหล็กเสริม
        result, logs = rc_design.design_section(m_max, v_max, b_mm, h_mm, fc, fy)

        # 3. แสดงผล
        st.divider()
        st.subheader("Results")
        st.write(f"**Max Moment:** {m_max:.2f} kNm")
        st.write(f"**Max Shear:** {v_max:.2f} kN")
        
        for line in logs:
            st.markdown(line)
            
    except Exception as e:
        st.error(f"Error in execution: {e}")
