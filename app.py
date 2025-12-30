import streamlit as st
import pandas as pd
import solver
import input_handler
import design_view 

# Config หน้าจอ
st.set_page_config(page_title="โปรแกรมคำนวณคาน (Beam Analysis)", layout="wide")

# CSS ปรับแต่งให้ตัวหนังสือใหญ่ขึ้น อ่านง่ายสำหรับผู้ใหญ่
st.markdown("""
<style>
    .stButton button {
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
        border-radius: 8px;
    }
    h1, h2, h3 { font-family: 'Sarabun', sans-serif; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🏗️ โปรแกรมวิเคราะห์คานต่อเนื่อง (Beam Analysis)")
    st.markdown("---")

    # แบ่งหน้าจอเป็น 2 ส่วน: ซ้าย (Input) | ขวา (Output)
    col_input, col_output = st.columns([35, 65], gap="large")

    # === PANEL ซ้าย: ข้อมูลนำเข้า ===
    with col_input:
        with st.container(border=True): # กรอบสวยงาม
            # 1. Sidebar settings (เรียกใช้)
            params = input_handler.render_sidebar()
            
            # 2. Model Geometry
            n, spans, sup_df, stable = input_handler.render_model_inputs(params)
            
            # 3. Loads
            st.markdown("---")
            loads = input_handler.render_loads(n, spans, params)
            
            # 4. ปุ่มคำนวณ (ใหญ่และชัดเจน)
            st.markdown("###")
            run_btn = st.button("▶️ กดเพื่อคำนวณ (CALCULATE)", type="primary", disabled=not stable)
            
            if not stable:
                st.error("⚠️ โครงสร้างไม่เสถียร (Unstable)! กรุณาเพิ่มจุดรองรับ")

    # === PANEL ขวา: ผลลัพธ์ ===
    with col_output:
        if run_btn or st.session_state.get('analysis_done'):
            if run_btn:
                # คำนวณ
                try:
                    engine = solver.BeamSolver(spans, sup_df, loads)
                    df_res, reactions = engine.solve()
                    # Save state
                    st.session_state.update({
                        'analysis_done': True, 
                        'df_res': df_res, 
                        'reactions': reactions,
                        'spans': spans, 
                        'sup_df': sup_df, 
                        'loads': loads
                    })
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")
                    st.stop()

            # ดึงค่ามาแสดง
            df = st.session_state['df_res']
            reac = st.session_state['reactions']
            spans = st.session_state['spans']

            # 1. แสดงกราฟ
            design_view.draw_interactive_diagrams(
                df, spans, st.session_state['sup_df'], 
                st.session_state['loads'], params['u_force'], params['u_len']
            )
            
            # 2. แสดงตารางผลลัพธ์
            st.markdown("---")
            design_view.render_result_tables(df, reac, spans)
            
        else:
            # หน้าจอเริ่มต้น
            st.info("👈 กรุณากรอกข้อมูลทางด้านซ้าย แล้วกดปุ่ม 'คำนวณ'")
            st.markdown("""
            **คู่มือการใช้งานเบื้องต้น:**
            1. ตั้งค่าหน่วยและวัสดุที่แถบซ้ายสุด (Sidebar)
            2. กำหนดจำนวนช่วงคาน และความยาว
            3. เลือกชนิดจุดรองรับในตาราง
            4. ใส่แรงกระทำ (อย่าลืมคูณ Load Factor มาก่อน)
            5. กดปุ่มคำนวณ
            """)

if __name__ == "__main__":
    main()
