# tab_report.py
import streamlit as st
import reporter # สมมติว่าไฟล์ report logic เดิมชื่อนี้

def render(final_design_res, project_name, engineer_name):
    st.header("📝 Detailed Calculation Reports")
    st.markdown(f"**Project:** {project_name} | **Engineer:** {engineer_name}")
    
    if not final_design_res: 
        st.warning("⚠️ Please complete the design in Tab 2 first.")
        return

    for i, res in enumerate(final_design_res):
        with st.expander(f"📘 Calculation Sheet: Span {i+1}", expanded=False):
            # เรียกใช้ฟังก์ชันเจน Report จากโมดูลเดิม หรือเขียนใหม่ตรงนี้ก็ได้
            reporter.render_calculation_report(res)
