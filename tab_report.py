import streamlit as st
import pandas as pd

def render(final_design_res, project_name, engineer_name):
    st.header("📋 Project Report")
    
    # --- จุดที่แก้ไข: เช็ค DataFrame ให้ถูกต้องเพื่อไม่ให้เกิด ValueError ---
    if final_design_res is None or (isinstance(final_design_res, pd.DataFrame) and final_design_res.empty):
        st.warning("⚠️ ไม่พบข้อมูลการคำนวณ กรุณาตรวจสอบที่ Tab 'Concrete Design' ก่อน")
        return

    # --- ส่วนหัวของรายงาน ---
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Project Name:** {project_name}")
    with col2:
        st.info(f"**Engineer:** {engineer_name}")

    st.write("---")

    # --- แสดงตารางผลการคำนวณ ---
    st.subheader("Summary of Beam Design")
    # แสดงตารางแบบเต็มความกว้าง และจัดฟอร์แมตตัวเลขให้สวยงาม
    st.dataframe(
        final_design_res.style.format({
            "Mu+ (kNm)": "{:.2f}",
            "As Bot (cm2)": "{:.2f}",
            "Mu- (kNm)": "{:.2f}",
            "As Top (cm2)": "{:.2f}",
            "Vu Max (kN)": "{:.2f}",
            "Deflect (cm)": "{:.3f}",
            "Allow (cm)": "{:.3f}"
        }), 
        use_container_width=True
    )

    st.write("---")

    # --- ส่วนของปุ่ม Export และจัดการไฟล์ ---
    st.subheader("Export Options")
    
    c1, c2, c3 = st.columns(3)
    
    # 1. ปุ่ม Download CSV
    csv_data = final_design_res.to_csv(index=False).encode('utf-8-sig') # ใช้ utf-8-sig สำหรับภาษาไทยใน Excel
    c1.download_button(
        label="📥 Download CSV Report",
        data=csv_data,
        file_name=f"Design_Report_{project_name}.csv",
        mime='text/csv',
        use_container_width=True
    )

    # 2. ปุ่ม Print (จำลองการเปิดโหมด Print ของ Browser)
    if c2.button("🖨️ Print Report (Browser)", use_container_width=True):
        st.info("💡 กดปุ่ม Ctrl + P เพื่อพิมพ์หน้านี้ออกเป็น PDF")

    # 3. ปุ่ม Clear Data (ถ้ามีของเดิม)
    if c3.button("🗑️ Clear All Results", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # --- เพิ่มเติม: ส่วนสรุปสถานะ ---
    st.write("")
    if "Check" in final_design_res.columns:
        fails = final_design_res[final_design_res["Check"] == "Fail"].shape[0]
        if fails > 0:
            st.error(f"❌ พบข้อผิดพลาด: มี {fails} ช่วง (Span) ที่ไม่ผ่านการตรวจสอบ Deflection")
        else:
            st.success("✅ การตรวจสอบโครงสร้าง: ทุกช่วงผ่านเกณฑ์มาตรฐาน (All Spans Passed)")

    return True
