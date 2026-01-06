import streamlit as st
import json
import pandas as pd

def export_data(params, spans, sup_df, loads_df):
    """
    รวบรวมข้อมูลทั้งหมดในโปรแกรม อัดเป็น JSON String เพื่อ Save
    """
    # แปลง Dataframe เป็น Dict เพื่อให้ Save ลง JSON ได้
    loads_list = loads_df.to_dict('records') if loads_df is not None else []
    sup_list = sup_df.to_dict('records') if not sup_df.empty else []
    
    project_data = {
        "version": "1.0",
        "params": params,
        "spans": spans,
        "supports": sup_list,
        "loads": loads_list
    }
    
    # แปลงเป็น Text (JSON)
    return json.dumps(project_data, indent=4)

def load_data(uploaded_file):
    """
    อ่านไฟล์ JSON แล้วแกะข้อมูลออกมาคืนค่ากลับสู่ระบบ
    """
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            
            # Validate เบื้องต้นว่าไฟล์ถูกไหม
            if "spans" not in data or "loads" not in data:
                st.error("❌ Invalid Project File")
                return None
                
            return data
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            return None
    return None
