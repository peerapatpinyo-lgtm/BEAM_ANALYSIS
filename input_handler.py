import streamlit as st
import pandas as pd

def render_sidebar():
    st.sidebar.header("⚙️ ตั้งค่า (Settings)")
    
    # 1. Units
    st.sidebar.subheader("1. หน่วยวัด (Units)")
    unit_sys = st.sidebar.radio("ระบบหน่วย", ["Metric (kg, m)", "SI (kN, m)"])
    if "kg" in unit_sys:
        u_force, u_len = "kg", "m"
    else:
        u_force, u_len = "kN", "m"
        
    # 2. Material
    st.sidebar.subheader("2. วัสดุ (Material)")
    fc = st.sidebar.number_input("คอนกรีต fc' (ksc/MPa)", value=240, step=10)
    fy = st.sidebar.number_input("เหล็กเสริม fy (ksc/MPa)", value=4000, step=100)
    
    # 3. Load Combinations (เอามาให้เห็นชัดๆ)
    st.sidebar.subheader("3. ตัวคูณน้ำหนัก (Load Factors)")
    st.sidebar.info("""
    💡 **คำแนะนำ (Note):**
    โปรแกรมวิเคราะห์แบบ Linear Elastic
    กรุณาป้อนน้ำหนักบรรทุกที่คูณค่าความปลอดภัยแล้ว
    (Factor Load: 1.4DL + 1.7LL หรือ 1.2D + 1.6L)
    """)
    
    return {'u_force': u_force, 'u_len': u_len, 'fc': fc, 'fy': fy}

def render_model_inputs(params):
    st.subheader("1. กำหนดช่วงคานและจุดรองรับ (Geometry & Supports)")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        n_spans = st.number_input("จำนวนช่วงคาน (Number of Spans)", min_value=1, max_value=10, value=2)
    
    # --- PART 1: SPAN LENGTHS ---
    st.write(f"**ความยาวแต่ละช่วง ({params['u_len']})**")
    spans = []
    cols = st.columns(min(n_spans, 5)) # แสดงทีละ 5 ช่องเพื่อไม่ให้เบียด
    for i in range(n_spans):
        # วนลูปสร้าง input แต่ถ้าเยอะเกิน 5 ให้ขึ้นบรรทัดใหม่
        with cols[i % 5]: 
            val = st.number_input(f"L{i+1}", min_value=0.1, value=4.0, step=0.5, key=f"len_{i}")
            spans.append(val)
            
    # --- PART 2: SUPPORTS (แก้ปัญหาซ้อนกัน โดยใช้ Data Editor) ---
    st.write("**จุดรองรับ (Supports)**")
    
    # สร้าง Dataframe เริ่มต้น
    sup_data = []
    default_types = ['Pin'] + ['Roller'] * (n_spans-1) + ['Roller']
    
    for i in range(n_spans + 1):
        sup_data.append({
            "Position": f"จุดที่ {i+1}", 
            "Type": default_types[i] if i < len(default_types) else 'Roller'
        })
    
    df_sup_input = pd.DataFrame(sup_data)
    
    # ใช้ Data Editor แทน Dropdown เรียงกัน
    edited_df = st.data_editor(
        df_sup_input,
        column_config={
            "Position": st.column_config.TextColumn("ตำแหน่ง (Node)", disabled=True),
            "Type": st.column_config.SelectboxColumn(
                "ชนิดจุดรองรับ",
                options=['Pin', 'Roller', 'Fixed', 'None'],
                required=True
            )
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Convert กลับเป็น Format เดิมเพื่อส่งเข้า Solver
    sup_config = []
    for idx, row in edited_df.iterrows():
        if row['Type'] != 'None':
            sup_config.append({'id': idx, 'type': row['Type']})
            
    sup_df = pd.DataFrame(sup_config)
    
    # Check Stability
    stable = True
    if len(sup_df) < 2 and not any(s['type'] == 'Fixed' for s in sup_config):
        stable = False
        
    return n_spans, spans, sup_df, stable

def render_loads(n_spans, spans, params):
    st.subheader("2. น้ำหนักบรรทุก (Loads)")
    
    if 'loads' not in st.session_state:
        st.session_state['loads'] = []

    # Input Box แบบง่ายๆ
    with st.container():
        c1, c2, c3 = st.columns([1.5, 1, 1])
        with c1:
            l_type = st.radio("ชนิดโหลด", ["Point Load (แรงจุด)", "Uniform Load (แรงแผ่)"], horizontal=True)
        with c2:
            span_idx = st.selectbox("กระทำที่ช่วงคานที่ (Span)", range(1, n_spans+1)) - 1
        with c3:
            mag = st.number_input(f"ขนาดแรง ({params['u_force']})", value=1000.0, step=100.0)

        # Inputs ย่อยตามประเภท
        if "Point" in l_type:
            loc = st.slider(f"ระยะจากซ้ายของช่วงคาน ({params['u_len']})", 
                            0.0, spans[span_idx], spans[span_idx]/2.0)
            if st.button("➕ เพิ่มแรงจุด (Add Point Load)"):
                st.session_state['loads'].append({'type': 'P', 'span_idx': span_idx, 'P': mag, 'x': loc})
        else:
            st.info(f"แรงแผ่เต็มช่วงคานที่ {span_idx+1}")
            if st.button("➕ เพิ่มแรงแผ่ (Add Uniform Load)"):
                st.session_state['loads'].append({'type': 'U', 'span_idx': span_idx, 'w': mag})

    # ตารางแสดง Load ที่ใส่ไปแล้ว
    if st.session_state['loads']:
        st.markdown("---")
        st.write("**รายการโหลดที่ใส่แล้ว:**")
        
        # แสดงเป็นตารางสวยๆ
        load_display = []
        for i, l in enumerate(st.session_state['loads']):
            if l['type'] == 'P':
                desc = f"แรงจุด P = {l['P']} {params['u_force']} @ ระยะ {l['x']} {params['u_len']}"
            else:
                desc = f"แรงแผ่ w = {l['w']} {params['u_force']}/{params['u_len']} (เต็มช่วง)"
            load_display.append({"No.": i+1, "Span": l['span_idx']+1, "Description": desc})
            
        st.dataframe(pd.DataFrame(load_display), hide_index=True, use_container_width=True)
        
        if st.button("ลบโหลดทั้งหมด (Clear All Loads)", type="secondary"):
            st.session_state['loads'] = []
            st.rerun()

    return st.session_state['loads']
