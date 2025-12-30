import streamlit as st
import pandas as pd

def render_sidebar():
    st.sidebar.header("⚙️ ตั้งค่า (Settings)")
    
    # Units
    st.sidebar.subheader("1. หน่วยวัด (Units)")
    unit_sys = st.sidebar.radio("ระบบหน่วย", ["Metric (kg, m)", "SI (kN, m)"])
    if "kg" in unit_sys:
        u_force, u_len = "kg", "m"
    else:
        u_force, u_len = "kN", "m"
        
    # Material
    st.sidebar.subheader("2. วัสดุ (Material)")
    fc = st.sidebar.number_input("คอนกรีต fc' (ksc/MPa)", value=240, step=10)
    fy = st.sidebar.number_input("เหล็กเสริม fy (ksc/MPa)", value=4000, step=100)
    
    return {'u_force': u_force, 'u_len': u_len, 'fc': fc, 'fy': fy}

def render_model_inputs(params):
    st.subheader("1. โครงสร้างคาน (Structure Model)")
    
    # 1. Spans
    c1, c2 = st.columns([1, 2])
    with c1:
        n_spans = st.number_input("จำนวนช่วงคาน", min_value=1, max_value=10, value=2)
    
    st.write(f"**ความยาวแต่ละช่วง ({params['u_len']})**")
    spans = []
    cols = st.columns(min(n_spans, 5))
    for i in range(n_spans):
        with cols[i % 5]: 
            # ใช้ Number Input กรอกค่าได้ละเอียด
            val = st.number_input(f"L{i+1}", min_value=0.1, value=4.0, step=0.1, format="%.2f", key=f"len_{i}")
            spans.append(val)
            
    # 2. Supports (Table Input)
    st.write("**จุดรองรับ (Supports)**")
    
    # เตรียมข้อมูลเริ่มต้นให้ Table
    default_types = ['Pin'] + ['Roller'] * (n_spans-1) + ['Roller']
    sup_data = [{"Node": i+1, "Type": default_types[i] if i < len(default_types) else 'Roller'} for i in range(n_spans + 1)]
    
    # ใช้ Data Editor เพื่อให้เลือกง่าย ไม่ซ้อนกัน
    edited_df = st.data_editor(
        pd.DataFrame(sup_data),
        column_config={
            "Node": st.column_config.TextColumn("ตำแหน่ง (Node)", disabled=True),
            "Type": st.column_config.SelectboxColumn("ชนิดจุดรองรับ", options=['Pin', 'Roller', 'Fixed', 'None'], required=True)
        },
        hide_index=True,
        use_container_width=True
    )
    
    sup_config = [{'id': i, 'type': row['Type']} for i, row in edited_df.iterrows() if row['Type'] != 'None']
    sup_df = pd.DataFrame(sup_config)
    
    # Stability Check
    stable = len(sup_df) >= 2 or any(s['type'] == 'Fixed' for s in sup_config)
        
    return n_spans, spans, sup_df, stable

def render_loads(n_spans, spans, params):
    st.subheader("2. น้ำหนักบรรทุก (Loads)")
    
    if 'loads' not in st.session_state:
        st.session_state['loads'] = []

    with st.container():
        # Input Layout
        c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
        with c1:
            l_type = st.radio("ชนิดโหลด", ["Point Load (แรงจุด)", "Uniform Load (แรงแผ่)"], horizontal=True)
        with c2:
            span_idx = st.selectbox("Span ที่", range(1, n_spans+1)) - 1
            max_len = spans[span_idx]
        with c3:
            mag = st.number_input(f"ขนาด ({params['u_force']})", value=1000.0, step=100.0)
        
        # Logic การกรอกระยะ (ใช้ Number Input ตามสั่ง)
        with c4:
            if "Point" in l_type:
                loc = st.number_input(f"ระยะ x ({params['u_len']})", 
                                    min_value=0.0, max_value=float(max_len), 
                                    value=float(max_len)/2, step=0.1)
                add_btn = st.button("➕ เพิ่มแรงจุด")
                if add_btn:
                    st.session_state['loads'].append({'type': 'P', 'span_idx': span_idx, 'P': mag, 'x': loc})
            else:
                st.info("เต็มช่วงคาน")
                st.write("") # Spacer
                add_btn = st.button("➕ เพิ่มแรงแผ่")
                if add_btn:
                    st.session_state['loads'].append({'type': 'U', 'span_idx': span_idx, 'w': mag})

    # Show Loads Table
    if st.session_state['loads']:
        st.markdown("---")
        load_data = []
        for i, l in enumerate(st.session_state['loads']):
            desc = f"P = {l['P']} @ {l['x']}m" if l['type'] == 'P' else f"w = {l['w']} (Full Span)"
            load_data.append({"No.": i+1, "Span": l['span_idx']+1, "Description": desc})
            
        c_table, c_del = st.columns([8, 2])
        with c_table:
            st.dataframe(pd.DataFrame(load_data), hide_index=True, use_container_width=True)
        with c_del:
            if st.button("ลบทั้งหมด (Clear)"):
                st.session_state['loads'] = []
                st.rerun()

    return st.session_state['loads']
