import streamlit as st
import pandas as pd
import file_manager  # <--- Import ไฟล์ใหม่

def render_sidebar():
    with st.sidebar:
        st.header("🏗️ Project Manager")
        
        # --- SAVE / LOAD SECTION ---
        # เราจะใช้ Session State เพื่อเก็บข้อมูลชั่วคราวตอนโหลด
        uploaded_file = st.file_uploader("📂 Open Project (.json)", type=["json"])
        
        loaded_data = None
        if uploaded_file:
            loaded_data = file_manager.load_data(uploaded_file)
            if loaded_data:
                st.success("✅ File Loaded! (Check inputs)")

        st.markdown("---")
        st.header("⚙️ Design Parameters")
        
        # --- MATERIAL PRESETS (New!) ---
        st.subheader("1. Materials")
        mat_type = st.selectbox("Concrete Grade", 
                                ["Custom", "C20 (fc'20)", "C25 (fc'25)", "C30 (fc'30)", "C35 (fc'35)"])
        
        # Default Values
        def_fc = 24.0
        def_E = 2e10
        
        if mat_type == "C20 (fc'20)": def_fc, def_E = 20.0, 2.1e10
        elif mat_type == "C25 (fc'25)": def_fc, def_E = 25.0, 2.3e10
        elif mat_type == "C30 (fc'30)": def_fc, def_E = 30.0, 2.5e10
        elif mat_type == "C35 (fc'35)": def_fc, def_E = 35.0, 2.7e10
        
        # ถ้าโหลดไฟล์มา ให้ใช้ค่าจากไฟล์
        if loaded_data:
            def_fc = loaded_data['params'].get('fc', def_fc) # ต้องแก้ app.py ให้รับ fc ใน params ด้วย
            def_E = loaded_data['params']['E']

        # Section Inputs
        c1, c2 = st.columns(2)
        # ถ้าโหลดไฟล์มา ใช้ค่าจากไฟล์
        val_b = loaded_data['params']['b'] if loaded_data else 0.25
        val_h = loaded_data['params']['h'] if loaded_data else 0.50
        
        b = c1.number_input("Width b (m)", value=val_b, step=0.05)
        h = c2.number_input("Depth h (m)", value=val_h, step=0.05)
        
        I_g = (b * h**3) / 12
        st.caption(f"I (Gross) = {I_g:.2e} m⁴")
        
        E = st.number_input("Elastic Modulus (E)", value=float(def_E), format="%e")
        
        # --- LOAD FACTORS ---
        st.subheader("2. Load Factors")
        val_dl = loaded_data['params']['gamma_dead'] if loaded_data else 1.4
        val_ll = loaded_data['params']['gamma_live'] if loaded_data else 1.7
        
        gamma_dead = st.number_input("DL Factor", value=val_dl, step=0.1)
        gamma_live = st.number_input("LL Factor", value=val_ll, step=0.1)
        
        # Units
        st.subheader("3. Units")
        u_force = st.selectbox("Force Unit", ["kN", "N", "kg"])
        
        # Return params + loaded_data (ส่งต่อให้ฟังก์ชันอื่นใช้)
        return {
            "E": E, "I": I_g, "b": b, "h": h,
            "gamma_dead": gamma_dead, "gamma_live": gamma_live,
            "u_force": u_force, "u_len": "m",
            "loaded_data": loaded_data  # ส่งข้อมูลที่โหลดมาออกไปข้างนอก
        }

def render_model_inputs(params):
    st.subheader("1. Geometry & Supports")
    
    loaded = params.get('loaded_data') # ดึงข้อมูลที่โหลดมา (ถ้ามี)
    
    # Set default values based on loaded file
    def_n = len(loaded['spans']) if loaded else 2
    
    col1, col2 = st.columns([1, 2])
    with col1:
        n_spans = st.number_input("Number of Spans", min_value=1, max_value=10, value=def_n)
    
    spans = []
    with col2:
        cols = st.columns(n_spans)
        for i in range(n_spans):
            # ดึงค่า span จากไฟล์ ถ้ามี
            val_span = loaded['spans'][i] if loaded and i < len(loaded['spans']) else 5.0
            spans.append(cols[i].number_input(f"L{i+1}", min_value=1.0, value=float(val_span), key=f"span_{i}"))

    # Support Config
    st.markdown("##### Support Configuration")
    sup_data = []
    num_nodes = n_spans + 1
    
    cols_sup = st.columns(num_nodes)
    possible_sups = ["Pin", "Roller", "Fixed", "None"]
    
    for i in range(num_nodes):
        # Logic เลือก Default Support จากไฟล์โหลด
        def_idx = 0 # Default Pin
        if i > 0: def_idx = 1 # Default Roller
        
        if loaded and i < len(loaded['supports']):
            type_loaded = loaded['supports'][i]['type']
            if type_loaded in possible_sups:
                def_idx = possible_sups.index(type_loaded)
        
        s_type = cols_sup[i].selectbox(f"Node {i+1}", possible_sups, index=def_idx, key=f"sup_{i}")
        if s_type != "None":
            sup_data.append({"id": i, "type": s_type})
    
    sup_df = pd.DataFrame(sup_data)
    stable = True
    if len(sup_data) < 2:
        if not (len(sup_data) == 1 and sup_data[0]['type'] == 'Fixed'):
            stable = False
            
    return n_spans, spans, sup_df, stable

def render_loads(n_spans, spans, params, sup_df):
    st.subheader("2. Applied Loads")
    
    # Logic การ Load Data เข้า session_state ครั้งเดียว
    if "load_list" not in st.session_state:
        st.session_state.load_list = []

    # ถ้ามีการโหลดไฟล์มา และ session_state ยังว่างอยู่ (หรือผู้ใช้กดโหลดใหม่)
    loaded = params.get('loaded_data')
    if loaded and "data_loaded_flag" not in st.session_state:
        st.session_state.load_list = loaded['loads']
        st.session_state.data_loaded_flag = True # mark ว่าโหลดแล้ว
        st.rerun() # รีเฟรชเพื่อแสดงผลทันที

    # ... (ส่วน Form Input โค้ดเดิม เป๊ะๆ ไม่ต้องแก้) ...
    with st.form("add_load_form"):
        c1, c2, c3, c4, c5 = st.columns([1, 1.2, 1, 1, 1])
        span_choice = c1.selectbox("Span No.", options=list(range(1, n_spans+1)))
        l_type = c2.selectbox("Type", ["Point (P)", "Uniform (w)", "Moment (M)"])
        l_case = c3.selectbox("Case", ["DL (Dead)", "LL (Live)"]) 
        mag = c4.number_input(f"Mag ({params['u_force']})", value=1000.0)
        current_span_len = spans[span_choice-1]
        x_loc = c5.number_input("Dist x (m)", value=current_span_len/2, max_value=float(current_span_len))
        
        dist_load = 0.0
        if "Uniform" in l_type: dist_load = current_span_len - x_loc

        if st.form_submit_button("➕ Add Load"):
            type_code = 'P'
            if "Uniform" in l_type: type_code = 'U'
            elif "Moment" in l_type: type_code = 'M'
            
            st.session_state.load_list.append({
                "span_index": span_choice - 1,
                "type": type_code,
                "case": "DL" if "DL" in l_case else "LL",
                "mag": mag,
                "x": x_loc,
                "dist": dist_load
            })
            # ลบ flag โหลดข้อมูล เพื่อให้รู้ว่ามีการแก้ไขแล้ว
            if "data_loaded_flag" in st.session_state:
                del st.session_state.data_loaded_flag
            st.rerun()
            
    # Display Loads & Save Button
    if st.session_state.load_list:
        loads_df = pd.DataFrame(st.session_state.load_list)
        for i, l in enumerate(st.session_state.load_list):
            l_text = f"Span {l['span_index']+1}: {l['type']} = {l['mag']} ({l['case']}) @ x={l['x']:.2f}"
            c_del_1, c_del_2 = st.columns([8, 1])
            c_del_1.text(l_text)
            if c_del_2.button("❌", key=f"del_{i}"):
                st.session_state.load_list.pop(i)
                st.rerun()
        
        # --- SAVE BUTTON (New!) ---
        st.markdown("---")
        # เตรียมข้อมูล JSON
        json_file = file_manager.export_data(params, spans, sup_df, loads_df)
        st.download_button(
            label="💾 Save Project (.json)",
            data=json_file,
            file_name="beam_project.json",
            mime="application/json",
            type="primary" # ปุ่มสีแดงเด่นๆ
        )
        
        return loads_df
    return None
