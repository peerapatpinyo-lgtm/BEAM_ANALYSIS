import streamlit as st
import matplotlib.pyplot as plt
from anastruct import SystemElements
import numpy as np
import pandas as pd

# ==========================================
# PART 1: CONFIGURATION & UTILS
# ==========================================
st.set_page_config(page_title="Continuous Beam Design (AnaStruct)", layout="wide")

FACTOR_DL = 1.4
FACTOR_LL = 1.7

def explain_calc(text):
    with st.expander("📝 ดูหลักการคำนวณ (Calculation Logic)"):
        st.markdown(text)

# ==========================================
# PART 2: ANALYSIS ENGINE (anaStruct)
# ==========================================
def analyze_structure(spans_data, supports_data, loads_data):
    """
    วิเคราะห์คานโดยใช้ anaStruct (2D FEM)
    """
    ss = SystemElements(EA=15000, EI=5000) # กำหนดค่า stiffness สมมติ (ไม่มีผลต่อ Force ในคาน Determinate/Indeterminate ทั่วไปถ้า EI คงที่)
    
    # 1. สร้าง Elements (คานแต่ละช่วง)
    # anaStruct สร้าง node อัตโนมัติ: Node 0, 1, 2...
    start_x = 0
    for length in spans_data:
        end_x = start_x + length
        # สร้าง element จากซ้ายไปขวา
        ss.add_element(location=[[start_x, 0], [end_x, 0]])
        start_x = end_x
    
    # 2. ใส่ Supports (จุดรองรับ)
    # Node id จะเรียงตามจุดต่อ: 0, 1, 2, ...
    for i, supp_type in enumerate(supports_data):
        node_id = i
        if supp_type == 'Fix':
            ss.add_support_fixed(node_id=node_id)
        elif supp_type == 'Pin':
            ss.add_support_hinged(node_id=node_id)
        elif supp_type == 'Roller':
            # Roller ใน anaStruct: direction=1 คือกลิ้งได้แกน x (รับแรงแกน y), direction=2 คือกลิ้งได้แกน y
            ss.add_support_roll(node_id=node_id, direction=1) 

    # 3. ใส่ Loads (Apply Load Combination)
    for load in loads_data:
        # Load Combination
        mag_dead = load['dl'] + load['sdl']
        mag_live = load['ll']
        wu_total = (FACTOR_DL * mag_dead) + (FACTOR_LL * mag_live)
        
        span_idx = load['span_idx']
        # anaStruct ใช้ Element ID เริ่มที่ 1, 2, 3... (ต่างจาก index list 0,1,2)
        element_id = span_idx + 1 
        
        if load['type'] == 'Uniform Load':
            # q_load (แรงกระจาย) ทิศลงค่าต้องเป็นลบ? ใน anaStruct q ใส่เป็นค่า + คือ load ลง (Gravity) *เช็ค doc แล้ว*
            # แต่เพื่อความชัวร์ anaStruct Convention: q=positive acts against y-axis (downwards)
            ss.q_load(q=wu_total, element_id=element_id)
            
        elif load['type'] == 'Point Load':
            # point_load: node_id หรือ position
            # ใส่ load ที่ element โดยระบุระยะจาก node ซ้าย
            # Fx, Fy (Fy ลบ คือทิศลง)
            ss.point_load(node_id=None, element_id=element_id, position=load['pos'], Fy=-wu_total)
    
    # 4. Analyze
    ss.solve()
    
    return ss

# ==========================================
# PART 3: RC DESIGN ENGINE (USD METHOD)
# ==========================================
def design_rc_beam(mu_kNm, vu_kN, b, h, cover, fc, fy):
    d = h - cover
    phi_b = 0.90
    phi_v = 0.85
    
    # Flexure
    mn_req = (abs(mu_kNm) * 10**6) / phi_b
    Rn = mn_req / (b * d**2)
    m = fy / (0.85 * fc)
    
    rho = 0.0
    try:
        val_root = 1 - (2 * m * Rn) / fy
        if val_root >= 0:
            rho = (1/m) * (1 - np.sqrt(val_root))
    except:
        pass
        
    As_req = rho * b * d
    as_min = max((np.sqrt(fc)/(4*fy))*b*d, (1.4/fy)*b*d)
    As_final = max(As_req, as_min)
    
    # Shear
    Vc = 0.17 * np.sqrt(fc) * b * d
    phi_Vc = phi_v * Vc / 1000
    
    message_shear = "OK (Concrete Only)"
    if abs(vu_kN) > phi_Vc:
        vs_req = (abs(vu_kN) - phi_Vc) / phi_v
        message_shear = f"Need Stirrup (Vs = {vs_req:.2f} kN)"
    
    status = "OK"
    if rho == 0 and mu_kNm > 0.1: # ignore extremely small moment
        status = "Fails (Section too small)"
        
    return {
        "As_req_cm2": As_final / 100,
        "Rho": rho,
        "Phi_Vc_kN": phi_Vc,
        "Status": status,
        "Shear_Msg": message_shear
    }

# ==========================================
# PART 4: UI
# ==========================================

st.title("🏗️ Continuous Beam Design (AnaStruct Engine)")
st.caption("Robust Cloud-Ready Version")

tab1, tab2, tab3 = st.tabs(["1. Input Data", "2. Analysis Results", "3. Concrete Design"])

with tab1:
    st.header("⚙️ กำหนดค่าโครงสร้าง")
    col1, col2 = st.columns(2)
    with col1:
        num_spans = st.number_input("จำนวนช่วงคาน", 2, 10, 2)
    
    spans = []
    supports = []
    loads = []
    
    st.subheader("1. ความยาว (Span Lengths)")
    cols_span = st.columns(num_spans)
    for i in range(num_spans):
        spans.append(cols_span[i].number_input(f"Span {i+1} (m)", 1.0, value=4.0, key=f"s{i}"))
        
    st.subheader("2. จุดรองรับ (Supports)")
    cols_supp = st.columns(num_spans + 1)
    opts = ['Pin', 'Roller', 'Fix']
    for i in range(num_spans + 1):
        def_idx = 0 if i==0 else 1
        supports.append(cols_supp[i].selectbox(f"Supp {i}", opts, index=def_idx, key=f"sup{i}"))
        
    st.subheader("3. Loads (1.4DL + 1.7LL)")
    for i in range(num_spans):
        with st.expander(f"📍 Load Span {i+1}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            ltype = c1.selectbox("Type", ["Uniform Load", "Point Load"], key=f"lt{i}")
            dl = c2.number_input("DL", value=10.0, key=f"d{i}")
            sdl = c3.number_input("SDL", value=5.0, key=f"sd{i}")
            ll = c4.number_input("LL", value=8.0, key=f"l{i}")
            pos = 0.0
            if ltype == "Point Load":
                pos = st.slider(f"Position (m)", 0.0, spans[i], spans[i]/2.0, key=f"p{i}")
            
            loads.append({"span_idx": i, "type": ltype, "dl": dl, "sdl": sdl, "ll": ll, "pos": pos})

    analyze_btn = st.button("🚀 Run Analysis", type="primary")

if analyze_btn:
    st.session_state['analyzed'] = True
    st.session_state['ss_model'] = analyze_structure(spans, supports, loads)

with tab2:
    if 'analyzed' in st.session_state:
        ss = st.session_state['ss_model']
        st.header("📊 Analysis Results")
        
        # Plotting using AnaStruct's built-in matplotlib wrapper
        # 1. Shear
        st.subheader("Shear Force Diagram")
        fig_sf = ss.show_shear_force(show=False)
        st.pyplot(fig_sf)
        
        # 2. Moment
        st.subheader("Bending Moment Diagram")
        fig_bm = ss.show_bending_moment(show=False)
        st.pyplot(fig_bm)
        
        # Extract Min/Max for Design
        # anaStruct stores results in nodes/elements range. 
        # We can extract moment array from elements
        m_vals = []
        v_vals = []
        for el in ss.element_map.values():
            # shear and moment arrays
            # note: anaStruct conventions might differ slightly, checking absolute max is safe
            m_vals.extend(el.moment_map.values())
            v_vals.extend(el.shear_map.values())
            
        # หาค่า Max Absolute
        m_max_abs = max([abs(x) for x in m_vals]) if m_vals else 0
        v_max_abs = max([abs(x) for x in v_vals]) if v_vals else 0
        
        c1, c2 = st.columns(2)
        c1.metric("Max Moment (Design)", f"{m_max_abs:.2f} kN-m")
        c2.metric("Max Shear (Design)", f"{v_max_abs:.2f} kN")
        
        st.session_state['max_moment'] = m_max_abs
        st.session_state['max_shear'] = v_max_abs
        
    else:
        st.info("Please Run Analysis")

with tab3:
    if 'max_moment' in st.session_state:
        st.header("🧱 Design RC")
        mu = st.session_state['max_moment']
        vu = st.session_state['max_shear']
        
        c1, c2 = st.columns(2)
        with c1:
            fc = st.number_input("fc (MPa)", value=24.0)
            fy = st.number_input("fy (MPa)", value=400.0)
        with c2:
            b = st.number_input("b (cm)", value=25.0)
            h = st.number_input("h (cm)", value=50.0)
            cover = st.number_input("cover (cm)", value=4.0)
            
        res = design_rc_beam(mu, vu, b*10, h*10, cover*10, fc, fy)
        
        st.write("---")
        c3, c4, c5 = st.columns(3)
        c3.metric("Status", res['Status'])
        c4.metric("As req", f"{res['As_req_cm2']:.2f} cm²")
        c5.metric("Phi Vc", f"{res['Phi_Vc_kN']:.2f} kN")
        st.warning(f"Shear Check: {res['Shear_Msg']}")
