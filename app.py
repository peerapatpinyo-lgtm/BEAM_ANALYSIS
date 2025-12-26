import streamlit as st
import matplotlib.pyplot as plt
from anastruct import SystemElements
import numpy as np
import pandas as pd

# ==========================================
# PART 1: CONFIGURATION & UTILS
# ==========================================
st.set_page_config(page_title="Continuous Beam Design (AnaStruct)", layout="wide")

# Thai Code Load Combinations (EIT Standard / USD Method)
FACTOR_DL = 1.4
FACTOR_LL = 1.7

def explain_calc(text):
    """Helper to display calculation logic explanation"""
    with st.expander("📝 ดูหลักการคำนวณ (Calculation Logic)"):
        st.markdown(text)

# ==========================================
# PART 2: ANALYSIS ENGINE (anaStruct)
# ==========================================
def analyze_structure(spans_data, supports_data, loads_data):
    """
    วิเคราะห์คานโดยใช้ anaStruct (2D FEM) - พร้อมระบบป้องกัน Stability Error
    """
    # สร้าง System Model
    ss = SystemElements(EA=15000, EI=5000) 
    
    # 1. สร้าง Elements (คานแต่ละช่วง)
    start_x = 0
    for length in spans_data:
        end_x = start_x + length
        ss.add_element(location=[[start_x, 0], [end_x, 0]])
        start_x = end_x
    
    # 2. ใส่ Supports (จุดรองรับ)
    for i, supp_type in enumerate(supports_data):
        node_id = i + 1
        
        # --- Stability Guard ---
        # ถ้าเป็น Node แรกสุด (Node 1) และผู้ใช้เลือก Roller
        # เราจะบังคับให้เป็น Pin (Hinged) เพื่อล็อคแกน X ไม่ให้คานไหล (Unstable)
        # ซึ่งไม่มีผลต่อ Moment ในคานรับแรงดิ่ง แต่ทำให้ Math คำนวณผ่าน
        if i == 0 and supp_type == 'Roller':
            supp_type = 'Pin' 
        # -----------------------

        if supp_type == 'Fix':
            ss.add_support_fixed(node_id=node_id)
        elif supp_type == 'Pin':
            ss.add_support_hinged(node_id=node_id)
        elif supp_type == 'Roller':
            # Roller direction=1 คือกลิ้งแกน x (รับแรงแกน y)
            ss.add_support_roll(node_id=node_id, direction=1) 

    # 3. ใส่ Loads
    for load in loads_data:
        mag_dead = load['dl'] + load['sdl']
        mag_live = load['ll']
        wu_total = (FACTOR_DL * mag_dead) + (FACTOR_LL * mag_live)
        
        span_idx = load['span_idx']
        element_id = span_idx + 1
        
        if load['type'] == 'Uniform Load':
            # ใส่ load ที่ element
            ss.q_load(q=wu_total, element_id=element_id)
            
        elif load['type'] == 'Point Load':
            # Fy ติดลบ = ทิศลง
            ss.point_load(node_id=None, element_id=element_id, position=load['pos'], Fy=-wu_total)
    
    # 4. Analyze
    ss.solve()
    
    return ss
# ==========================================
# PART 3: RC DESIGN ENGINE (USD METHOD)
# ==========================================
def design_rc_beam(mu_kNm, vu_kN, b, h, cover, fc, fy):
    """
    ออกแบบหน้าตัดคอนกรีตเสริมเหล็ก (USD Method)
    """
    d = h - cover # Effective depth
    phi_b = 0.90  # Flexure reduction factor
    phi_v = 0.85  # Shear reduction factor
    
    # --- Flexure Design ---
    # Convert units: kNm -> N-mm
    mn_req = (abs(mu_kNm) * 10**6) / phi_b 
    
    # Rn = Mn / (b * d^2) (Input b, h in mm -> result MPa)
    # แต่ function รับ b, h เป็น mm (จากการคูณ 10 ที่ UI)
    Rn = mn_req / (b * d**2)
    
    # Rho (Reinforcement Ratio)
    m = fy / (0.85 * fc)
    rho = 0.0
    try:
        term = 1 - (2 * m * Rn) / fy
        if term >= 0:
            rho = (1/m) * (1 - np.sqrt(term))
    except:
        rho = 0.0 # Calculation error (Complex number)
        
    As_req = rho * b * d
    
    # Minimum Steel Check
    as_min = max((np.sqrt(fc)/(4*fy))*b*d, (1.4/fy)*b*d)
    As_final = max(As_req, as_min)
    
    # --- Shear Design ---
    # Vc = 0.17 * sqrt(fc) * b * d (Newton)
    Vc = 0.17 * np.sqrt(fc) * b * d 
    phi_Vc = phi_v * Vc / 1000 # Convert to kN
    
    message_shear = "OK (Concreteรับไหว)"
    vs_req = 0
    if abs(vu_kN) > phi_Vc:
        vs_req = (abs(vu_kN) - phi_Vc) / phi_v
        message_shear = f"Need Stirrup (Vs = {vs_req:.2f} kN)"
    
    status = "OK"
    if rho == 0 and mu_kNm > 0.5: # ถ้า Moment เยอะแต่ rho=0 แสดงว่าหน้าตัดระเบิด
        status = "Fails (Section too small)"
        
    return {
        "As_req_cm2": As_final / 100, # mm2 -> cm2
        "Rho": rho,
        "Phi_Vc_kN": phi_Vc,
        "Status": status,
        "Shear_Msg": message_shear
    }

# ==========================================
# PART 4: MAIN UI APPLICATION
# ==========================================

st.title("🏗️ Continuous Beam Design (AnaStruct Engine)")
st.caption("Robust Cloud-Ready Version | Fixed Node Indexing")

# --- UI TAB SEPARATION ---
tab1, tab2, tab3 = st.tabs(["1. Input Data", "2. Analysis Results", "3. Concrete Design"])

# ---------------------------------------
# TAB 1: INPUT
# ---------------------------------------
with tab1:
    st.header("⚙️ กำหนดค่าโครงสร้าง (Input Parameters)")
    
    col1, col2 = st.columns(2)
    with col1:
        num_spans = st.number_input("จำนวนช่วงคาน (Number of Spans)", min_value=1, max_value=10, value=2)
    
    st.markdown("---")
    
    spans = []
    supports = []
    loads = []
    
    # 1. Span Lengths
    st.subheader("1. ความยาวแต่ละช่วง (Span Lengths)")
    cols_span = st.columns(num_spans)
    for i in range(num_spans):
        l = cols_span[i].number_input(f"Span {i+1} (m)", min_value=1.0, value=4.0, key=f"span_{i}")
        spans.append(l)
        
    # 2. Supports
    st.subheader("2. จุดรองรับ (Supports)")
    cols_supp = st.columns(num_spans + 1)
    support_options = ['Pin', 'Roller', 'Fix']
    for i in range(num_spans + 1):
        default_idx = 0 if i==0 else 1 
        s = cols_supp[i].selectbox(f"Supp {i+1}", support_options, index=default_idx, key=f"supp_{i}")
        supports.append(s)
        
    # 3. Loads
    st.subheader("3. น้ำหนักบรรทุก (Loads)")
    st.info(f"Load Combination: {FACTOR_DL}*(DL+SDL) + {FACTOR_LL}*LL")
    
    for i in range(num_spans):
        with st.expander(f"📍 Load on Span {i+1}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            load_type = c1.selectbox("Load Type", ["Uniform Load", "Point Load"], key=f"ltype_{i}")
            dl = c2.number_input("Dead Load (DL)", value=10.0, key=f"dl_{i}", help="kN/m or kN")
            sdl = c3.number_input("Super Dead Load (SDL)", value=5.0, key=f"sdl_{i}")
            ll = c4.number_input("Live Load (LL)", value=8.0, key=f"ll_{i}")
            
            pos = 0.0
            if load_type == "Point Load":
                pos = st.slider(f"Position from left of Span {i+1} (m)", 0.0, spans[i], spans[i]/2, key=f"pos_{i}")
            
            loads.append({
                "span_idx": i,
                "type": load_type,
                "dl": dl, "sdl": sdl, "ll": ll,
                "pos": pos
            })

    analyze_btn = st.button("🚀 Run Analysis", type="primary")

# Processing Logic
if analyze_btn:
    st.session_state['analyzed'] = True
    try:
        st.session_state['ss_model'] = analyze_structure(spans, supports, loads)
    except Exception as e:
        st.error(f"Analysis Error: {e}")

# ---------------------------------------
# TAB 2: OUTPUT (ANALYSIS)
# ---------------------------------------
with tab2:
    if 'analyzed' in st.session_state and 'ss_model' in st.session_state:
        ss = st.session_state['ss_model']
        
        st.header("📊 ผลการวิเคราะห์ (Analysis Results)")
        
        # Plotting Diagrams using AnaStruct built-in plotting
        # Note: show=False returns the figure object
        
        st.subheader("Shear Force Diagram (SFD)")
        fig_sf = ss.show_shear_force(show=False)
        st.pyplot(fig_sf)
        
        st.subheader("Bending Moment Diagram (BMD)")
        fig_bm = ss.show_bending_moment(show=False)
        st.pyplot(fig_bm)
        
        # --- Extract Values for Design ---
        # AnaStruct stores results in elements. We need to iterate to find Max Abs values.
        max_moment_val = 0.0
        max_shear_val = 0.0
        
        # Iterate over all elements to find global max
        for el in ss.element_map.values():
            # Check Moment (array of values along element)
            if hasattr(el, 'moment') and len(el.moment) > 0:
                current_max_m = np.max(np.abs(el.moment))
                if current_max_m > max_moment_val:
                    max_moment_val = current_max_m
            
            # Check Shear
            if hasattr(el, 'shear') and len(el.shear) > 0:
                current_max_v = np.max(np.abs(el.shear))
                if current_max_v > max_shear_val:
                    max_shear_val = current_max_v
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Max Design Moment (|Mu|)", f"{max_moment_val:.2f} kN-m")
        col_res2.metric("Max Design Shear (|Vu|)", f"{max_shear_val:.2f} kN")
        
        # Store for Design Tab
        st.session_state['max_moment'] = max_moment_val
        st.session_state['max_shear'] = max_shear_val
        
        explain_calc(r"""
        **Analysis Method:** Finite Element Method (2D Frame) via `anastruct` library.
        - **Supports:** Converted to Nodes (Fixed/Hinged/Roller).
        - **Loads:** Applied as Element Loads ($q$) or Nodal Forces.
        """)
        
    else:
        st.info("กรุณากดปุ่ม 'Run Analysis' ในหน้า Input ก่อนครับ")

# ---------------------------------------
# TAB 3: DESIGN
# ---------------------------------------
with tab3:
    st.header("🧱 ออกแบบหน้าตัดคาน (RC Design)")
    
    if 'max_moment' in st.session_state:
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.subheader("Material Properties")
            fc = st.number_input("f'c (MPa)", value=24.0)
            fy = st.number_input("fy (MPa) - Main Rebar", value=400.0)
            
        with col_d2:
            st.subheader("Section Properties")
            b = st.number_input("ความกว้าง b (cm)", value=25.0)
            h = st.number_input("ความลึก h (cm)", value=50.0)
            cover = st.number_input("Covering (cm)", value=4.0)
            
        # Design Calculation
        mu = st.session_state['max_moment']
        vu = st.session_state['max_shear']
        
        st.markdown("---")
        st.subheader("ผลการออกแบบ (Design Output)")
        st.write(f"Designing for Mu: **{mu:.2f} kN-m**, Vu: **{vu:.2f} kN**")
        
        # Convert cm to mm for calculation
        res = design_rc_beam(mu, vu, b*10, h*10, cover*10, fc, fy)
        
        r1, r2, r3 = st.columns(3)
        status_color = "normal" if res['Status']=="OK" else "inverse"
        r1.metric("Status", res['Status'], delta_color=status_color)
        r2.metric("As required", f"{res['As_req_cm2']:.2f} cm²")
        r3.metric("Concrete Shear (Phi Vc)", f"{res['Phi_Vc_kN']:.2f} kN")
        
        if "Need Stirrup" in res['Shear_Msg']:
            st.error(f"Shear Check: {res['Shear_Msg']}")
        else:
            st.success(f"Shear Check: {res['Shear_Msg']}")
            
        explain_calc(r"""
        **USD Method Formulas:**
        1. **Required Moment Strength:** $M_n = M_u / 0.9$
        2. **Reinforcement Ratio ($\rho$):** $$ \rho = \frac{1}{m} \left( 1 - \sqrt{1 - \frac{2m R_n}{f_y}} \right) $$
        3. **Shear Capacity:** $\phi V_c = 0.85 \times 0.17 \sqrt{f_c'} b d$
        """)
        
    else:
        st.warning("รอผลการวิเคราะห์จาก Tab 2 ก่อนครับ")

