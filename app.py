import streamlit as st
import matplotlib.pyplot as plt
from indeterminbeam import Beam, Support, PointLoadV, UDLV
import numpy as np
import pandas as pd

# ==========================================
# PART 1: CONFIGURATION & UTILS
# ==========================================
st.set_page_config(page_title="Continuous Beam Design (Thai Code)", layout="wide")

# Thai Code Load Combinations (EIT Standard / USD Method)
FACTOR_DL = 1.4
FACTOR_LL = 1.7

def explain_calc(text):
    """Helper to display calculation logic explanation"""
    with st.expander("📝 ดูหลักการคำนวณ (Calculation Logic)"):
        st.markdown(text)

# ==========================================
# PART 2: ANALYSIS ENGINE (ENGINEERING LOGIC)
# ==========================================
def analyze_structure(spans_data, supports_data, loads_data):
    """
    วิเคราะห์หา Moment และ Shear โดยใช้ indeterminbeam
    """
    # 1. Setup Beam Span
    # indeterminbeam ใช้พิกัดรวม (Global Coordinate) เราต้องแปลงความยาวช่วงเป็นพิกัด
    span_coords = [0]
    current_length = 0
    for length in spans_data:
        current_length += length
        span_coords.append(current_length)
    
    beam = Beam(current_length)
    
    # 2. Setup Supports
    # Supports: Fix (1,1,1), Pin (1,1,0), Roller (0,1,0) -> (x, y, moment)
    support_objects = []
    for i, supp_type in enumerate(supports_data):
        coord = span_coords[i]
        if supp_type == 'Fix':
            # Fix ต้องรับ Force Y และ Moment (ใน library นี้ fix X ไม่จำเป็นสำหรับคานแนวราบล้วนๆ แต่ใส่ไว้ได้)
            support_objects.append(Support(coord, (1, 1, 1)))
        elif supp_type == 'Pin':
            support_objects.append(Support(coord, (1, 1, 0)))
        elif supp_type == 'Roller':
            support_objects.append(Support(coord, (0, 1, 0)))
            
    beam.add_supports(*support_objects)
    
    # 3. Setup Loads (Apply Load Combination here)
    # เราจะ Combine load ก่อนวิเคราะห์เพื่อให้ได้ Ultimate Internal Forces
    load_objects = []
    
    for load in loads_data:
        # load = {'type': 'UDL/Point', 'dl': val, 'sdl': val, 'll': val, 'pos': val, 'span_idx': val}
        
        # Calculate Factored Load (Wu)
        # Wu = 1.4(DL + SDL) + 1.7(LL)
        mag_dead = load['dl'] + load['sdl']
        mag_live = load['ll']
        factored_load = (FACTOR_DL * mag_dead) + (FACTOR_LL * mag_live)
        
        # Determine Global Position based on span index
        start_x = span_coords[load['span_idx']]
        
        if load['type'] == 'Uniform Load':
            # UDLV (Uniform Distributed Load Vertical)
            # รับค่าเป็น (force, coordinate range)
            span_len = spans_data[load['span_idx']]
            # ใส่เครื่องหมายลบ เพราะ load ลงทิศล่าง
            load_objects.append(UDLV(-factored_load, (start_x, start_x + span_len)))
            
        elif load['type'] == 'Point Load':
            # PointLoadV
            # pos คือระยะจากซ้ายสุดของ span นั้นๆ
            abs_pos = start_x + load['pos']
            load_objects.append(PointLoadV(-factored_load, abs_pos))
            
    beam.add_loads(*load_objects)
    
    # 4. Analyze
    beam.analyse()
    
    return beam

# ==========================================
# PART 3: RC DESIGN ENGINE (SDM / USD METHOD)
# ==========================================
def design_rc_beam(mu_kNm, vu_kN, b, h, cover, fc, fy):
    """
    ออกแบบหน้าตัดคอนกรีตเสริมเหล็ก (USD Method)
    """
    d = h - cover # Effective depth
    phi_b = 0.90  # Flexure reduction factor
    phi_v = 0.85  # Shear reduction factor
    
    # --- Flexure Design ---
    # Mu = phi * Mn
    # Mn_req = Mu / phi
    mn_req = (abs(mu_kNm) * 10**6) / phi_b # N-mm
    
    # Rn = Mn / (b * d^2)
    Rn = mn_req / (b * d**2)
    
    # Rho (Reinforcement Ratio)
    m = fy / (0.85 * fc)
    try:
        rho = (1/m) * (1 - np.sqrt(1 - (2 * m * Rn) / fy))
    except:
        rho = 0.0 # กรณีรากติดลบ (หน้าตัดเล็กไป)
        
    As_req = rho * b * d
    
    # Minimum Steel Check (Simplified ACI/EIT)
    as_min = max((np.sqrt(fc)/(4*fy))*b*d, (1.4/fy)*b*d)
    As_final = max(As_req, as_min)
    
    # --- Shear Design ---
    # Vc = 0.53 * sqrt(fc) * b * d (หน่วย kg/cm2 -> ต้องแปลงหน่วยให้ตรง)
    # ใช้สูตร SI: Vc = 0.17 * sqrt(fc) * b * d (N)
    Vc = 0.17 * np.sqrt(fc) * b * d # Newton
    phi_Vc = phi_v * Vc / 1000 # kN
    
    vs_req = 0
    message_shear = "OK (Concreteรับไหว)"
    if abs(vu_kN) > phi_Vc:
        vs_req = (abs(vu_kN) - phi_Vc) / phi_v
        message_shear = f"Need Stirrup (Vs = {vs_req:.2f} kN)"
    
    status = "OK"
    if rho == 0:
        status = "Fails (Section too small)"
        
    return {
        "As_req_cm2": As_final / 100, # Convert mm2 to cm2
        "Rho": rho,
        "Phi_Vc_kN": phi_Vc,
        "Status": status,
        "Shear_Msg": message_shear
    }

# ==========================================
# PART 4: MAIN UI APPLICATION
# ==========================================

st.title("🏗️ Continuous Beam Design (USD Method)")
st.caption("Developed by Senior Structural Engineer & Dev")

# --- UI TAB SEPARATION ---
tab1, tab2, tab3 = st.tabs(["1. Input Data", "2. Analysis Results", "3. Concrete Design"])

# ---------------------------------------
# TAB 1: INPUT
# ---------------------------------------
with tab1:
    st.header("⚙️ กำหนดค่าโครงสร้าง (Input Parameters)")
    
    col1, col2 = st.columns(2)
    with col1:
        num_spans = st.number_input("จำนวนช่วงคาน (Number of Spans)", min_value=2, max_value=10, value=2)
    
    st.markdown("---")
    
    # Dynamic Input Containers
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
    # มี support จำนวน num_spans + 1
    cols_supp = st.columns(num_spans + 1)
    support_options = ['Pin', 'Roller', 'Fix']
    for i in range(num_spans + 1):
        default_idx = 0 if i==0 else 1 # Fix/Pin start, Roller others typically
        s = cols_supp[i].selectbox(f"Supp {i}", support_options, index=default_idx, key=f"supp_{i}")
        supports.append(s)
        
    # 3. Loads
    st.subheader("3. น้ำหนักบรรทุก (Loads)")
    st.info(f"Load Combination (Thai Code): {FACTOR_DL}*(DL+SDL) + {FACTOR_LL}*LL")
    
    for i in range(num_spans):
        with st.expander(f"📍 Load on Span {i+1}", expanded=True):
            # Allow multiple loads per span? For simplicity here, assume 1 Main UDL + Optional Point
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

    # Button to Trigger Analysis
    analyze_btn = st.button("🚀 Run Analysis", type="primary")

# ---------------------------------------
# LOGIC PROCESSING
# ---------------------------------------
if analyze_btn:
    st.session_state['analyzed'] = True
    st.session_state['beam_model'] = analyze_structure(spans, supports, loads)
    st.session_state['input_config'] = {'spans': spans} # Save for design usage

# ---------------------------------------
# TAB 2: OUTPUT (ANALYSIS)
# ---------------------------------------
with tab2:
    if 'analyzed' in st.session_state and st.session_state['analyzed']:
        beam = st.session_state['beam_model']
        
        st.header("📊 ผลการวิเคราะห์ (Analysis Results)")
        
        # Plotting Diagrams
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Indeterminbeam plotting helper
        beam.plot_shear_force(ax=ax1)
        ax1.set_title("Shear Force Diagram (SFD)")
        ax1.set_ylabel("Shear (kN)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        beam.plot_bending_moment(ax=ax2)
        ax2.set_title("Bending Moment Diagram (BMD)")
        ax2.set_ylabel("Moment (kN-m)")
        ax2.set_xlabel("Length (m)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        st.pyplot(fig)
        
        # Get Max Values for Design
        try:
            # indeterminbeam 2.x returns min/max
            v_max = beam.get_shear_force(return_max=True)
            v_min = beam.get_shear_force(return_min=True)
            m_max = beam.get_bending_moment(return_max=True)
            m_min = beam.get_bending_moment(return_min=True)
            
            st.metric("Max Positive Moment (+M)", f"{m_max:.2f} kN-m")
            st.metric("Max Negative Moment (-M)", f"{m_min:.2f} kN-m")
            st.metric("Max Shear (V)", f"{max(abs(v_max), abs(v_min)):.2f} kN")
            
            # Store max absolute values for simple design
            st.session_state['max_moment'] = max(abs(m_max), abs(m_min))
            st.session_state['max_shear'] = max(abs(v_max), abs(v_min))
            
        except Exception as e:
            st.error(f"Error extracting values: {e}")

        explain_calc(f"""
        **Calculation Details:**
        - **Load Combination:** $1.4(DL+SDL) + 1.7(LL)$
        - **Method:** Stiffness Matrix Method (via `indeterminbeam` library)
        - **Supports:** Converted to stiffness nodes (Fix/Pin/Roller)
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
            fy = st.number_input("fy (MPa) - เหล็กเสริมหลัก", value=400.0)
            
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
        st.write(f"Based on Max Moment: **{mu:.2f} kN-m** and Max Shear: **{vu:.2f} kN**")
        
        # Convert units for calculation (input cm -> mm, output cm2)
        res = design_rc_beam(mu, vu, b*10, h*10, cover*10, fc, fy)
        
        r1, r2, r3 = st.columns(3)
        r1.metric("Status", res['Status'], delta_color="normal" if res['Status']=="OK" else "inverse")
        r2.metric("เหล็กเสริมที่ต้องการ (As req)", f"{res['As_req_cm2']:.2f} cm²")
        r3.metric("กำลังรับแรงเฉือนคอนกรีต", f"{res['Phi_Vc_kN']:.2f} kN")
        
        st.info(f"Shear Check: {res['Shear_Msg']}")
        
        # Visualization of Section
        #  - Conceptually represented by text/plot below
        st.caption("ℹ️ เหล็กเสริมที่คำนวณได้เป็นปริมาณเหล็กรับแรงดึงสูงสุด (Bottom or Top) ตาม Moment ที่มากที่สุด")
        
        explain_calc(r"""
        **Design Formulas (USD Method):**
        1. **Effective Depth:** $d = h - cover$
        2. **Required Moment Strength:** $M_n = M_u / \phi_b$ ($\phi_b=0.9$)
        3. **Reinforcement Ratio ($\rho$):**
           $$ \rho = \frac{1}{m} \left( 1 - \sqrt{1 - \frac{2m R_n}{f_y}} \right) $$
           Where $m = \frac{f_y}{0.85 f_c'}$ and $R_n = \frac{M_n}{b d^2}$
        4. **Shear Capacity:** $\phi V_c = 0.85 \times 0.17 \sqrt{f_c'} b d$
        """)
        
    else:
        st.warning("รอผลการวิเคราะห์จาก Tab 2 ก่อนครับ")
