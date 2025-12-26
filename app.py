import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from anastruct import SystemElements

# ==========================================
# PART 1: CONFIGURATION & UTILS
# ==========================================
st.set_page_config(page_title="Continuous Beam Design (Pro)", layout="wide")

# Thai Code Load Combinations (USD Method)
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
    วิเคราะห์คานโดยใช้ anaStruct (2D FEM)
    พร้อมระบบป้องกัน Stability Error และ Node Indexing Fix
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
        node_id = i + 1  # <--- Fix: Node เริ่มที่ 1
        
        # --- Stability Guard ---
        # ถ้า Node แรกเป็น Roller ระบบจะคำนวณไม่ได้ (Unstable mechanism)
        # ต้องบังคับให้เป็น Pin เพื่อล็อคแกน X (ไม่มีผลต่อ Moment)
        if i == 0 and supp_type == 'Roller':
            supp_type = 'Pin' 
        # -----------------------

        if supp_type == 'Fix':
            ss.add_support_fixed(node_id=node_id)
        elif supp_type == 'Pin':
            ss.add_support_hinged(node_id=node_id)
        elif supp_type == 'Roller':
            ss.add_support_roll(node_id=node_id, direction=1) 

    # 3. ใส่ Loads (Apply Load Combination)
    for load in loads_data:
        mag_dead = load['dl'] + load['sdl']
        mag_live = load['ll']
        wu_total = (FACTOR_DL * mag_dead) + (FACTOR_LL * mag_live)
        
        span_idx = load['span_idx']
        element_id = span_idx + 1 # Element ID เริ่มที่ 1
        
        if load['type'] == 'Uniform Load':
            # q_load: anastruct convention
            ss.q_load(q=wu_total, element_id=element_id)
            
        elif load['type'] == 'Point Load':
            # Fy ติดลบ = ทิศลง
            ss.point_load(node_id=None, element_id=element_id, position=load['pos'], Fy=-wu_total)
    
    # 4. Analyze
    ss.solve()
    
    return ss

def get_detailed_results(ss):
    """
    ดึงค่า Shear/Moment จากทุก Element มาต่อกันเป็นกราฟยาวเส้นเดียว
    เพื่อให้ Plotly วาดกราฟต่อเนื่องได้ละเอียด
    """
    x_vals = []
    shear_vals = []
    moment_vals = []
    
    # เรียง Element ตามลำดับพิกัด X (เพื่อให้กราฟไม่กระโดด)
    sorted_elements = sorted(ss.element_map.values(), key=lambda e: e.vertex_1.loc[0])
    
    for el in sorted_elements:
        x0 = el.vertex_1.loc[0]
        x1 = el.vertex_2.loc[0]
        
        # ดึงค่า Force Array
        s_arr = np.array(el.shear).flatten()
        m_arr = np.array(el.moment).flatten()
        
        # สร้าง x array สำหรับ element นี้
        # anastruct แบ่ง element ย่อยข้างใน เราต้อง map x ให้ตรงกัน
        x_arr = np.linspace(x0, x1, len(s_arr))
        
        x_vals.extend(x_arr)
        shear_vals.extend(s_arr)
        moment_vals.extend(m_arr)
        
    return pd.DataFrame({
        "x": x_vals,
        "shear": shear_vals,
        "moment": moment_vals
    })

def plot_interactive(df, y_col, title, color_line, y_label):
    """ฟังก์ชันสร้างกราฟ Interactive ด้วย Plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['x'], 
        y=df[y_col],
        mode='lines',
        name=title,
        line=dict(color=color_line, width=2),
        fill='tozeroy', 
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Distance (m)",
        yaxis_title=y_label,
        hovermode="x unified",
        showlegend=False,
        height=400
    )
    return fig

# ==========================================
# PART 3: RC DESIGN ENGINE (USD METHOD)
# ==========================================
def design_rc_beam(mu_kNm, vu_kN, b, h, cover, fc, fy):
    """ออกแบบหน้าตัดคอนกรีตเสริมเหล็ก (USD Method)"""
    d = h - cover
    phi_b = 0.90
    phi_v = 0.85
    
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
    
    Vc = 0.17 * np.sqrt(fc) * b * d
    phi_Vc = phi_v * Vc / 1000 
    
    message_shear = "OK (Concrete Only)"
    if abs(vu_kN) > phi_Vc:
        vs_req = (abs(vu_kN) - phi_Vc) / phi_v
        message_shear = f"Need Stirrup (Vs = {vs_req:.2f} kN)"
    
    status = "OK"
    if rho == 0 and mu_kNm > 0.5:
        status = "Fails (Section too small)"
        
    return {
        "As_req_cm2": As_final / 100,
        "Rho": rho,
        "Phi_Vc_kN": phi_Vc,
        "Status": status,
        "Shear_Msg": message_shear
    }

# ==========================================
# PART 4: MAIN UI APPLICATION
# ==========================================

st.title("🏗️ Continuous Beam Analysis & Design")
st.caption("Interactive FEM Engine | RC Design (EIT/ACI)")

tab1, tab2, tab3 = st.tabs(["1. Input Data", "2. Analysis Results", "3. Concrete Design"])

# --- TAB 1: INPUT ---
with tab1:
    st.header("⚙️ กำหนดค่าโครงสร้าง")
    col1, col2 = st.columns(2)
    with col1:
        num_spans = st.number_input("จำนวนช่วงคาน", 1, 10, 2)
    
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
        supports.append(cols_supp[i].selectbox(f"Supp {i+1}", opts, index=def_idx, key=f"sup{i}"))
        
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
    try:
        st.session_state['ss_model'] = analyze_structure(spans, supports, loads)
    except Exception as e:
        st.error(f"Analysis Error: {e}")

# --- TAB 2: ANALYSIS ---
with tab2:
    if 'analyzed' in st.session_state and 'ss_model' in st.session_state:
        ss = st.session_state['ss_model']
        st.header("📊 Interactive Results")
        
        # Extract Data
        df_res = get_detailed_results(ss)
        max_m = df_res['moment'].abs().max()
        max_v = df_res['shear'].abs().max()
        
        c1, c2 = st.columns(2)
        c1.metric("Max Moment (|Mu|)", f"{max_m:.2f} kN-m")
        c2.metric("Max Shear (|Vu|)", f"{max_v:.2f} kN")
        
        # Plot Shear
        st.subheader("Shear Force Diagram (SFD)")
        fig_v = plot_interactive(df_res, 'shear', "Shear Force (kN)", "#FF4B4B", "Shear (kN)")
        st.plotly_chart(fig_v, use_container_width=True)
        
        # Plot Moment
        st.subheader("Bending Moment Diagram (BMD)")
        # Note: anastruct convention (Sagging +, Hogging -)
        fig_m = plot_interactive(df_res, 'moment', "Bending Moment (kN-m)", "#1f77b4", "Moment (kN-m)")
        st.plotly_chart(fig_m, use_container_width=True)
        
        with st.expander("Show Raw Data Table"):
            st.dataframe(df_res)
        
        st.session_state['max_moment'] = max_m
        st.session_state['max_shear'] = max_v
        
    else:
        st.info("Please click 'Run Analysis' first.")

# --- TAB 3: DESIGN ---
with tab3:
    if 'max_moment' in st.session_state:
        st.header("🧱 RC Design")
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
        
        if "Need Stirrup" in res['Shear_Msg']:
            st.error(res['Shear_Msg'])
        else:
            st.success(res['Shear_Msg'])
    else:
        st.warning("No analysis data found.")
