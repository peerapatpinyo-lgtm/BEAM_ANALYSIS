import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from anastruct import SystemElements

# ==========================================
# PART 1: SETUP & CONSTANTS
# ==========================================
st.set_page_config(page_title="Continuous Beam Design", layout="wide")

# Load Factors (Thai/EIT USD Method)
FACTOR_DL = 1.4
FACTOR_LL = 1.7

# ==========================================
# PART 2: ANALYSIS ENGINE
# ==========================================
def analyze_structure(spans, supports, loads):
    # mesh=50 ช่วยให้กราฟละเอียดโค้งเนียน
    ss = SystemElements(EA=15000, EI=5000, mesh=50) 
    
    # 1. Elements (สร้างคานตามช่วง)
    start_x = 0
    for L in spans:
        ss.add_element(location=[[start_x, 0], [start_x + L, 0]])
        start_x += L
        
    # 2. Stability Check & Supports
    # ป้องกันโครงสร้าง Unstable โดยบังคับ Node แรกเป็น Pin หากไม่มีตัวยึดแกน X
    if not any(s in ['Pin', 'Fix'] for s in supports):
        supports[0] = 'Pin'
        st.toast("⚠️ Auto-fixed: Changed first support to 'Pin' for stability.", icon="🔧")
        
    for i, s_type in enumerate(supports):
        node_id = i + 1
        if s_type == 'Fix': ss.add_support_fixed(node_id=node_id)
        elif s_type == 'Pin': ss.add_support_hinged(node_id=node_id)
        elif s_type == 'Roller': ss.add_support_roll(node_id=node_id, direction=2) # 2=Vertical Support

    # 3. Loads
    for load in loads:
        # รวม Load: 1.4(DL + SDL) + 1.7(LL)
        w_total = (FACTOR_DL * (load['dl'] + load['sdl'])) + (FACTOR_LL * load['ll'])
        elem_id = load['span_idx'] + 1
        
        if load['type'] == 'Uniform Load':
            ss.q_load(q=w_total, element_id=elem_id)
        elif load['type'] == 'Point Load':
            ss.point_load(node_id=None, element_id=elem_id, position=load['pos'], Fy=-w_total)

    # 4. Solve
    ss.solve()
    return ss

def extract_plot_data(ss, num_spans):
    """
    ดึงข้อมูลกราฟแบบปลอดภัย (แก้บั๊ก Vertex object has no attribute)
    ใช้วิธีเรียก Element ID ตรงๆ แทนการ sort object
    """
    # Trigger calculation
    try:
        ss.show_shear_force(show=False)
        ss.show_bending_moment(show=False)
        plt.close('all')
    except: pass

    x_list, v_list, m_list = [], [], []
    
    # Iterate ตาม ID ของ Element (1 ถึง num_spans) ชัวร์กว่าการวน Loop map
    for i in range(1, num_spans + 1):
        if i in ss.element_map:
            el = ss.element_map[i]
            
            # หาพิกัดเริ่มต้นของ Element นี้
            # พยายามดึง loc หรือ coordinates แล้วแต่ version
            if hasattr(el.vertex_1, 'loc'):
                x0 = el.vertex_1.loc[0]
                x1 = el.vertex_2.loc[0]
            else:
                x0 = el.vertex_1.coordinates[0]
                x1 = el.vertex_2.coordinates[0]

            # ดึงค่าแรง (Array)
            v = np.array(getattr(el, 'shear', [])).flatten()
            m = np.array(getattr(el, 'moment', [])).flatten()
            
            # ถ้ามีข้อมูล ให้เก็บเข้า List
            if len(v) > 0:
                x = np.linspace(x0, x1, len(v))
                x_list.extend(x)
                v_list.extend(v)
                m_list.extend(m)
            
    return pd.DataFrame({'x': x_list, 'shear': v_list, 'moment': m_list})

# ==========================================
# PART 3: RC DESIGN (USD)
# ==========================================
def calculate_rc_design(mu_kNm, vu_kN, b_cm, h_cm, cover_cm, fc, fy):
    # Convert Units
    b = b_cm / 100.0   # m
    h = h_cm / 100.0   # m
    d = h - (cover_cm / 100.0) # m
    mu = abs(mu_kNm) * 1000.0  # N-m
    vu = abs(vu_kN) * 1000.0   # N
    
    # Constants
    phi_b = 0.90
    phi_v = 0.85
    fc_pa = fc * 1e6 # Pascal
    fy_pa = fy * 1e6 # Pascal
    
    # --- Flexure Design ---
    mn_req = mu / phi_b
    Rn = mn_req / (b * d**2)
    m_ratio = fy / (0.85 * fc)
    
    rho = 0.0
    status = "OK"
    try:
        term = 1 - (2 * m_ratio * Rn) / fy_pa
        if term < 0:
            status = "Fails (Section too small)"
            rho = 0
        else:
            rho = (1 / m_ratio) * (1 - np.sqrt(term))
    except:
        status = "Error"

    as_calc = rho * b * d
    # Min Reinforcement
    as_min = max((np.sqrt(fc)/(4*fy))*b*d, (1.4/fy)*b*d)
    as_final = max(as_calc, as_min)
    
    # --- Shear Design ---
    vc = 0.17 * np.sqrt(fc) * b * d * 1e6 # Newton
    phi_vc = phi_v * vc
    
    shear_msg = ""
    if vu <= phi_vc / 2:
        shear_msg = "No Stirrups Needed"
    elif vu <= phi_vc:
        shear_msg = "Minimum Stirrups"
    else:
        vs_req = (vu - phi_vc) / phi_v
        shear_msg = f"Design Stirrups (Vs = {vs_req/1000:.2f} kN)"

    return {
        "status": status,
        "as_req_cm2": as_final * 10000,
        "phi_vc_kN": phi_vc / 1000,
        "vu_kN": vu / 1000,
        "shear_msg": shear_msg,
        "rho": rho
    }

# ==========================================
# PART 4: UI APPLICATION
# ==========================================
st.title("🏗️ Continuous Beam Analysis & Design")
st.caption("Interactive FEM | Includes SDL | EIT/ACI 318")

# TAB NAVIGATION
tab1, tab2, tab3 = st.tabs(["1️⃣ Input Data", "2️⃣ Analysis Diagrams", "3️⃣ RC Design Report"])

# --- TAB 1: INPUTS ---
with tab1:
    col_n, _ = st.columns([1, 3])
    n_span = col_n.number_input("Number of Spans", 1, 10, 2)
    
    spans = []
    supports = []
    loads = []
    
    st.markdown("---")
    st.subheader("1. Geometry (m) & Supports")
    
    # Spans
    cols = st.columns(n_span)
    for i in range(n_span):
        spans.append(cols[i].number_input(f"Span {i+1} Length (m)", 1.0, 20.0, 4.0, key=f"s{i}"))
        
    # Supports
    cols_sup = st.columns(n_span + 1)
    opts = ['Pin', 'Roller', 'Fix']
    for i in range(n_span + 1):
        def_idx = 0 if i == 0 else 1
        supports.append(cols_sup[i].selectbox(f"Supp {i+1}", opts, index=def_idx, key=f"sup{i}"))
        
    st.markdown("---")
    st.subheader("2. Loads (Factored: 1.4(DL+SDL) + 1.7LL)")
    
    for i in range(n_span):
        with st.expander(f"📍 Loads on Span {i+1}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            ltype = c1.selectbox("Type", ["Uniform Load", "Point Load"], key=f"lt{i}")
            
            # Added SDL Here
            dl = c2.number_input(f"DL (kN/m)", value=10.0, key=f"dl{i}", help="Self-weight + Dead Load")
            sdl = c3.number_input(f"SDL (kN/m)", value=2.0, key=f"sdl{i}", help="Superimposed Dead Load (Floor finish, Wall, etc.)")
            ll = c4.number_input(f"LL (kN/m)", value=5.0, key=f"ll{i}", help="Live Load")
            
            pos = 0.0
            if ltype == "Point Load":
                pos = st.slider(f"Position from left of span {i+1} (m)", 0.0, spans[i], spans[i]/2.0, key=f"pos{i}")
                
            loads.append({"span_idx": i, "type": ltype, "dl": dl, "sdl": sdl, "ll": ll, "pos": pos})
            
    if st.button("🚀 Run Analysis", type="primary"):
        st.session_state['analyzed'] = True
        try:
            with st.spinner("Calculating..."):
                ss = analyze_structure(spans, supports, loads)
                st.session_state['ss'] = ss
                st.session_state['df_res'] = extract_plot_data(ss, len(spans))
        except Exception as e:
            st.error(f"Analysis Error: {e}")

# --- TAB 2: DIAGRAMS ---
with tab2:
    if st.session_state.get('analyzed') and 'df_res' in st.session_state:
        df = st.session_state['df_res']
        
        # Calculate Max Values
        m_max_pos = df['moment'].max()
        m_max_neg = df['moment'].min()
        v_max = df['shear'].abs().max()
        
        # Summary Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Moment (+)", f"{m_max_pos:.2f} kN-m")
        c2.metric("Max Moment (-)", f"{m_max_neg:.2f} kN-m")
        c3.metric("Max Shear (|V|)", f"{v_max:.2f} kN")
        
        # Plot SFD
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D32F2F'), name='Shear'))
        fig_v.update_layout(title="Shear Force Diagram (SFD)", xaxis_title="Distance (m)", yaxis_title="Shear Force (kN)")
        st.plotly_chart(fig_v, use_container_width=True)
        
        # Plot BMD
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1976D2'), name='Moment'))
        fig_m.update_layout(title="Bending Moment Diagram (BMD)", xaxis_title="Distance (m)", yaxis_title="Bending Moment (kN-m)")
        st.plotly_chart(fig_m, use_container_width=True)
        
        # Store for Design
        st.session_state['des_mu'] = max(abs(m_max_pos), abs(m_max_neg))
        st.session_state['des_vu'] = v_max
        
    else:
        st.info("👈 Please enter data and click 'Run Analysis' in Tab 1")

# --- TAB 3: DESIGN REPORT ---
with tab3:
    if 'des_mu' in st.session_state:
        st.header("📝 RC Beam Design Report")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Material")
            fc = st.number_input("Concrete fc' (MPa)", 15.0, 50.0, 24.0)
            fy = st.number_input("Steel fy (MPa)", 240.0, 600.0, 400.0)
        with c2:
            st.subheader("Section Size")
            b = st.number_input("Width b (cm)", 10.0, 100.0, 25.0)
            h = st.number_input("Depth h (cm)", 20.0, 200.0, 50.0)
            cover = st.number_input("Cover (cm)", 2.0, 10.0, 4.0)

        # Calculate Design
        res = calculate_rc_design(
            st.session_state['des_mu'], 
            st.session_state['des_vu'], 
            b, h, cover, fc, fy
        )
        
        st.divider()
        st.subheader("✅ Design Results")
        
        # Flexure Results
        col_flex, col_shear = st.columns(2)
        
        with col_flex:
            st.markdown(f"**1. Flexural Design (Moment)**")
            st.write(f"Design Moment $M_u$: **{st.session_state['des_mu']:.2f} kN-m**")
            
            if "Fails" in res['status']:
                st.error(f"Status: {res['status']}")
            else:
                st.success(f"Status: {res['status']}")
                st.metric("Required Steel ($A_{s,req}$)", f"{res['as_req_cm2']:.2f} cm²")
                st.caption(f"Reinforcement Ratio ($\\rho$): {res['rho']:.4f}")
        
        with col_shear:
            st.markdown(f"**2. Shear Design**")
            st.write(f"Design Shear $V_u$: **{res['vu_kN']:.2f} kN**")
            st.metric("Concrete Capacity ($\\phi V_c$)", f"{res['phi_vc_kN']:.2f} kN")
            
            if "No Stirrups" in res['shear_msg']:
                st.success(res['shear_msg'])
            elif "Minimum" in res['shear_msg']:
                st.warning(res['shear_msg'])
            else:
                st.error(res['shear_msg'])
                
    else:
        st.warning("No results available. Please run analysis first.")
