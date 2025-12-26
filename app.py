import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from anastruct import SystemElements

# ==========================================
# PART 1: SETUP & CONFIG
# ==========================================
st.set_page_config(page_title="Continuous Beam Pro (Final)", layout="wide")

# Constants
FACTOR_DL = 1.4
FACTOR_LL = 1.7

# ==========================================
# PART 2: ANALYSIS ENGINE (High Res)
# ==========================================
def analyze_structure(spans, supports, loads):
    # mesh=50 คือแบ่งชิ้นส่วนย่อย 50 จุดต่อคาน เพื่อให้กราฟโค้งเนียนและละเอียด
    ss = SystemElements(EA=15000, EI=5000, mesh=50) 
    
    # 1. Elements
    start_x = 0
    for L in spans:
        ss.add_element(location=[[start_x, 0], [start_x + L, 0]])
        start_x += L
        
    # 2. Stability Check & Supports
    # บังคับให้ Node แรกเป็น Pin ถ้าไม่มีจุดยึดแกน X เพื่อกัน Error
    if not any(s in ['Pin', 'Fix'] for s in supports):
        supports[0] = 'Pin'
        st.toast("⚠️ Auto-corrected first support to 'Pin' for stability.")
        
    for i, s_type in enumerate(supports):
        node_id = i + 1
        if s_type == 'Fix': ss.add_support_fixed(node_id=node_id)
        elif s_type == 'Pin': ss.add_support_hinged(node_id=node_id)
        elif s_type == 'Roller': ss.add_support_roll(node_id=node_id, direction=2) # 2=Vertical

    # 3. Loads
    for load in loads:
        w_total = (FACTOR_DL * (load['dl'] + load['sdl'])) + (FACTOR_LL * load['ll'])
        elem_id = load['span_idx'] + 1
        
        if load['type'] == 'Uniform Load':
            ss.q_load(q=w_total, element_id=elem_id)
        elif load['type'] == 'Point Load':
            ss.point_load(node_id=None, element_id=elem_id, position=load['pos'], Fy=-w_total)

    # 4. Solve
    ss.solve()
    return ss

def extract_plot_data(ss):
    """ดึงข้อมูลแบบละเอียดทุกจุด Mesh"""
    # Trigger calculation internally
    try:
        ss.show_shear_force(show=False)
        ss.show_bending_moment(show=False)
        plt.close('all')
    except: pass

    x_list, v_list, m_list = [], [], []
    
    # Sort elements by X position
    def get_x(el):
        return el.vertex_1.coordinates[0] if hasattr(el.vertex_1, 'coordinates') else el.vertex_1.loc[0]
        
    sorted_els = sorted(ss.element_map.values(), key=get_x)
    
    for el in sorted_els:
        x0 = get_x(el)
        x1 = get_x(el.vertex_2) if hasattr(el.vertex_2, 'coordinates') else el.vertex_2.loc[0]
        
        # ดึง array แรงที่ละเอียดตามจำนวน Mesh
        v = np.array(getattr(el, 'shear', [])).flatten()
        m = np.array(getattr(el, 'moment', [])).flatten()
        
        if len(v) > 0:
            # สร้าง array ระยะ x ให้ตรงกับจำนวนจุดข้อมูล
            x = np.linspace(x0, x1, len(v))
            x_list.extend(x)
            v_list.extend(v)
            m_list.extend(m)
            
    return pd.DataFrame({'x': x_list, 'shear': v_list, 'moment': m_list})

# ==========================================
# PART 3: RC DESIGN CALCULATION
# ==========================================
def calculate_rc_design(mu_kNm, vu_kN, b_cm, h_cm, cover_cm, fc, fy):
    # Convert units
    b = b_cm / 100.0
    h = h_cm / 100.0
    d = h - (cover_cm / 100.0)
    mu = abs(mu_kNm) * 1000.0 # kN-m -> N-m
    vu = abs(vu_kN) * 1000.0  # kN -> N
    
    # Constants
    phi_b = 0.90
    phi_v = 0.85
    fc_pascal = fc * 1e6
    fy_pascal = fy * 1e6
    
    # 1. Flexure Design (Mu)
    # Mn required
    mn = mu / phi_b
    
    # R_n = Mn / (b * d^2)
    Rn = mn / (b * d**2)
    
    # Rho calculation
    m_ratio = fy / (0.85 * fc)
    rho = 0.0
    status = "OK"
    
    try:
        # Check against Rho max (approx 0.75 rho_b) - simplified check
        # Exact Formula: rho = (1/m) * (1 - sqrt(1 - 2*m*Rn/fy))
        term = 1 - (2 * m_ratio * Rn) / fy_pascal
        if term < 0:
            status = "Section too small (Compression Fail)"
            rho = 0
        else:
            rho = (1 / m_ratio) * (1 - np.sqrt(term))
    except:
        status = "Calculation Error"

    # As Required
    as_calc = rho * b * d
    
    # As Min (ACI 318 / EIT)
    as_min1 = (np.sqrt(fc) / (4 * fy)) * b * d
    as_min2 = (1.4 / fy) * b * d
    as_min = max(as_min1, as_min2)
    
    as_final = max(as_calc, as_min)
    
    # 2. Shear Design (Vu)
    # Vc = 0.17 * sqrt(fc) * b * d
    vc = 0.17 * np.sqrt(fc) * b * d * 1e6 # N
    phi_vc = phi_v * vc
    
    shear_status = ""
    vs_req = 0.0
    
    if vu <= phi_vc / 2:
        shear_status = "No Stirrups Req."
    elif vu <= phi_vc:
        shear_status = "Min Stirrups Req."
    else:
        # Need calculation Vs = (Vu - phi*Vc) / phi
        vs_force = (vu - phi_vc) / phi_v
        shear_status = f"Design Stirrups (Vs = {vs_force/1000:.2f} kN)"
        vs_req = vs_force

    return {
        "status_flexure": status,
        "as_req": as_final * 10000, # cm2
        "rho_actual": rho,
        "phi_vc": phi_vc / 1000, # kN
        "vu": vu / 1000, # kN
        "shear_status": shear_status
    }

# ==========================================
# PART 4: UI & MAIN APP
# ==========================================
st.title("🏗️ Continuous Beam Analysis & Design (Pro)")

# --- TAB 1: INPUT ---
tab1, tab2, tab3 = st.tabs(["1. Model Inputs", "2. Analysis Diagrams", "3. Design Report"])

with tab1:
    col_main1, col_main2 = st.columns([1, 2])
    with col_main1:
        n_span = st.number_input("Number of Spans", 1, 8, 2)
    
    # Dynamic Inputs
    spans = []
    supports = []
    loads = []
    
    st.markdown("### 📏 Geometry & Supports")
    cols = st.columns(n_span + 1)
    for i in range(n_span):
        spans.append(cols[i].number_input(f"L{i+1} (m)", 1.0, 20.0, 4.0, key=f"s{i}"))
        
    cols_supp = st.columns(n_span + 1)
    opts = ['Pin', 'Roller', 'Fix']
    for i in range(n_span + 1):
        def_idx = 0 if i==0 else 1
        supports.append(cols_supp[i].selectbox(f"S{i+1}", opts, index=def_idx, key=f"sup{i}"))
        
    st.markdown("### ⬇️ Loads (1.4DL + 1.7LL)")
    for i in range(n_span):
        with st.expander(f"Loads on Span {i+1}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            ltype = c1.selectbox("Type", ["Uniform Load", "Point Load"], key=f"lt{i}")
            dl = c2.number_input(f"DL (kN/m or kN)", value=10.0, key=f"dl{i}")
            ll = c3.number_input(f"LL (kN/m or kN)", value=8.0, key=f"ll{i}")
            
            pos = 0.0
            if ltype == "Point Load":
                pos = c4.slider("Position (m)", 0.0, spans[i], spans[i]/2.0, key=f"pos{i}")
                
            loads.append({"span_idx": i, "type": ltype, "dl": dl, "sdl": 0, "ll": ll, "pos": pos})
            
    if st.button("🚀 Calculate", type="primary"):
        st.session_state['run'] = True
        with st.spinner("Analyzing finite elements..."):
            try:
                ss = analyze_structure(spans, supports, loads)
                st.session_state['ss'] = ss
                st.session_state['df'] = extract_plot_data(ss)
            except Exception as e:
                st.error(f"Analysis Error: {e}")

# --- TAB 2: DIAGRAMS ---
with tab2:
    if 'run' in st.session_state and 'df' in st.session_state:
        df = st.session_state['df']
        
        # Max Values
        max_m_pos = df['moment'].max()
        max_m_neg = df['moment'].min()
        max_v = df['shear'].abs().max()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Positive Moment", f"{max_m_pos:.2f} kN-m")
        c2.metric("Max Negative Moment", f"{max_m_neg:.2f} kN-m")
        c3.metric("Max Shear Force", f"{max_v:.2f} kN")
        
        # SFD
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#FF4B4B'), name='Shear'))
        fig_v.update_layout(title="Shear Force Diagram (SFD)", xaxis_title="Distance (m)", yaxis_title="Shear (kN)", hovermode="x")
        st.plotly_chart(fig_v, use_container_width=True)
        
        # BMD
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1E88E5'), name='Moment'))
        fig_m.update_layout(title="Bending Moment Diagram (BMD)", xaxis_title="Distance (m)", yaxis_title="Moment (kN-m)", hovermode="x")
        st.plotly_chart(fig_m, use_container_width=True)
        
        # Save Max for Design
        st.session_state['des_mu'] = max(abs(max_m_pos), abs(max_m_neg))
        st.session_state['des_vu'] = max_v
    else:
        st.info("👈 Please run analysis first.")

# --- TAB 3: DESIGN REPORT ---
with tab3:
    if 'des_mu' in st.session_state:
        st.header("📝 Reinforced Concrete Design (USD)")
        
        c_mat, c_sect = st.columns(2)
        with c_mat:
            st.subheader("Material Properties")
            fc = st.number_input("fc' (MPa)", value=24.0, step=1.0)
            fy = st.number_input("fy (MPa)", value=400.0, step=10.0)
            
        with c_sect:
            st.subheader("Section Dimensions")
            b = st.number_input("Width b (cm)", value=25.0)
            h = st.number_input("Depth h (cm)", value=50.0)
            cover = st.number_input("Cover (cm)", value=4.0)
            
        # Perform Design
        des = calculate_rc_design(
            st.session_state['des_mu'], 
            st.session_state['des_vu'], 
            b, h, cover, fc, fy
        )
        
        st.markdown("---")
        st.subheader("Design Results")
        
        # Flexure Result
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.info(f"**Flexural Status:** {des['status_flexure']}")
            st.write(f"Design Moment (Mu): **{st.session_state['des_mu']:.2f} kN-m**")
            st.metric("Required Steel (As)", f"{des['as_req']:.2f} cm²")
            st.caption(f"Rho actual: {des['rho_actual']:.4f}")
            
        # Shear Result
        with res_col2:
            st.info(f"**Shear Status:** {des['shear_status']}")
            st.write(f"Design Shear (Vu): **{des['vu']:.2f} kN**")
            st.metric("Concrete Capacity (Phi Vc)", f"{des['phi_vc']:.2f} kN")
            
    else:
        st.warning("No analysis data available. Go to Tab 1 & Run Analysis.")
