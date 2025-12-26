import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
# ตั้งค่า Backend ทันทีเพื่อป้องกัน Error ใน Streamlit Cloud
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from anastruct import SystemElements

# ==========================================
# PART 1: SETUP
# ==========================================
st.set_page_config(page_title="Continuous Beam Design", layout="wide")

FACTOR_DL = 1.4
FACTOR_LL = 1.7

# ==========================================
# PART 2: ANALYSIS ENGINE
# ==========================================
def analyze_and_extract(spans, supports, loads):
    # Mesh = 50 เพื่อกราฟละเอียด
    ss = SystemElements(EA=15000, EI=5000, mesh=50) 
    
    # 1. Build Elements
    start_x = 0
    for L in spans:
        ss.add_element(location=[[start_x, 0], [start_x + L, 0]])
        start_x += L
        
    # 2. Add Supports & Stability Check
    # กฏเหล็ก: ต้องมี Pin หรือ Fix อย่างน้อย 1 จุดเพื่อกันการเลื่อนแกน X (Sliding)
    # ถ้า User เลือก Roller หมดเลย เราจะแก้จุดแรกเป็น Pin ให้
    has_x_restraint = any(s in ['Pin', 'Fix'] for s in supports)
    
    if not has_x_restraint:
        supports[0] = 'Pin'
        st.toast("⚠️ Warning: Auto-changed Support 1 to 'Pin' to prevent instability.", icon="🔧")
        
    for i, s_type in enumerate(supports):
        node_id = i + 1
        if s_type == 'Fix': 
            ss.add_support_fixed(node_id=node_id)
        elif s_type == 'Pin': 
            ss.add_support_hinged(node_id=node_id)
        elif s_type == 'Roller': 
            # direction=2 คือรับแรงแนวดิ่ง (Vertical Support)
            ss.add_support_roll(node_id=node_id, direction=2) 

    # 3. Add Loads
    for load in loads:
        # รวม Load: 1.4(DL+SDL) + 1.7LL
        w_total = (FACTOR_DL * (load['dl'] + load['sdl'])) + (FACTOR_LL * load['ll'])
        elem_id = load['span_idx'] + 1
        
        if load['type'] == 'Uniform Load':
            ss.q_load(q=w_total, element_id=elem_id)
        elif load['type'] == 'Point Load':
            ss.point_load(node_id=None, element_id=elem_id, position=load['pos'], Fy=-w_total)

    # 4. SOLVE (Crucial Fix: Force Linear)
    # force_linear=True ช่วยลดโอกาสเกิด NaN จากการคำนวณซ้ำแบบ Non-linear
    ss.solve(force_linear=True)
    
    # 5. Extract Data
    # ต้องเรียก plot เงียบๆ เพื่อให้ library คำนวณค่าลง array
    try:
        fig = plt.figure()
        ss.show_shear_force(show=False)
        ss.show_bending_moment(show=False)
        plt.close(fig)
        plt.close('all')
    except:
        pass

    x_list, v_list, m_list = [], [], []
    
    # ดึงข้อมูลจาก Element ทีละตัว
    for i in range(1, len(spans) + 1):
        if i in ss.element_map:
            el = ss.element_map[i]
            
            # หาพิกัด X
            if hasattr(el.vertex_1, 'loc'):
                x0 = el.vertex_1.loc[0]
                x1 = el.vertex_2.loc[0]
            else:
                x0 = el.vertex_1.coordinates[0]
                x1 = el.vertex_2.coordinates[0]

            # ดึงค่าแรง
            v = np.array(getattr(el, 'shear', [])).flatten()
            m = np.array(getattr(el, 'moment', [])).flatten()
            
            # ถ้า array ว่าง หรือ มีค่า NaN ให้ข้าม
            if len(v) > 0:
                x = np.linspace(x0, x1, len(v))
                x_list.extend(x)
                v_list.extend(v)
                m_list.extend(m)
            
    df = pd.DataFrame({'x': x_list, 'shear': v_list, 'moment': m_list})
    
    # 6. Final Check for NaN
    if df.isnull().values.any():
        raise ValueError("Structure is unstable (Singular Matrix). Results contain NaN.")
        
    return df

# ==========================================
# PART 3: DESIGN CALCULATION
# ==========================================
def design_rc_beam(mu, vu, b, h, cover, fc, fy):
    # Units: Input cm, MPa -> Calc using m, Pascal
    b_m = b / 100.0
    d_m = (h - cover) / 100.0
    mu_Nm = abs(mu) * 1000.0
    vu_N = abs(vu) * 1000.0
    
    phi_b = 0.90
    phi_v = 0.85
    
    # Flexure
    Mn_req = mu_Nm / phi_b
    Rn = Mn_req / (b_m * d_m**2) # Pascal
    m = fy / (0.85 * fc)
    
    rho = 0.0
    status = "OK"
    try:
        term = 1 - (2 * m * Rn) / (fy * 1e6)
        if term < 0:
            status = "Section too small"
            rho = 0
        else:
            rho = (1/m) * (1 - np.sqrt(term))
    except:
        status = "Calc Error"
        
    As_req = rho * b_m * d_m * 10000 # cm2
    As_min = max((np.sqrt(fc)/(4*fy))*b_m*d_m, (1.4/fy)*b_m*d_m) * 10000
    As_final = max(As_req, As_min)
    
    # Shear
    Vc = 0.17 * np.sqrt(fc) * b_m * d_m * 1e6 # N
    Phi_Vc = phi_v * Vc
    
    shear_msg = ""
    if vu_N <= Phi_Vc/2: shear_msg = "No Stirrup Req."
    elif vu_N <= Phi_Vc: shear_msg = "Min Stirrup Req."
    else: shear_msg = f"Design Stirrup (Vs = {(vu_N - Phi_Vc)/1000:.2f} kN)"
    
    return {
        "status": status,
        "as_req": As_final,
        "phi_vc": Phi_Vc / 1000,
        "shear_msg": shear_msg
    }

# ==========================================
# PART 4: UI
# ==========================================
st.title("🏗️ Continuous Beam Design (Stable V.6)")

tab1, tab2, tab3 = st.tabs(["Inputs", "Analysis", "Design"])

with tab1:
    col_n, _ = st.columns([1,3])
    n_span = col_n.number_input("Spans", 1, 10, 2)
    
    spans, supports, loads = [], [], []
    
    st.subheader("Geometry")
    cols = st.columns(n_span)
    for i in range(n_span):
        spans.append(cols[i].number_input(f"L{i+1} (m)", 1.0, 20.0, 4.0))
        
    st.subheader("Supports")
    cols = st.columns(n_span+1)
    opts = ['Pin', 'Roller', 'Fix']
    for i in range(n_span+1):
        def_idx = 0 if i==0 else 1
        supports.append(cols[i].selectbox(f"S{i+1}", opts, index=def_idx))
        
    st.subheader("Loads")
    for i in range(n_span):
        with st.expander(f"Span {i+1} Load", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            ltype = c1.selectbox("Type", ["Uniform Load", "Point Load"], key=f"t{i}")
            dl = c2.number_input(f"DL (kN/m)", 10.0, key=f"d{i}")
            sdl = c3.number_input(f"SDL (kN/m)", 2.0, key=f"sd{i}")
            ll = c4.number_input(f"LL (kN/m)", 5.0, key=f"l{i}")
            pos = 0.0
            if ltype == "Point Load":
                pos = st.slider(f"Pos (m)", 0.0, spans[i], spans[i]/2, key=f"p{i}")
            loads.append({"span_idx": i, "type": ltype, "dl": dl, "sdl": sdl, "ll": ll, "pos": pos})
            
    btn = st.button("🚀 Analyze", type="primary")

if btn:
    st.session_state['run'] = True
    try:
        with st.spinner("Calculating..."):
            df = analyze_and_extract(spans, supports, loads)
            st.session_state['df'] = df
            st.session_state['error'] = None
    except Exception as e:
        st.session_state['df'] = None
        st.session_state['error'] = str(e)

with tab2:
    if st.session_state.get('error'):
        st.error(f"❌ Analysis Failed: {st.session_state['error']}")
        st.warning("💡 Hint: Try changing supports. Ensure at least one 'Pin' or 'Fix'.")
        
    elif st.session_state.get('df') is not None:
        df = st.session_state['df']
        m_max = df['moment'].max()
        m_min = df['moment'].min()
        v_max = df['shear'].abs().max()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Max M (+)", f"{m_max:.2f} kN-m")
        c2.metric("Max M (-)", f"{m_min:.2f} kN-m")
        c3.metric("Max V", f"{v_max:.2f} kN")
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='red'), name='Shear'))
        fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='blue'), name='Moment'))
        fig.update_layout(title="Internal Forces", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.session_state['res_m'] = max(abs(m_max), abs(m_min))
        st.session_state['res_v'] = v_max

with tab3:
    if 'res_m' in st.session_state:
        st.header("Design Results")
        c1, c2 = st.columns(2)
        with c1:
            fc = st.number_input("fc' (MPa)", value=24.0)
            fy = st.number_input("fy (MPa)", value=400.0)
        with c2:
            b = st.number_input("b (cm)", value=25.0)
            h = st.number_input("h (cm)", value=50.0)
            cover = st.number_input("cov (cm)", value=4.0)
            
        des = design_rc_beam(st.session_state['res_m'], st.session_state['res_v'], b, h, cover, fc, fy)
        
        st.info(f"Design Moment: {st.session_state['res_m']:.2f} kN-m | Shear: {st.session_state['res_v']:.2f} kN")
        st.write(f"**Status:** {des['status']}")
        st.write(f"**As Required:** {des['as_req']:.2f} cm²")
        st.write(f"**Shear:** {des['shear_msg']}")
