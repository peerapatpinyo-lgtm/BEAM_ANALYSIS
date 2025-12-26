import streamlit as st

def render_sidebar():
    st.sidebar.header("⚙️ Settings")
    code = st.sidebar.selectbox("Design Standard", ["EIT 1007 (WSD)", "EIT 1008 (SDM)", "ACI 318 (SDM)"], index=1)
    
    # Import function helper เล็กน้อยเพื่อ Default ค่า
    def_dl = 1.0 if "WSD" in code else (1.2 if "ACI" in code else 1.4)
    def_ll = 1.0 if "WSD" in code else (1.6 if "ACI" in code else 1.7)
    method = "WSD" if "WSD" in code else "SDM"

    st.sidebar.divider()
    c1, c2 = st.sidebar.columns(2)
    fdl = c1.number_input("DL Factor", value=def_dl, step=0.1)
    fll = c2.number_input("LL Factor", value=def_ll, step=0.1)
    
    st.sidebar.divider()
    unit = st.sidebar.radio("System", ["MKS (kg, m, ksc)", "SI (kN, m, MPa)"], index=0)
    
    return code, method, fdl, fll, unit

def render_geometry_input():
    st.info("📍 Geometry")
    n = st.number_input("Number of Spans", 1, 5, 2)
    spans, supports = [], []
    c1, c2 = st.columns(2)
    sl = c1.selectbox("Left Sup", ["Pin", "Roller", "Fix"], key="SL")
    sr = c2.selectbox("Right Sup", ["Pin", "Roller", "Fix"], index=1, key="SR")
    
    supports.append(sl)
    for i in range(n):
        l = st.number_input(f"Span {i+1} (m)", 0.5, 20.0, 4.0, key=f"L{i}")
        spans.append(l)
        if i < n-1: supports.append("Roller")
    supports.append(sr)
    return n, spans, supports

def render_loads_input(n_span, spans, fdl, fll, unit_sys):
    st.info(f"⬇️ Loads ({unit_sys.split(' ')[0]})")
    loads = []
    tabs = st.tabs([f"Span {i+1}" for i in range(n_span)])
    
    # Unit Helper
    to_N = 1000.0 if "kN" in unit_sys else 9.80665
    
    for i, tab in enumerate(tabs):
        with tab:
            c1, c2 = st.columns(2)
            udl = c1.number_input(f"Uniform DL", 0.0, key=f"udl{i}")
            ull = c2.number_input(f"Uniform LL", 0.0, key=f"ull{i}")
            w = udl*fdl + ull*fll
            if w > 0: loads.append({'span_idx': i, 'type': 'Uniform', 'total_w': w*to_N, 'display_val': udl+ull})
            
            st.markdown("---")
            if st.checkbox("Add Point Load", key=f"chk{i}"):
                c3, c4, c5 = st.columns(3)
                pdl = c3.number_input("P DL", 0.0, key=f"pdl{i}")
                pll = c4.number_input("P LL", 0.0, key=f"pll{i}")
                px = c5.number_input("Dist x", 0.0, spans[i], spans[i]/2, key=f"px{i}")
                p = pdl*fdl + pll*fll
                if p > 0: loads.append({'span_idx': i, 'type': 'Point', 'total_w': p*to_N, 'pos': px, 'display_val': pdl+pll})
    return loads

def render_design_input(unit_sys):
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1.5])
    
    unit_s = "MPa" if "kN" in unit_sys else "ksc"
    
    with c1:
        st.markdown("##### 🧱 Materials")
        fc = st.number_input(f"f'c ({unit_s})", value=240.0 if "ksc" in unit_s else 24.0)
        fy = st.number_input(f"fy ({unit_s})", value=4000.0 if "ksc" in unit_s else 400.0)
    
    with c2:
        st.markdown("##### 📐 Section")
        b = st.number_input("b (cm)", value=25.0)
        h = st.number_input("h (cm)", value=50.0)
        cov = st.number_input("Cover (cm)", value=3.0)
        
    with c3:
        st.markdown("##### ⛓️ Rebar")
        m_bar = st.selectbox("Main", ["DB12", "DB16", "DB20", "DB25", "DB28"], index=1)
        s_bar = st.selectbox("Stirrup", ["RB6", "RB9", "DB10"], index=1)
        
    st.markdown('</div>', unsafe_allow_html=True)
    return fc, fy, b, h, cov, m_bar, s_bar
