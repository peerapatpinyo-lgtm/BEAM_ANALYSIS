import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- IMPORT ENGINE ---
try:
    from beam_engine import SimpleBeamSolver
except ImportError:
    st.error("⚠️ ไม่พบไฟล์ 'beam_engine.py' กรุณาวางไฟล์ engine ไว้ในโฟลเดอร์เดียวกัน")
    st.stop()

# ==========================================
# ⚙️ GLOBAL CONFIG & STYLES
# ==========================================
st.set_page_config(page_title="Beam Design Pro (ACI 318-19/22)", layout="wide")

# Custom CSS for better spacing
st.markdown("""
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 5rem;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🎨 DRAWING FUNCTIONS
# ==========================================

def draw_beam_schema(spans, supports, loads):
    """Draw the longitudinal beam model"""
    fig = go.Figure()
    # Main Beam
    fig.add_trace(go.Scatter(x=[0, sum(spans)], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=6), name='Beam', hoverinfo='none'))
    
    cx, sx = 0, 0
    # Dimensions
    for L in spans:
        fig.add_annotation(x=cx+L/2, y=-0.5, text=f"<b>{L} m</b>", showarrow=False, font=dict(color="blue", size=14))
        fig.add_shape(type="line", x0=cx+L, y0=-0.3, x1=cx+L, y1=0.3, line=dict(color="gray", dash="dot"))
        cx += L
        
    # Supports
    for i, s in enumerate(supports):
        sym = "triangle-up" if s != "Fix" else "square"
        col = "green" if s != "Fix" else "red"
        fig.add_trace(go.Scatter(x=[sx], y=[-0.15], mode='markers', 
                                 marker=dict(symbol=sym, size=16, color=col), 
                                 showlegend=False, hovertemplate=f"Support {i+1}: {s}"))
        if i < len(spans): sx += spans[i]

    # Loads
    for ld in loads:
        start_x = sum(spans[:ld['span_idx']])
        val = ld['display_val']
        if ld['type'] == 'Point':
            fig.add_annotation(x=start_x+ld['pos'], y=0.1, ax=0, ay=-40, 
                               text=f"<b>{val:.1f}</b>", showarrow=True, arrowhead=2, arrowcolor="#D32F2F")
        elif ld['type'] == 'Uniform':
            end_x = start_x + spans[ld['span_idx']]
            fig.add_shape(type="rect", x0=start_x, y0=0, x1=end_x, y1=0.2, 
                          fillcolor="rgba(255,0,0,0.1)", line_width=0)
            fig.add_annotation(x=(start_x+end_x)/2, y=0.25, text=f"<b>w={val:.1f}</b>", showarrow=False, font=dict(color="#D32F2F"))

    fig.update_layout(height=250, xaxis=dict(showgrid=False, visible=False), 
                      yaxis=dict(visible=False, range=[-0.8, 1.0]), 
                      margin=dict(t=10, b=10, l=10, r=10), plot_bgcolor="white")
    return fig

def draw_section_detail(b, h, cover, n_bars, bar_name, stirrup_name):
    """Draw Cross Section of the Beam"""
    fig = go.Figure()
    
    # 1. Concrete Face
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, 
                  line=dict(color="black", width=3), fillcolor="#F5F5F5")
    
    # 2. Stirrup (Rectangular Loop)
    c = cover
    fig.add_shape(type="rect", x0=c, y0=c, x1=b-c, y1=h-c,
                  line=dict(color="#D32F2F", width=2), fillcolor="rgba(0,0,0,0)")
    
    # 3. Main Bars (Bottom)
    bar_dia_approx = 2.0 # visual size
    # Calculate positions
    if n_bars > 0:
        if n_bars == 1:
             x_pos = [b/2]
        else:
             # Distribute evenly between stirrups
             start_x = c + bar_dia_approx/2
             end_x = b - c - bar_dia_approx/2
             x_pos = np.linspace(start_x, end_x, n_bars)
             
        for bx in x_pos:
            by = c + bar_dia_approx/2
            fig.add_shape(type="circle", 
                          x0=bx-bar_dia_approx/2, y0=by-bar_dia_approx/2, 
                          x1=bx+bar_dia_approx/2, y1=by+bar_dia_approx/2,
                          fillcolor="#1565C0", line_color="black")

    # 4. Hanger Bars (Top - Dummy 2 bars)
    for i in [0, 1]:
        bx = c + bar_dia_approx/2 if i==0 else b - c - bar_dia_approx/2
        by = h - c - bar_dia_approx/2
        fig.add_shape(type="circle", 
                      x0=bx-bar_dia_approx/2, y0=by-bar_dia_approx/2, 
                      x1=bx+bar_dia_approx/2, y1=by+bar_dia_approx/2,
                      fillcolor="#90CAF9", line_color="black")

    # Annotations
    fig.add_annotation(x=b/2, y=h/2, text=f"<b>{b:.0f} x {h:.0f} cm</b>", showarrow=False, font=dict(size=16))
    fig.add_annotation(x=b/2, y=c, text=f"{n_bars}-{bar_name}", yshift=-25, showarrow=False, font=dict(color="#1565C0", size=14, weight="bold"))
    fig.add_annotation(x=b+2, y=h/2, text=f"Stirrup: {stirrup_name}", textangle=-90, showarrow=False, font=dict(color="#D32F2F"))

    fig.update_layout(
        width=300, height=350,
        xaxis=dict(visible=False, range=[-5, b+10]),
        yaxis=dict(visible=False, range=[-5, h+5], scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white"
    )
    return fig

# ==========================================
# 🖥️ MAIN UI
# ==========================================
st.title(f"🏗️ RC Beam Design Pro")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Design Standards")

# 1. Code Selection (Updated)
code_options = [
    "EIT 1007 (WSD - Thai)",
    "ACI 318-14 / EIT (SDM)",
    "ACI 318-19 (SDM)",
    "ACI 318-22 (SDM)"
]
design_code = st.sidebar.selectbox("Select Design Code", code_options, index=2)
is_sdm = "SDM" in design_code

# 2. Unit Selection
unit_opt = st.sidebar.radio("Units", ["SI (kN, m)", "MKS (kg, m)"])
if "kN" in unit_opt:
    UNIT_F, UNIT_L, UNIT_M = "kN", "kN/m", "kN-m"
    TO_N, FROM_N = 1000.0, 0.001
else:
    UNIT_F, UNIT_L, UNIT_M = "kg", "kg/m", "kg-m"
    TO_N, FROM_N = 9.80665, 1/9.80665

st.sidebar.markdown("---")
st.sidebar.info(f"Using: **{design_code}**\n\nMethod: **{'Strength Design' if is_sdm else 'Working Stress'}**")

# ==========================================
# 1. INPUT SECTION
# ==========================================
c_geo, c_load = st.columns([1, 1.2])

with c_geo:
    st.subheader("1. Geometry")
    n_span = st.number_input("Spans", 1, 5, 2)
    
    spans = []
    supports = []
    
    # Compact Geometry Input
    for i in range(n_span):
        c1, c2, c3 = st.columns([1, 1.2, 1])
        if i == 0:
            supports.append(c1.selectbox(f"Sup 1", ['Pin', 'Roller', 'Fix'], key='s0'))
        else:
            c1.write(f"Sup {i+1} (Mid)")
            
        spans.append(c2.number_input(f"L{i+1} (m)", 0.5, 20.0, 4.0, key=f'l{i}'))
        
        # Last support logic
        supports.append(c3.selectbox(f"Sup {i+2}", ['Pin', 'Roller', 'Fix'], index=1, key=f's{i+1}'))

with c_load:
    st.subheader("2. Loads & Factors")
    
    # Auto-set Factors based on Code
    if is_sdm:
        # ACI 318-19/22 uses 1.2D + 1.6L, Older uses 1.4D + 1.7L
        if "19" in design_code or "22" in design_code:
            def_dl, def_ll = 1.2, 1.6
        else:
            def_dl, def_ll = 1.4, 1.7
        
        # ✅ FIXED: Unpacking 2 variables from 2 columns (Previously caused error)
        cf1, cf2 = st.columns(2) 
        f_dl = cf1.number_input("Factor DL", 1.0, 2.0, def_dl, step=0.1)
        f_ll = cf2.number_input("Factor LL", 1.0, 2.0, def_ll, step=0.1)
    else:
        st.info("WSD: Factors fixed at 1.0")
        f_dl, f_ll = 1.0, 1.0

    # Load Inputs (Simplified)
    loads_input = []
    with st.expander("📝 Edit Loads (Click to Open)", expanded=True):
        for i in range(n_span):
            st.markdown(f"**Span {i+1}**")
            cl1, cl2 = st.columns(2)
            udl = cl1.number_input(f"DL (Uniform) {i+1}", 0.0, key=f"udl{i}")
            ull = cl2.number_input(f"LL (Uniform) {i+1}", 0.0, key=f"ull{i}")
            
            w_total = f_dl*udl + f_ll*ull
            if w_total > 0:
                loads_input.append({'span_idx': i, 'type': 'Uniform', 'total_w': w_total*TO_N, 'display_val': w_total})

# ==========================================
# 2. ANALYSIS
# ==========================================
if st.button("🚀 Analyze & Design", type="primary", use_container_width=True):
    solver = SimpleBeamSolver(spans, supports, loads_input)
    u, err = solver.solve()
    
    if err:
        st.error(err)
    else:
        df = solver.get_internal_forces(100)
        df['V'] = df['shear'] * FROM_N
        df['M'] = df['moment'] * FROM_N
        
        st.session_state['res'] = df
        st.session_state['input_data'] = (spans, supports, loads_input)
        st.session_state['run'] = True

# ==========================================
# 3. RESULTS & VISUALIZATION
# ==========================================
if st.session_state.get('run', False):
    df = st.session_state['res']
    spans, supports, loads_input = st.session_state['input_data']
    
    st.markdown("---")
    # Draw Physical Beam
    st.plotly_chart(draw_beam_schema(spans, supports, loads_input), use_container_width=True)
    
    # Graphs
    c_graph1, c_graph2 = st.columns(2)
    
    # SFD with Labels
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=df['x'], y=df['V'], fill='tozeroy', line=dict(color='#D32F2F', width=2), name="Shear"))
    # Add labels at max/min
    v_max, v_min = df['V'].max(), df['V'].min()
    fig_v.add_annotation(x=df.loc[df['V'].idxmax(), 'x'], y=v_max, text=f"<b>{v_max:.2f}</b>", showarrow=False, yshift=15, font=dict(color="#D32F2F"))
    fig_v.add_annotation(x=df.loc[df['V'].idxmin(), 'x'], y=v_min, text=f"<b>{v_min:.2f}</b>", showarrow=False, yshift=-15, font=dict(color="#D32F2F"))
    fig_v.update_layout(title=f"Shear Force (SFD) [{UNIT_F}]", hovermode="x unified")
    c_graph1.plotly_chart(fig_v, use_container_width=True)
    
    # BMD with Labels
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(x=df['x'], y=df['M'], fill='tozeroy', line=dict(color='#1976D2', width=2), name="Moment"))
    m_max, m_min = df['M'].max(), df['M'].min()
    # Find local peaks/valleys could be done better, but max/min is sufficient for design
    fig_m.add_annotation(x=df.loc[df['M'].idxmax(), 'x'], y=m_max, text=f"<b>{m_max:.2f}</b>", showarrow=False, yshift=15, font=dict(color="#1976D2"))
    fig_m.add_annotation(x=df.loc[df['M'].idxmin(), 'x'], y=m_min, text=f"<b>{m_min:.2f}</b>", showarrow=False, yshift=-15, font=dict(color="#1976D2"))
    fig_m.update_layout(title=f"Bending Moment (BMD) [{UNIT_M}]", yaxis=dict(autorange="reversed"), hovermode="x unified")
    c_graph2.plotly_chart(fig_m, use_container_width=True)

    # ==========================================
    # 4. DESIGN SECTION (INTERACTIVE)
    # ==========================================
    st.markdown("---")
    st.header(f"🛠️ Section Design ({design_code})")
    
    # Layout: Control Panel | Section Drawing | Calculation Text
    col_input, col_draw, col_calc = st.columns([1, 1, 1.2])
    
    with col_input:
        st.markdown("##### 🧱 Material & Section")
        fc = st.number_input("Concrete f'c (ksc)", value=240.0)
        fy = st.number_input("Main Steel fy (ksc)", value=4000.0)
        fys = st.number_input("Stirrup fys (ksc)", value=2400.0)
        
        b = st.number_input("Width b (cm)", value=25.0)
        h = st.number_input("Height h (cm)", value=50.0)
        cover = st.number_input("Cover (cm)", value=3.0)
        
        st.markdown("##### ⛓️ Reinforcement")
        bar_opts = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
        main_bar = st.selectbox("Main Bar Size", list(bar_opts.keys()), index=1)
        
        # New: Stirrup Selector
        stirrup_type = st.radio("Stirrup Type", ["RB9 (Round)", "DB12 (Deformed)"], horizontal=True)
        if "RB9" in stirrup_type:
            Av_stirrup = 2 * 0.636 # 2 legs of 9mm
            stirrup_name = "RB9"
        else:
            Av_stirrup = 2 * 1.131 # 2 legs of 12mm
            stirrup_name = "DB12"

    # --- CALCULATION LOGIC ---
    # Convert Units to N, mm, MPa
    fc_mpa, fy_mpa, fys_mpa = fc*0.0981, fy*0.0981, fys*0.0981
    b_mm, d_mm = b*10, (h-cover)*10
    
    # Design Values
    Mu = max(abs(m_max), abs(m_min)) * (1e6 if "kN" in unit_opt else 9.81*1000) # N-mm
    Vu = abs(df['V']).max() * (1000 if "kN" in unit_opt else 9.81) # N
    
    # Design Result Placeholders
    design_status = "OK"
    req_As = 0
    n_bars = 0
    spacing_req = 0

    # ------------------
    # DESIGN PROCESS
    # ------------------
    if is_sdm:
        # Strength Design (ACI/SDM)
        phi_b = 0.9
        Rn = Mu / (phi_b * b_mm * d_mm**2)
        m = fy_mpa / (0.85 * fc_mpa)
        
        # Check rho
        term = 1 - (2 * m * Rn) / fy_mpa
        if term < 0:
            design_status = "FAIL"
            rho_des = 0
        else:
            rho_req = (1/m)*(1 - np.sqrt(term))
            rho_min = max(np.sqrt(fc_mpa)/(4*fy_mpa), 1.4/fy_mpa)
            rho_des = max(rho_req, rho_min)
            
        req_As = rho_des * b_mm * d_mm # mm2
        req_As_cm2 = req_As / 100
        
        # Shear (ACI Simplified 318-19 valid for Av > Avmin)
        lambda_c = 1.0 # normal weight
        Vc = 0.17 * lambda_c * np.sqrt(fc_mpa) * b_mm * d_mm
        phi_v = 0.75 # ACI standard for shear
        phiVc = phi_v * Vc
        
        if Vu > phiVc:
            Vs = (Vu - phiVc) / phi_v
            s_req = (Av_stirrup*100 * fys_mpa * d_mm) / Vs # mm
            spacing_show = min(s_req, d_mm/2, 600)
            shear_txt = f"USE {stirrup_name} @ {int(spacing_show/10)} cm"
            shear_color = "red"
        elif Vu > 0.5 * phiVc:
             shear_txt = f"Min Stirrup: {stirrup_name} @ {int(d_mm/20)} cm"
             shear_color = "orange"
        else:
             shear_txt = "Concrete Shear OK (Min Stirrups)"
             shear_color = "green"

    else:
        # Working Stress (WSD)
        n = 135 / np.sqrt(fc) if fc > 0 else 10 # Approx n
        fc_all, fs_all = 0.45*fc, 0.5*fy
        k = 1 / (1 + fs_all/(n*fc_all))
        j = 1 - k/3
        
        # Check Moment Capacity
        Mc_kgm = (0.5 * fc_all * k * j * b * (h-cover)**2) / 100
        M_chk_kgm = Mu / (9.81 * 100) # Convert Mu(Nmm) back to kg-m approximately for check
        
        if M_chk_kgm > Mc_kgm:
             design_status = "FAIL"
             req_As_cm2 = 0
        else:
             req_As_cm2 = (M_chk_kgm * 100) / (fs_all * j * (h-cover)) # cm2
        
        # Shear
        vc_all = 0.29 * np.sqrt(fc) # ksc
        v_act = (Vu/9.81) / (b*d_mm/10) # ksc
        if v_act > vc_all:
             shear_txt = f"Design Stirrups ({stirrup_name})"
             shear_color = "red"
        else:
             shear_txt = "Concrete Shear OK"
             shear_color = "green"

    # Number of Bars
    if design_status != "FAIL":
        n_bars = max(2, int(np.ceil(req_As_cm2 / bar_opts[main_bar])))
        
    # --- VISUALIZATION (Middle Column) ---
    with col_draw:
        st.markdown("##### 📐 Section View")
        if design_status == "FAIL":
            st.error("❌ Section Too Small (Concrete Fail)")
        else:
            fig_sec = draw_section_detail(b, h, cover, n_bars, main_bar, stirrup_name)
            st.plotly_chart(fig_sec, use_container_width=True)

    # --- RESULTS (Right Column) ---
    with col_calc:
        st.markdown("##### 📝 Design Results")
        
        if design_status == "FAIL":
            st.error(f"Cannot Design: Moment too high for this section size.")
        else:
            st.success(f"**Flexure:** Use {n_bars} - {main_bar}")
            st.write(f"As Required: {req_As_cm2:.2f} cm²")
            st.write(f"As Provided: {n_bars * bar_opts[main_bar]:.2f} cm²")
            
            st.divider()
            
            st.markdown(f"**Shear:** :{shear_color}[{shear_txt}]")
            st.write(f"Vu Max: {Vu/1000 if 'kN' in unit_opt else Vu/9.81:.2f} {UNIT_F}")
            
            with st.expander("Show Calculation Details"):
                if is_sdm:
                    st.latex(f"M_u = {Mu/1e6:.2f} \\text{{ kNm}}")
                    st.latex(f"R_n = {Rn:.2f} \\text{{ MPa}}")
                    st.latex(f"\\rho_{{req}} = {rho_req:.4f}")
                    st.latex(f"A_s = {req_As_cm2:.2f} \\text{{ cm}}^2")
                    st.markdown("**Shear (Simplified ACI):**")
                    st.latex(f"\\phi V_c = {phiVc/1000:.2f} \\text{{ kN}}")
                else:
                    st.latex(f"M = {Mu/(9.81*100):.2f} \\text{{ kg-m}}")
                    st.latex(f"M_c = {Mc_kgm:.2f} \\text{{ kg-m}}")
                    st.latex(f"A_s = M / (f_s j d)")
