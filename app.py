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
# ⚙️ CONFIG & STYLES
# ==========================================
st.set_page_config(page_title="RC Beam Pro V.14", layout="wide")
st.markdown("""
<style>
    .report-box { border: 1px solid #ddd; padding: 20px; border-radius: 10px; background-color: #f9f9f9; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 5px; }
    .stTabs [aria-selected="true"] { background-color: #e3f2fd; border-bottom: 2px solid #1976d2; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🎨 DRAWING FUNCTIONS
# ==========================================
def draw_beam_schema(spans, supports, loads):
    fig = go.Figure()
    
    # 1. Main Beam
    total_len = sum(spans)
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=6), name='Beam', hoverinfo='none'))
    
    cx = 0
    # 2. Dimensions & Supports
    for i, L in enumerate(spans):
        # Dimension Line
        fig.add_annotation(x=cx+L/2, y=-0.6, text=f"<b>{L:.2f} m</b>", showarrow=False, font=dict(color="blue", size=14))
        fig.add_shape(type="line", x0=cx+L, y0=-0.3, x1=cx+L, y1=0.3, line=dict(color="gray", dash="dot"))
        
        # Support (Left of current span)
        sx = cx
        s_type = supports[i]
        sym = "triangle-up" if s_type != "Fix" else "square"
        col = "green" if s_type != "Fix" else "red"
        fig.add_trace(go.Scatter(x=[sx], y=[-0.15], mode='markers', 
                                 marker=dict(symbol=sym, size=16, color=col), showlegend=False, 
                                 name=f"Sup {i+1}"))
        cx += L
        
    # Last Support
    fig.add_trace(go.Scatter(x=[total_len], y=[-0.15], mode='markers', 
                             marker=dict(symbol="triangle-up" if supports[-1]!="Fix" else "square", size=16, color="green" if supports[-1]!="Fix" else "red"), 
                             showlegend=False, name=f"Sup {len(supports)}"))

    # 3. Loads Visualization
    max_load_val = 1.0
    for ld in loads:
        max_load_val = max(max_load_val, ld['display_val'])

    for ld in loads:
        start_x = sum(spans[:ld['span_idx']])
        val = ld['display_val']
        
        if ld['type'] == 'Point':
            # Point Load Arrow
            fig.add_annotation(
                x=start_x + ld['pos'], y=0.1, ax=0, ay=-50,
                text=f"<b>P={val:.1f}</b>", showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#D32F2F",
                font=dict(color="#D32F2F")
            )
        elif ld['type'] == 'Uniform':
            # Distributed Block
            end_x = start_x + spans[ld['span_idx']]
            fig.add_shape(type="rect", x0=start_x, y0=0, x1=end_x, y1=0.25, 
                          fillcolor="rgba(211, 47, 47, 0.2)", line_width=0)
            fig.add_annotation(x=(start_x+end_x)/2, y=0.3, text=f"<b>w={val:.1f}</b>", showarrow=False, font=dict(color="#D32F2F"))

    fig.update_layout(height=280, xaxis=dict(showgrid=False, visible=False, range=[-0.5, total_len+0.5]), 
                      yaxis=dict(visible=False, range=[-1, 1]), margin=dict(t=20, b=10, l=10, r=10))
    return fig

def draw_section_detail(b, h, cover, n_bars, bar_name, stirrup_name, spacing):
    fig = go.Figure()
    
    # Concrete Section
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=3), fillcolor="#F0F0F0")
    
    # Stirrup
    c = cover
    fig.add_shape(type="rect", x0=c, y0=c, x1=b-c, y1=h-c, line=dict(color="#D32F2F", width=2))
    
    # Main Bars (Bottom)
    dia = 2.5 # Visual size
    if n_bars > 0:
        gap = (b - 2*c - dia) / (n_bars - 1) if n_bars > 1 else 0
        for i in range(n_bars):
            bx = c + dia/2 + i*gap if n_bars > 1 else b/2
            by = c + dia/2
            fig.add_shape(type="circle", x0=bx-dia/2, y0=by-dia/2, x1=bx+dia/2, y1=by+dia/2, fillcolor="#1565C0", line_color="black")
            
    # Hanger Bars (Top)
    for i in [0, 1]:
        bx = c + dia/2 if i==0 else b - c - dia/2
        by = h - c - dia/2
        fig.add_shape(type="circle", x0=bx-dia/2, y0=by-dia/2, x1=bx+dia/2, y1=by+dia/2, fillcolor="#90CAF9", line_color="black")

    # Annotations
    fig.add_annotation(x=b/2, y=h/2, text=f"<b>{b}x{h} cm</b>", showarrow=False, font=dict(size=18, color="rgba(0,0,0,0.3)"))
    fig.add_annotation(x=b/2, y=c-5, text=f"<b>{n_bars}-{bar_name}</b>", showarrow=False, font=dict(color="#1565C0", size=14))
    fig.add_annotation(x=b+5, y=h/2, text=f"Stirrup:<br>{stirrup_name}<br>@{spacing:.0f} cm", showarrow=False, font=dict(color="#D32F2F", size=12))

    fig.update_layout(width=300, height=350, xaxis=dict(visible=False, range=[-10, b+15]), yaxis=dict(visible=False, range=[-10, h+5]), 
                      margin=dict(l=10,r=10,t=10,b=10), plot_bgcolor="white")
    return fig

# ==========================================
# 🖥️ MAIN UI
# ==========================================
st.title("🏗️ RC Beam Design Pro V.14")
st.caption("Engineered for Civil Engineers | Supports Thai EIT (WSD/SDM) & ACI 318")

# --- SIDEBAR: SETTINGS ---
st.sidebar.header("1. Design Standards")

# CODE SELECTION
code_map = {
    "EIT 1007 (WSD)":     {"type": "WSD", "DL":1.0, "LL":1.0, "phi_b":1.0, "phi_v":1.0},
    "EIT 1008 (SDM Thai)": {"type": "SDM", "DL":1.4, "LL":1.7, "phi_b":0.90, "phi_v":0.85},
    "ACI 318-11 (SDM)":    {"type": "SDM", "DL":1.2, "LL":1.6, "phi_b":0.90, "phi_v":0.75},
    "ACI 318-19/22 (SDM)": {"type": "SDM", "DL":1.2, "LL":1.6, "phi_b":0.90, "phi_v":0.75}
}
design_code = st.sidebar.selectbox("Design Code", list(code_map.keys()), index=1)
params = code_map[design_code]
is_sdm = params["type"] == "SDM"

# UNIT SELECTION
unit_sys = st.sidebar.radio("Unit System", ["SI (kN, m, MPa)", "MKS (kg, m, ksc)"])
if "kN" in unit_sys:
    U_F, U_L, U_M, U_S = "kN", "kN/m", "kN-m", "MPa"
    TO_N, FROM_N = 1000.0, 0.001
    G_ACC = 1.0 # In SI input is usually mass or force? Assuming Force input directly
else:
    U_F, U_L, U_M, U_S = "kg", "kg/m", "kg-m", "ksc"
    TO_N, FROM_N = 9.80665, 1/9.80665
    G_ACC = 1.0

st.sidebar.markdown(f"""
---
**Design Parameters:**
* Method: `{params['type']}`
* Load Factors: `{params['DL']}D + {params['LL']}L`
* Phi Flexure: `{params['phi_b']}`
* Phi Shear: `{params['phi_v']}`
""")

# ==========================================
# 2. INPUTS
# ==========================================
c1, c2 = st.columns([1, 1.3])

with c1:
    st.subheader("🔹 Geometry")
    n_span = st.number_input("Number of Spans", 1, 5, 2)
    spans, supports = [], []
    
    for i in range(n_span):
        col_sup, col_len = st.columns([1, 2])
        if i == 0:
            supports.append(col_sup.selectbox("Sup 1", ["Pin", "Roller", "Fix"], key="s0"))
        else:
            col_sup.text(f"Sup {i+1}")
            
        spans.append(col_len.number_input(f"Length {i+1} (m)", 0.5, 20.0, 4.0, key=f"L{i}"))
        supports.append("Roller") # Intermediate default
        
    supports.append(st.selectbox(f"Sup {n_span+1}", ["Pin", "Roller", "Fix"], index=1, key=f"s_last"))
    # Allow changing intermediate supports
    # (Simplified for now, assumes Roller for intermediate)

with c2:
    st.subheader("🔹 Loads")
    
    # Load Factors Display/Edit
    cf1, cf2 = st.columns(2)
    f_dl = cf1.number_input("DL Factor", 0.0, 2.0, float(params["DL"]), step=0.1)
    f_ll = cf2.number_input("LL Factor", 0.0, 2.0, float(params["LL"]), step=0.1)
    
    loads_input = []
    
    with st.expander("📝 Edit Loads (Uniform & Point)", expanded=True):
        for i in range(n_span):
            st.markdown(f"**Span {i+1}**")
            
            # Uniform
            c_u1, c_u2 = st.columns(2)
            udl = c_u1.number_input(f"Span {i+1} DL ({U_L})", 0.0, key=f"udl_{i}")
            ull = c_u2.number_input(f"Span {i+1} LL ({U_L})", 0.0, key=f"ull_{i}")
            
            w_factored = f_dl*udl + f_ll*ull
            if w_factored > 0:
                loads_input.append({'span_idx': i, 'type': 'Uniform', 'total_w': w_factored*TO_N, 'display_val': w_factored})
            
            # Point Loads (✅ Fixed Logic)
            n_pt = st.number_input(f"Add Point Loads on Span {i+1}", 0, 5, 0, key=f"np_{i}")
            for j in range(n_pt):
                cp1, cp2, cp3 = st.columns([1, 1, 1.5])
                p_dl = cp1.number_input(f"P{j+1} DL", 0.0, key=f"pdl_{i}_{j}")
                p_ll = cp2.number_input(f"P{j+1} LL", 0.0, key=f"pll_{i}_{j}")
                x_pos = cp3.number_input(f"Dist from Left (m)", 0.0, spans[i], spans[i]/2, key=f"px_{i}_{j}")
                
                p_factored = f_dl*p_dl + f_ll*p_ll
                if p_factored > 0:
                    loads_input.append({
                        'span_idx': i, 'type': 'Point', 
                        'total_w': p_factored*TO_N, 
                        'pos': x_pos, 
                        'display_val': p_factored
                    })

# ==========================================
# 3. ANALYSIS & DESIGN
# ==========================================
st.markdown("---")
if st.button("🚀 Run Analysis & Design", type="primary", use_container_width=True):
    solver = SimpleBeamSolver(spans, supports, loads_input)
    u, err = solver.solve()
    
    if err:
        st.error(f"Analysis Error: {err}")
    else:
        # Save results to session
        df = solver.get_internal_forces(100)
        df['V_disp'] = df['shear'] * FROM_N
        df['M_disp'] = df['moment'] * FROM_N
        
        st.session_state['res_df'] = df
        st.session_state['inputs'] = (spans, supports, loads_input)
        st.session_state['analyzed'] = True

if st.session_state.get('analyzed', False):
    df = st.session_state['res_df']
    spans, supports, loads_input = st.session_state['inputs']
    
    # --- Visuals ---
    st.plotly_chart(draw_beam_schema(spans, supports, loads_input), use_container_width=True)
    
    g1, g2 = st.columns(2)
    
    # SFD
    fig_v = go.Figure(go.Scatter(x=df['x'], y=df['V_disp'], fill='tozeroy', line=dict(color='#D32F2F'), name='Shear'))
    v_max, v_min = df['V_disp'].max(), df['V_disp'].min()
    fig_v.add_annotation(x=df.loc[df['V_disp'].idxmax(), 'x'], y=v_max, text=f"{v_max:.2f}", showarrow=False, yshift=10)
    fig_v.add_annotation(x=df.loc[df['V_disp'].idxmin(), 'x'], y=v_min, text=f"{v_min:.2f}", showarrow=False, yshift=-10)
    fig_v.update_layout(title=f"Shear Force (SFD) [{U_F}]", hovermode="x")
    g1.plotly_chart(fig_v, use_container_width=True)
    
    # BMD
    fig_m = go.Figure(go.Scatter(x=df['x'], y=df['M_disp'], fill='tozeroy', line=dict(color='#1565C0'), name='Moment'))
    m_max, m_min = df['M_disp'].max(), df['M_disp'].min()
    fig_m.add_annotation(x=df.loc[df['M_disp'].idxmax(), 'x'], y=m_max, text=f"{m_max:.2f}", showarrow=False, yshift=10)
    fig_m.add_annotation(x=df.loc[df['M_disp'].idxmin(), 'x'], y=m_min, text=f"{m_min:.2f}", showarrow=False, yshift=-10)
    fig_m.update_layout(title=f"Bending Moment (BMD) [{U_M}]", yaxis=dict(autorange="reversed"))
    g2.plotly_chart(fig_m, use_container_width=True)

    # ==========================================
    # 4. DETAILED DESIGN
    # ==========================================
    st.header(f"🛠️ Design Calculation ({design_code})")
    
    # Inputs for Design
    col_mat, col_sect, col_rebar = st.columns(3)
    with col_mat:
        st.markdown("**Material Properties**")
        fc = st.number_input("f'c (ksc)", value=240.0)
        fy = st.number_input("fy (ksc)", value=4000.0)
        fys = st.number_input("Stirrup fy (ksc)", value=2400.0)
    with col_sect:
        st.markdown("**Section Size**")
        b = st.number_input("b (cm)", value=25.0)
        h = st.number_input("h (cm)", value=50.0)
        cover = st.number_input("Cover (cm)", value=3.0)
    with col_rebar:
        st.markdown("**Reinforcement**")
        bar_db = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
        main_bar = st.selectbox("Main Bar", list(bar_db.keys()), index=1)
        stirrup_sel = st.radio("Stirrup", ["RB9", "DB12"], horizontal=True)
        Av_stirrup = 2 * (0.636 if stirrup_sel=="RB9" else 1.131)

    # --- Calculation Engine ---
    # Convert everything to N, mm, MPa for calculation
    fc_mpa, fy_mpa, fys_mpa = fc*0.0981, fy*0.0981, fys*0.0981
    b_mm, d_mm = b*10, (h-cover)*10
    
    # Max Forces
    Mu_user = max(abs(m_max), abs(m_min))
    Vu_user = max(abs(v_max), abs(v_min))
    
    if "kN" in unit_sys:
        Mu_Nmm = Mu_user * 1e6
        Vu_N = Vu_user * 1000
    else:
        Mu_Nmm = Mu_user * 9.80665 * 1000
        Vu_N = Vu_user * 9.80665
        
    # --- TABS: Summary vs Detailed Report ---
    tab1, tab2 = st.tabs(["📊 Summary & Drawing", "📄 Detailed Calculation Report"])
    
    with tab1:
        # (Summary Logic from previous version - simplified)
        c_draw, c_res = st.columns([1, 1.5])
        
        # Design Logic
        status = "OK"
        if is_sdm:
            phi_b, phi_v = params['phi_b'], params['phi_v']
            Rn = Mu_Nmm / (phi_b * b_mm * d_mm**2)
            m_ratio = fy_mpa / (0.85 * fc_mpa)
            rho_req = (1/m_ratio)*(1 - np.sqrt(1 - (2*m_ratio*Rn)/fy_mpa)) if (1 - (2*m_ratio*Rn)/fy_mpa) >= 0 else 999
            
            if rho_req == 999: status = "FAIL"
            else:
                rho_min = max(np.sqrt(fc_mpa)/(4*fy_mpa), 1.4/fy_mpa)
                As_req = max(rho_req, rho_min) * b_mm * d_mm / 100
                
                # Shear
                Vc = 0.17 * np.sqrt(fc_mpa) * b_mm * d_mm
                phiVc = phi_v * Vc
                if Vu_N > phiVc:
                    Vs = (Vu_N - phiVc)/phi_v
                    s_req = (Av_stirrup*100 * fys_mpa * d_mm)/Vs
                    s_use = min(s_req, d_mm/2, 600)
                else:
                    s_use = d_mm/2 # Min stirrup
        else:
            # WSD
            n = 135/np.sqrt(fc); k = 1/(1 + (0.5*fy)/(n*0.45*fc)); j = 1-k/3
            As_req = (Mu_user*100 if "kg" in unit_sys else Mu_user*1e4/9.81) / (0.5*fy * j * (h-cover))
            s_use = 200 # Dummy for WSD simplified
            
        nb = max(2, int(np.ceil(As_req / bar_db[main_bar]))) if status=="OK" else 0
        
        with c_draw:
            if status=="FAIL": st.error("Section Failed")
            else: st.plotly_chart(draw_section_detail(b, h, cover, nb, main_bar, stirrup_sel, s_use/10), use_container_width=True)
            
        with c_res:
            if status=="OK":
                st.success(f"**Flexure:** Use {nb} - {main_bar} ($A_s$ = {nb*bar_db[main_bar]:.2f} cm²)")
                st.info(f"**Shear:** Use {stirrup_sel} @ {int(s_use/10)} cm")
                st.metric("Moment Capacity (Mu)", f"{Mu_user:.2f} {U_M}")
                st.metric("Shear Capacity (Vu)", f"{Vu_user:.2f} {U_F}")

    # --- DETAILED REPORT (THE REQUESTED FEATURE) ---
    with tab2:
        st.markdown(f"### 📑 รายการคำนวณการออกแบบ ({design_code})")
        st.markdown("---")
        
        # 1. Properties
        st.markdown("**1. Design Properties & Loads (ข้อมูลเบื้องต้น)**")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.latex(f"f'_c = {fc:.0f} \\text{{ ksc}} \\approx {fc_mpa:.2f} \\text{{ MPa}}")
            st.latex(f"f_y = {fy:.0f} \\text{{ ksc}} \\approx {fy_mpa:.2f} \\text{{ MPa}}")
            st.latex(f"b = {b:.0f} \\text{{ cm}}, \\, h = {h:.0f} \\text{{ cm}}")
        with col_r2:
            st.latex(f"d = h - cover = {d_mm/10:.1f} \\text{{ cm}}")
            st.latex(f"M_u = {Mu_user:.2f} \\text{{ {U_M}}}")
            st.latex(f"V_u = {Vu_user:.2f} \\text{{ {U_F}}}")

        st.markdown("---")
        
        # 2. Flexure Calculation (SDM Only for Report Demo, WSD is simpler)
        if is_sdm:
            st.markdown("**2. Flexure Design (ออกแบบรับแรงดัด)**")
            
            st.markdown("ตรวจสอบหน้าตัด (Check Section Capacity):")
            st.latex(f"\\phi = {phi_b}")
            st.latex(f"M_u = {Mu_Nmm/1e6:.3f} \\text{{ kNm}}")
            st.latex(f"R_n = \\frac{{M_u}}{{\\phi b d^2}} = \\frac{{{Mu_Nmm:.0f}}}{{{phi_b} \\cdot {b_mm} \\cdot {d_mm}^2}} = {Rn:.2f} \\text{{ MPa}}")
            
            term = 1 - (2*m_ratio*Rn)/fy_mpa
            if term < 0:
                st.error("❌ Section too small (หน้าตัดเล็กเกินไป รับโมเมนต์ไม่ได้)")
            else:
                st.markdown("คำนวณปริมาณเหล็กเสริม (Calculate Reinforcement):")
                st.latex(f"m = \\frac{{f_y}}{{0.85 f'_c}} = \\frac{{{fy_mpa:.2f}}}{{0.85 \\cdot {fc_mpa:.2f}}} = {m_ratio:.2f}")
                st.latex(f"\\rho_{{req}} = \\frac{{1}}{{m}}\\left(1 - \\sqrt{{1 - \\frac{{2mR_n}}{{f_y}}}}\\right) = {rho_req:.5f}")
                
                rho_min1 = np.sqrt(fc_mpa)/(4*fy_mpa)
                rho_min2 = 1.4/fy_mpa
                st.latex(f"\\rho_{{min}} = \\max\\left(\\frac{{\\sqrt{{f'_c}}}}{{4f_y}}, \\frac{{1.4}}{{f_y}}\\right) = {max(rho_min1, rho_min2):.5f}")
                
                rho_use = max(rho_req, max(rho_min1, rho_min2))
                As_calc = rho_use * b_mm * d_mm / 100
                st.latex(f"A_{{s,req}} = \\rho_{{use}} b d = {rho_use:.5f} \\cdot {b:.1f} \\cdot {d_mm/10:.1f} = \\mathbf{{{As_calc:.2f} \\text{{ cm}}^2}}")
                
                st.markdown(f"เลือกใช้เหล็ก: **{nb} - {main_bar}** ($A_s = {nb*bar_db[main_bar]:.2f} \\text{{ cm}}^2$)")

            st.markdown("---")
            st.markdown("**3. Shear Design (ออกแบบรับแรงเฉือน)**")
            
            Vc_val = 0.17 * np.sqrt(fc_mpa) * b_mm * d_mm
            phiVc_val = phi_v * Vc_val
            
            st.latex(f"V_c = 0.17 \\sqrt{{f'_c}} b d = 0.17 \\sqrt{{{fc_mpa:.2f}}} \\cdot {b_mm} \\cdot {d_mm} = {Vc_val/1000:.2f} \\text{{ kN}}")
            st.latex(f"\\phi V_c = {phi_v} \\cdot {Vc_val/1000:.2f} = {phiVc_val/1000:.2f} \\text{{ kN}}")
            
            Vu_check = Vu_N # N
            
            if Vu_check > phiVc_val:
                st.warning(f"Condition: $V_u > \\phi V_c$ ({Vu_check/1000:.2f} > {phiVc_val/1000:.2f}) $\\rightarrow$ **ต้องการเหล็กปลอก (Req. Stirrups)**")
                Vs_req = (Vu_check - phiVc_val) / phi_v
                st.latex(f"V_s = \\frac{{V_u - \\phi V_c}}{{\\phi}} = {Vs_req/1000:.2f} \\text{{ kN}}")
                
                s_cal = (Av_stirrup*100 * fys_mpa * d_mm) / Vs_req # mm
                st.latex(f"s_{{req}} = \\frac{{A_v f_{{ys}} d}}{{V_s}} = \\frac{{{Av_stirrup*100:.2f} \\cdot {fys_mpa:.2f} \\cdot {d_mm}}}{{{Vs_req:.0f}}} = {s_cal:.1f} \\text{{ mm}}")
                st.write(f"👉 Use Stirrup {stirrup_sel} @ {int(min(s_cal, 600)/10)} cm")
            elif Vu_check > 0.5 * phiVc_val:
                 st.info(f"Condition: $V_u > 0.5\\phi V_c$ $\\rightarrow$ **ใช้เหล็กปลอกขั้นต่ำ (Min Stirrups)**")
                 s_max = d_mm/2
                 st.latex(f"s_{{max}} = d/2 = {s_max:.0f} \\text{{ mm}}")
                 st.write(f"👉 Use Stirrup {stirrup_sel} @ {int(s_max/10)} cm")
            else:
                 st.success("Condition: $V_u < 0.5\\phi V_c$ $\\rightarrow$ **คอนกรีตรับแรงเฉือนได้เพียงพอ (No Stirrups theoretically required)**")

        else:
            # WSD Report Logic (Simplified)
            st.markdown("**2. Flexure & Shear (WSD Method)**")
            st.write("การคำนวณแบบ WSD ใช้หน่วยแรงใช้งานจริง:")
            st.latex(f"f_c = 0.45 f'_c = {0.45*fc:.1f} \\text{{ ksc}}")
            st.latex(f"f_s = 0.5 f_y = {0.5*fy:.0f} \\text{{ ksc}}")
            st.latex(f"M = {Mu_user:.2f} \\text{{ {U_M}}}")
            st.latex(f"A_s = \\frac{{M}}{{f_s j d}} = \\mathbf{{{As_req:.2f} \\text{{ cm}}^2}}")
