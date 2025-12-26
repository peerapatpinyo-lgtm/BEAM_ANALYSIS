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
# ⚙️ GLOBAL CONFIG
# ==========================================
st.set_page_config(page_title="Beam Design Pro", layout="wide")

# --- SIDEBAR: SETTINGS ---
st.sidebar.header("⚙️ Project Settings (ตั้งค่าโครงการ)")

# 1. Code Selection
design_code = st.sidebar.selectbox(
    "1. Design Code (มาตรฐานการออกแบบ)",
    ["EIT 1007 (WSD - วิธีหน่วยแรงใช้งาน)", "ACI 318 / EIT (SDM - วิธีกำลัง)"]
)
is_sdm = "SDM" in design_code

# 2. Unit Selection
unit_opt = st.sidebar.radio("2. Unit System (ระบบหน่วย)", ["SI Units (kN, m)", "MKS Units (kg, m)"])

# Define Unit Variables
if "kN" in unit_opt:
    UNIT_F = "kN"; UNIT_L = "kN/m"; UNIT_M = "kN-m"; UNIT_S = "MPa"
    TO_N = 1000.0; FROM_N = 1/1000.0; TO_MPA = 1.0
else:
    UNIT_F = "kg"; UNIT_L = "kg/m"; UNIT_M = "kg-m"; UNIT_S = "ksc"
    TO_N = 9.80665; FROM_N = 1/9.80665; TO_MPA = 0.0980665

# ==========================================
# 🎨 HELPER FUNCTIONS
# ==========================================
def draw_beam(spans, supports, loads):
    fig = go.Figure()
    # Beam Line
    fig.add_trace(go.Scatter(x=[0, sum(spans)], y=[0, 0], mode='lines', line=dict(color='black', width=5), name='Beam'))
    
    cx = 0; sx = 0
    # Dimensions
    for L in spans:
        fig.add_annotation(x=cx+L/2, y=-0.6, text=f"{L} m", showarrow=False, font=dict(color="blue"))
        fig.add_shape(type="line", x0=cx+L, y0=-0.3, x1=cx+L, y1=0.3, line=dict(color="gray", dash="dot"))
        cx += L
        
    # Supports
    for i, s in enumerate(supports):
        sym = "triangle-up" if s != "Fix" else "square"
        col = "green" if s != "Fix" else "red"
        fig.add_trace(go.Scatter(x=[sx], y=[-0.2], mode='markers', marker=dict(symbol=sym, size=14, color=col), showlegend=False, hovertext=f"Support {i+1}: {s}"))
        if i < len(spans): sx += spans[i]

    # Loads
    for ld in loads:
        start_x = sum(spans[:ld['span_idx']])
        val_disp = ld['display_val']
        if ld['type'] == 'Point':
            fig.add_annotation(x=start_x+ld['pos'], y=0.1, ax=0, ay=-40, text=f"{val_disp:.1f}", showarrow=True, arrowhead=2, arrowcolor="red")
        elif ld['type'] == 'Uniform':
            end_x = start_x + spans[ld['span_idx']]
            fig.add_shape(type="rect", x0=start_x, y0=0, x1=end_x, y1=0.25, fillcolor="rgba(255,0,0,0.1)", line_width=0)
            fig.add_annotation(x=(start_x+end_x)/2, y=0.3, text=f"{val_disp:.1f}", showarrow=False, font=dict(color="red"))

    fig.update_layout(height=250, xaxis=dict(showgrid=False, visible=True, title="Distance (m)"), yaxis=dict(visible=False, range=[-1, 1.5]), margin=dict(t=30, b=20), plot_bgcolor="white")
    return fig

# ==========================================
# 🖥️ MAIN UI
# ==========================================
st.title(f"🏗️ RC Beam Design")
st.markdown(f"**Code:** {design_code} | **Unit:** {unit_opt}")
st.markdown("---")

# ------------------------------------------------
# SECTION 1: GEOMETRY SETUP
# ------------------------------------------------
st.header("1. Geometry Configuration (กำหนดขนาดและจุดรองรับ)")

# Number of Spans
c_nspan, c_dummy = st.columns([1, 3])
with c_nspan:
    n_span = st.number_input("Number of Spans (จำนวนช่วงคาน)", min_value=1, max_value=6, value=2)

st.info("กรุณาระบุความยาวและประเภทจุดรองรับสำหรับแต่ละช่วง")

spans = []
supports = []

# --- Dynamic Input Rows ---
# ใช้ Container เพื่อจัดกลุ่ม Input ให้ดูเต็มตา ไม่เบียด
for i in range(n_span):
    with st.container():
        st.markdown(f"##### 📌 Span {i+1}")
        c1, c2, c3 = st.columns([1.5, 1.5, 1.5])
        
        # Support Left (Only for first span, others share)
        if i == 0:
            with c1:
                s_type = st.selectbox(f"Support {i+1} Type", ['Pin', 'Roller', 'Fix'], key=f"sup_{i}")
                supports.append(s_type)
        else:
            with c1:
                st.markdown(f"**Support {i+1}**")
                st.caption(f"(Connected to Span {i})")

        # Span Length
        with c2:
            l_val = st.number_input(f"Span {i+1} Length (m)", min_value=0.5, max_value=20.0, value=4.0, key=f"len_{i}")
            spans.append(l_val)
            
        # Support Right (Current Span's End)
        with c3:
            # Note: This will be the start of next span, or the end of beam
            s_next_label = f"Support {i+2} Type"
            s_next = st.selectbox(s_next_label, ['Pin', 'Roller', 'Fix'], index=1, key=f"sup_{i+1}")
            supports.append(s_next)
    
    st.divider() # เส้นคั่นแต่ละ Span

# ------------------------------------------------
# SECTION 2: LOADS
# ------------------------------------------------
st.header("2. Loads Input (น้ำหนักบรรทุก)")

# Factors
if is_sdm:
    st.write("🔧 **Load Factors for SDM**")
    cf1, cf2, cf3 = st.columns(4)
    f_dl = cf1.number_input("Dead Load Factor", 1.0, 2.0, 1.4, step=0.1)
    f_ll = cf2.number_input("Live Load Factor", 1.0, 2.0, 1.7, step=0.1)
    st.caption(f"Total Load = {f_dl:.1f}DL + {f_ll:.1f}LL")
else:
    st.info("💡 **WSD Mode:** Using Service Loads (Factor = 1.0)")
    f_dl = 1.0; f_ll = 1.0

# Loads Per Span
loads_input = []

for i in range(n_span):
    with st.expander(f"📍 Loads on Span {i+1} (น้ำหนักบนช่วงที่ {i+1})", expanded=True):
        c_uni, c_pt = st.columns([1, 1.5])
        
        # Uniform Load
        with c_uni:
            st.markdown("**Uniform Distributed Load**")
            udl = st.number_input(f"Dead Load (DL) [{UNIT_L}]", 0.0, key=f"u_dl_{i}")
            ull = st.number_input(f"Live Load (LL) [{UNIT_L}]", 0.0, key=f"u_ll_{i}")
            
            total_u = (f_dl * udl + f_ll * ull)
            if total_u > 0:
                loads_input.append({
                    'span_idx': i, 'type': 'Uniform',
                    'total_w': total_u * TO_N,
                    'display_val': total_u
                })
        
        # Point Loads
        with c_pt:
            st.markdown("**Point Loads (Concentrated)**")
            n_pt = st.number_input(f"Number of Point Loads", 0, 5, 0, key=f"n_pt_{i}")
            
            for j in range(n_pt):
                cc1, cc2, cc3 = st.columns(3)
                pd = cc1.number_input(f"P{j+1} DL [{UNIT_F}]", 0.0, key=f"pd_{i}_{j}")
                pl = cc2.number_input(f"P{j+1} LL [{UNIT_F}]", 0.0, key=f"pl_{i}_{j}")
                pp = cc3.number_input(f"Distance from Left (m)", 0.0, spans[i], spans[i]/2, key=f"pp_{i}_{j}")
                
                total_p = (f_dl * pd + f_ll * pl)
                if total_p > 0:
                    loads_input.append({
                        'span_idx': i, 'type': 'Point',
                        'total_w': total_p * TO_N,
                        'pos': pp,
                        'display_val': total_p
                    })

# ------------------------------------------------
# ACTION BUTTON
# ------------------------------------------------
st.markdown("---")
if st.button("🚀 Run Analysis & Design", type="primary", use_container_width=True):
    solver = SimpleBeamSolver(spans, supports, loads_input)
    u, err = solver.solve()
    
    if err:
        st.error(f"Analysis Error: {err}")
    else:
        # Process Results
        df = solver.get_internal_forces(100)
        df['shear_d'] = df['shear'] * FROM_N
        df['moment_d'] = df['moment'] * FROM_N
        
        st.session_state['res'] = df
        st.session_state['viz'] = draw_beam(spans, supports, loads_input)
        st.session_state['done'] = True

# ==========================================
# 📊 RESULTS
# ==========================================
if st.session_state.get('done', False):
    df = st.session_state['res']
    
    st.header("3. Analysis Results (ผลการวิเคราะห์)")
    st.plotly_chart(st.session_state['viz'], use_container_width=True)
    
    # Diagrams
    c1, c2 = st.columns(2)
    
    # Shear Diagram
    fig_v = go.Figure(go.Scatter(x=df['x'], y=df['shear_d'], fill='tozeroy', line=dict(color='#D32F2F')))
    fig_v.update_layout(title=f"Shear Force Diagram (SFD) [{UNIT_F}]", hovermode="x")
    c1.plotly_chart(fig_v, use_container_width=True)
    
    # Moment Diagram
    fig_m = go.Figure(go.Scatter(x=df['x'], y=df['moment_d'], fill='tozeroy', line=dict(color='#1976D2')))
    fig_m.update_layout(title=f"Bending Moment Diagram (BMD) [{UNIT_M}]", yaxis=dict(autorange="reversed"))
    c2.plotly_chart(fig_m, use_container_width=True)
    
    # Values
    v_max = df['shear_d'].abs().max()
    m_max = max(abs(df['moment_d'].max()), abs(df['moment_d'].min()))
    
    st.info(f"**Max Shear (Vu):** {v_max:.2f} {UNIT_F} | **Max Moment (Mu):** {m_max:.2f} {UNIT_M}")

    # ==========================================
    # 📝 DESIGN SECTION
    # ==========================================
    st.markdown("---")
    st.header(f"4. Design Calculation ({'SDM/ACI' if is_sdm else 'WSD/EIT'})")
    
    with st.form("design_panel"):
        st.subheader("🛠️ Design Parameters")
        col_mat, col_sect, col_bar = st.columns(3)
        
        with col_mat:
            st.markdown("**Material**")
            fc = st.number_input("Concrete f'c (ksc)", value=240.0)
            fy = st.number_input("Steel fy (ksc)", value=4000.0)
            
        with col_sect:
            st.markdown("**Section Size**")
            b_val = st.number_input("Width b (cm)", 15.0, 100.0, 25.0)
            h_val = st.number_input("Depth h (cm)", 20.0, 200.0, 50.0)
            cover = st.number_input("Covering (cm)", 2.0, 5.0, 3.0)
            
        with col_bar:
            st.markdown("**Reinforcement**")
            bar_map = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
            main_bar = st.selectbox("Main Bar Size", list(bar_map.keys()), index=1)
            
        st.form_submit_button("🔄 Recalculate Design")

    # --- CALCULATION LOGIC ---
    # Prepare Data
    fc_mpa = fc * 0.0980665
    fy_mpa = fy * 0.0980665
    b_mm, d_mm = b_val*10, (h_val-cover)*10
    
    # Convert Forces to Design Units (N, mm)
    if "kN" in unit_opt:
        M_des_Nmm = m_max * 1e6
        V_des_N = v_max * 1000
    else:
        M_des_Nmm = m_max * 9.80665 * 1000
        V_des_N = v_max * 9.80665

    c_res, c_sheet = st.columns([1, 1.2])

    # --- SDM ---
    if is_sdm:
        with c_res:
            st.markdown("### ✅ Design Results (SDM)")
            phi = 0.9
            Rn = (M_des_Nmm / phi) / (b_mm * d_mm**2)
            m_rat = fy_mpa / (0.85 * fc_mpa)
            term = 1 - (2*m_rat*Rn)/fy_mpa
            
            if term < 0:
                st.error("❌ Section Failed (Too small)")
            else:
                rho_req = (1/m_rat)*(1 - np.sqrt(term))
                rho_min = max(np.sqrt(fc_mpa)/(4*fy_mpa), 1.4/fy_mpa)
                rho_use = max(rho_req, rho_min)
                As_req = rho_use * b_mm * d_mm / 100 # cm2
                
                nb = max(2, int(np.ceil(As_req / bar_map[main_bar])))
                st.success(f"**Flexure:** Use {nb} - {main_bar}")
                st.metric("As Required", f"{As_req:.2f} cm²")
                
                # Shear
                Vc_N = 0.17 * np.sqrt(fc_mpa) * b_mm * d_mm
                phiVc = 0.85 * Vc_N * FROM_N
                st.markdown("---")
                if V_des_N <= 0.85 * Vc_N:
                    st.info(f"**Shear:** Min Stirrups (Vu < phiVc)")
                else:
                    st.warning(f"**Shear:** Stirrups Required (Vu > phiVc)")

        with c_sheet:
            with st.expander("📝 Detailed Calculation (SDM)", expanded=True):
                st.latex(f"M_u = {m_max:.2f} \\text{{ {UNIT_M}}}")
                st.latex(f"R_n = {Rn:.2f} \\text{{ MPa}}")
                st.latex(f"\\rho_{{req}} = {rho_req:.4f} \\rightarrow \\text{{Use }} {rho_use:.4f}")
                st.latex(f"A_s = {As_req:.2f} \\text{{ cm}}^2")

    # --- WSD ---
    else:
        with c_res:
            st.markdown("### ✅ Design Results (WSD)")
            n = round(2.04e6 / (15100*np.sqrt(fc)))
            fc_all = 0.45 * fc
            fs_all = min(0.5 * fy, 2500) # Cap fs
            k = 1 / (1 + fs_all/(n*fc_all))
            j = 1 - k/3
            
            # Moment Check
            Mc_kgm = (0.5 * fc_all * k * j * b_val * (h_val-cover)**2) / 100
            M_chk = m_max * (1000/9.81) if "kN" in unit_opt else m_max
            
            if M_chk > Mc_kgm:
                st.error(f"❌ Concrete Fail (Mc={Mc_kgm:.0f} < M={M_chk:.0f})")
            else:
                As_req = (M_chk * 100) / (fs_all * j * (h_val-cover))
                nb = max(2, int(np.ceil(As_req / bar_map[main_bar])))
                st.success(f"**Flexure:** Use {nb} - {main_bar}")
                st.metric("As Required", f"{As_req:.2f} cm²")
                
                # Shear
                v_act = (V_des_N/9.81) / (b_val * (h_val-cover))
                vc_all = 0.29 * np.sqrt(fc)
                st.markdown("---")
                if v_act > vc_all:
                    st.warning(f"**Shear:** Stirrups Required (v={v_act:.1f} > vc={vc_all:.1f})")
                else:
                    st.info(f"**Shear:** Concrete OK (v={v_act:.1f} < vc={vc_all:.1f})")

        with c_sheet:
            with st.expander("📝 Detailed Calculation (WSD)", expanded=True):
                st.latex(f"n={n}, k={k:.3f}, j={j:.3f}")
                st.latex(f"M_{{design}} = {M_chk:.0f} \\text{{ kg-m}}")
                st.latex(f"A_s = \\frac{{M}}{{f_s j d}} = {As_req:.2f} \\text{{ cm}}^2")
