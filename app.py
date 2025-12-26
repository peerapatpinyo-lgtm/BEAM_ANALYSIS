import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- IMPORT ENGINE ---
try:
    from beam_engine import SimpleBeamSolver
except ImportError:
    st.error("⚠️ ไม่พบไฟล์ 'beam_engine.py' กรุณาวางไฟล์ไว้ในโฟลเดอร์เดียวกัน")
    st.stop()

# ==========================================
# ⚙️ GLOBAL CONFIG
# ==========================================
st.set_page_config(page_title="Beam Design Pro (Thai/ACI)", layout="wide")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Project Settings")

# 1. Select Code
design_code = st.sidebar.selectbox(
    "1. เลือกมาตรฐานการออกแบบ (Design Code)",
    ["EIT 1007 (WSD - วิธีหน่วยแรงใช้งาน)", "ACI 318 / EIT (SDM - วิธีกำลัง)"]
)

is_sdm = "SDM" in design_code

# 2. Select Unit
unit_opt = st.sidebar.radio("2. ระบบหน่วย (Units)", ["SI Units (kN, m)", "MKS Units (kg, m)"])

# Define Unit Factors
if "kN" in unit_opt:
    UNIT_F = "kN"
    UNIT_L = "kN/m"
    UNIT_M = "kN-m"
    UNIT_S = "MPa" # Stress Unit
    TO_N = 1000.0  
    FROM_N = 1/1000.0 
    TO_MPA = 1.0
else:
    UNIT_F = "kg"
    UNIT_L = "kg/m"
    UNIT_M = "kg-m"
    UNIT_S = "ksc" # Stress Unit
    TO_N = 9.80665 
    FROM_N = 1/9.80665
    TO_MPA = 0.0980665 # ksc -> MPa

# ==========================================
# 🎨 VISUALIZATION HELPER
# ==========================================
def add_peak_labels(fig, x, y, inverted=False):
    if len(y) == 0: return
    max_i, min_i = np.argmax(y), np.argmin(y)
    peaks = [(x[max_i], y[max_i]), (x[min_i], y[min_i])]
    
    for px, py in peaks:
        shift = 15 if (py >= 0 and not inverted) or (py < 0 and inverted) else -15
        fig.add_annotation(
            x=px, y=py, text=f"{py:.2f}",
            showarrow=False, yshift=shift,
            font=dict(color="black", size=11), bgcolor="rgba(255,255,255,0.7)"
        )

def draw_beam(spans, supports, loads):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, sum(spans)], y=[0, 0], mode='lines', line=dict(color='black', width=5)))
    
    cx = 0
    sx = 0
    for L in spans:
        fig.add_annotation(x=cx+L/2, y=-0.6, text=f"{L} m", showarrow=False, font=dict(color="blue"))
        fig.add_shape(type="line", x0=cx+L, y0=-0.3, x1=cx+L, y1=0.3, line=dict(color="gray", dash="dot"))
        cx += L
        
    for i, s in enumerate(supports):
        sym = "triangle-up" if s != "Fix" else "square"
        col = "green" if s != "Fix" else "red"
        fig.add_trace(go.Scatter(x=[sx], y=[-0.2], mode='markers', marker=dict(symbol=sym, size=14, color=col), showlegend=False))
        if i < len(spans): sx += spans[i]

    for ld in loads:
        start_x = sum(spans[:ld['span_idx']])
        val_disp = ld['display_val']
        if ld['type'] == 'Point':
            fig.add_annotation(x=start_x+ld['pos'], y=0.1, ax=0, ay=-40, text=f"{val_disp:.1f}", showarrow=True, arrowhead=2, arrowcolor="red")
        elif ld['type'] == 'Uniform':
            end_x = start_x + spans[ld['span_idx']]
            fig.add_shape(type="rect", x0=start_x, y0=0, x1=end_x, y1=0.3, fillcolor="rgba(255,0,0,0.1)", line_width=0)
            fig.add_annotation(x=(start_x+end_x)/2, y=0.35, text=f"{val_disp:.1f}", showarrow=False, font=dict(color="red"))

    fig.update_layout(height=280, xaxis=dict(showgrid=False, visible=True), yaxis=dict(visible=False, range=[-1, 1.5]), margin=dict(t=30, b=20))
    return fig

# ==========================================
# 🖥️ MAIN INPUT UI
# ==========================================
st.title(f"🏗️ RC Beam Design: {design_code}")

# --- 1. GEOMETRY ---
with st.expander("1. Geometry & Supports", expanded=True):
    c1, c2 = st.columns([1, 3])
    n_span = c1.number_input("Spans", 1, 6, 2)
    spans = [st.columns(n_span)[i].number_input(f"L{i+1} (m)", 1.0, 20.0, 4.0) for i in range(n_span)]
    supports = [st.columns(n_span+1)[i].selectbox(f"Sup{i+1}", ['Pin', 'Roller', 'Fix'], index=0 if i==0 else 1) for i in range(n_span+1)]

# --- 2. LOADS & FACTORS ---
st.subheader("2. Loads Definition")

# Factor Settings
if is_sdm:
    st.info("💡 **SDM Mode:** Please specify Load Factors (Common: 1.4DL + 1.7LL or 1.2DL + 1.6LL)")
    cf1, cf2, cf3 = st.columns(3)
    f_dl = cf1.number_input("DL Factor", 1.0, 2.0, 1.4, step=0.1)
    f_ll = cf2.number_input("LL Factor", 1.0, 2.0, 1.7, step=0.1)
else:
    st.info("💡 **WSD Mode:** Service Loads (Factor = 1.0) are used for analysis.")
    f_dl = 1.0
    f_ll = 1.0

# Load Inputs
loads_input = []
cols_L = st.columns(n_span)
for i in range(n_span):
    with cols_L[i]:
        st.markdown(f"**Span {i+1}**")
        udl = st.number_input(f"UDL-DL ({UNIT_L})", 0.0, key=f"udl_{i}")
        ull = st.number_input(f"UDL-LL ({UNIT_L})", 0.0, key=f"ull_{i}")
        
        n_pt = st.number_input(f"Point Loads", 0, 5, 0, key=f"np_{i}")
        for j in range(n_pt):
            pd = st.number_input(f"P{j+1}-DL", 0.0, key=f"pd_{i}_{j}")
            pl = st.number_input(f"P{j+1}-LL", 0.0, key=f"pl_{i}_{j}")
            pp = st.number_input(f"Pos (m)", 0.0, spans[i], spans[i]/2, key=f"pp_{i}_{j}")
            
            # Combine Load based on Factors
            total_p = (f_dl*pd + f_ll*pl)
            if total_p > 0:
                loads_input.append({
                    'span_idx': i, 'type': 'Point', 
                    'total_w': total_p * TO_N, # To Newton
                    'pos': pp,
                    'display_val': total_p # Show Factored/Service Value
                })
        
        total_u = (f_dl*udl + f_ll*ull)
        if total_u > 0:
            loads_input.append({
                'span_idx': i, 'type': 'Uniform', 
                'total_w': total_u * TO_N, # To N/m
                'display_val': total_u
            })

# --- CALCULATE ---
if st.button("🚀 Analyze Beam", type="primary", use_container_width=True):
    solver = SimpleBeamSolver(spans, supports, loads_input)
    u, err = solver.solve()
    
    if err:
        st.error(err)
    else:
        df = solver.get_internal_forces(100)
        # Convert output to selected unit
        df['shear_d'] = df['shear'] * FROM_N
        df['moment_d'] = df['moment'] * FROM_N
        
        st.session_state['res'] = df
        st.session_state['viz'] = draw_beam(spans, supports, loads_input)
        st.session_state['analyzed'] = True

# ==========================================
# 📊 RESULTS & DESIGN
# ==========================================
if st.session_state.get('analyzed', False):
    df = st.session_state['res']
    
    # Show Model
    st.plotly_chart(st.session_state['viz'], use_container_width=True)
    
    # Graphs
    c1, c2 = st.columns(2)
    fig_v = go.Figure(go.Scatter(x=df['x'], y=df['shear_d'], fill='tozeroy', line=dict(color='#D32F2F')))
    add_peak_labels(fig_v, df['x'], df['shear_d'])
    fig_v.update_layout(title=f"Shear Force ({UNIT_F})", hovermode="x")
    c1.plotly_chart(fig_v, use_container_width=True)
    
    fig_m = go.Figure(go.Scatter(x=df['x'], y=df['moment_d'], fill='tozeroy', line=dict(color='#1976D2')))
    add_peak_labels(fig_m, df['x'], df['moment_d'], inverted=True)
    fig_m.update_layout(title=f"Bending Moment ({UNIT_M})", yaxis=dict(autorange="reversed"))
    c2.plotly_chart(fig_m, use_container_width=True)

    # ==========================================
    # 📝 DETAILED DESIGN SECTION
    # ==========================================
    st.markdown("---")
    st.header(f"🛠️ Detailed Design Calculation ({'SDM' if is_sdm else 'WSD'})")
    
    # --- Design Input Panel ---
    with st.form("design_form"):
        col_mat, col_sect, col_bar = st.columns(3)
        with col_mat:
            st.markdown("###### Material Properties")
            fc = st.number_input("f'c (ksc)", value=240.0)
            fy = st.number_input("fy (ksc)", value=4000.0)
        with col_sect:
            st.markdown("###### Section Size")
            b_val = st.number_input("b (cm)", 15.0, 100.0, 25.0)
            h_val = st.number_input("h (cm)", 20.0, 200.0, 50.0)
            cover = st.number_input("Covering (cm)", 2.0, 5.0, 3.0)
        with col_bar:
            st.markdown("###### Reinforcement")
            bar_map = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
            main_bar = st.selectbox("Main Bar Size", list(bar_map.keys()), index=1)
        
        st.form_submit_button("🔄 Recalculate Design")

    # --- Calculation Logic ---
    
    # 1. Prepare Variables
    d_val = h_val - cover
    # Convert to SI for calc (N, mm, MPa)
    fc_mpa = fc * 0.0980665
    fy_mpa = fy * 0.0980665
    b_mm, d_mm = b_val*10, d_val*10
    
    # Get Max Forces
    M_max_val = max(abs(df['moment_d'].max()), abs(df['moment_d'].min()))
    V_max_val = df['shear_d'].abs().max()
    
    # Convert Moment to N-mm
    if "kN" in unit_opt:
        M_design_Nmm = M_max_val * 1e6
        V_design_N = V_max_val * 1000
    else:
        M_design_Nmm = M_max_val * 9.80665 * 1000
        V_design_N = V_max_val * 9.80665

    col_res, col_cal = st.columns([1, 1.5])
    
    # ==========================
    # 🅰️ CASE 1: SDM (Strength)
    # ==========================
    if is_sdm:
        with col_res:
            st.subheader("Results (SDM)")
            phi = 0.9
            Mn_req = M_design_Nmm / phi
            Rn = Mn_req / (b_mm * d_mm**2) # MPa
            
            m = fy_mpa / (0.85 * fc_mpa)
            term = 1 - (2 * m * Rn) / fy_mpa
            
            if term < 0:
                st.error(f"❌ **FAIL**: Section Too Small (Rn={Rn:.2f})")
            else:
                rho_req = (1/m)*(1 - np.sqrt(term))
                rho_min = max(np.sqrt(fc_mpa)/(4*fy_mpa), 1.4/fy_mpa)
                rho_final = max(rho_req, rho_min)
                
                As_req = rho_final * b_mm * d_mm # mm2
                As_cm2 = As_req / 100
                
                # Bars
                bar_area = bar_map[main_bar]
                nb = max(2, int(np.ceil(As_cm2 / bar_area)))
                
                st.success(f"✅ **PASS** | Use {nb}-{main_bar}")
                st.metric("Required As", f"{As_cm2:.2f} cm²")
                st.metric(f"Provided As ({nb}-{main_bar})", f"{nb*bar_area:.2f} cm²")

        with col_cal:
            with st.expander("📝 รายการคำนวณละเอียด (SDM)", expanded=True):
                st.markdown("**1. Design Parameters**")
                st.latex(f"f'_c = {fc:.0f} \\text{{ ksc}}, \\quad f_y = {fy:.0f} \\text{{ ksc}}")
                st.latex(f"M_u = {M_max_val:.2f} \\text{{ {UNIT_M}}}")
                st.latex(f"d = {h_val} - {cover} = {d_val:.1f} \\text{{ cm}}")

                st.markdown("**2. Flexural Calculation**")
                st.latex(f"R_n = \\frac{{M_u}}{{\\phi b d^2}} = \\frac{{{M_max_val} \\times 10^6}}{{0.9 \\times {b_mm} \\times {d_mm}^2}} = {Rn:.2f} \\text{{ MPa}}")
                st.latex(f"\\rho_{{req}} = \\frac{{1}}{{m}}\\left(1-\\sqrt{{1-\\frac{{2mR_n}}{{f_y}}}}\\right) = {rho_req:.5f}")
                st.latex(f"\\rho_{{min}} = {rho_min:.5f} \\rightarrow \\text{{Use }} \\rho = {rho_final:.5f}")
                st.latex(f"A_s = \\rho b d = {rho_final:.5f} \\times {b_val} \\times {d_val} = \\mathbf{{{As_cm2:.2f} \\text{{ cm}}^2}}")

                st.markdown("**3. Shear Check**")
                Vc_N = 0.17 * np.sqrt(fc_mpa) * b_mm * d_mm
                phiVc = 0.85 * Vc_N * FROM_N # convert to display unit
                
                st.latex(f"V_u = {V_max_val:.2f} \\text{{ {UNIT_F}}}")
                st.latex(f"\\phi V_c = 0.85(0.17\\sqrt{{f'_c}}bd) = {phiVc:.2f} \\text{{ {UNIT_F}}}")
                if V_design_N > 0.85*Vc_N:
                     st.write("🔴 **Stirrups Required:** $V_u > \\phi V_c$")
                else:
                     st.write("🟢 **Min Stirrups OK:** $V_u < \\phi V_c$")

    # ==========================
    # 🅱️ CASE 2: WSD (Working Stress)
    # ==========================
    else:
        with col_res:
            st.subheader("Results (WSD)")
            # Allowable Stresses (EIT 1007 typical)
            fc_allow = 0.45 * fc # ksc
            fs_allow = 0.5 * fy  # ksc (Simplified, normally capped at 1700 or 2500 depending on steel grade)
            if fs_allow > 2500: fs_allow = 2500 # Cap reasonable for high strength bar in WSD
            
            # Constants
            Es = 2.04e6 # ksc
            Ec = 15100 * np.sqrt(fc) # ksc
            n = round(Es / Ec)
            
            k = 1 / (1 + fs_allow / (n * fc_allow))
            j = 1 - k/3
            
            # Moment Capacity of Concrete
            Mc_kgm = (0.5 * fc_allow * k * j * b_val * d_val**2) / 100 # kg-m
            
            # Convert M_design to kg-m for comparison
            if "kN" in unit_opt:
                M_chk_kgm = M_max_val * (1000/9.81)
            else:
                M_chk_kgm = M_max_val
            
            if M_chk_kgm > Mc_kgm:
                 st.error(f"❌ **FAIL**: Concrete Crush (Mc={Mc_kgm:.0f} < M={M_chk_kgm:.0f})")
                 st.write("Increase Section Size (Doubly reinforced not supported here)")
            else:
                # Steel Area
                As_req_cm2 = (M_chk_kgm * 100) / (fs_allow * j * d_val)
                
                # Bars
                bar_area = bar_map[main_bar]
                nb = max(2, int(np.ceil(As_req_cm2 / bar_area)))
                
                st.success(f"✅ **PASS** | Use {nb}-{main_bar}")
                st.metric("Required As", f"{As_req_cm2:.2f} cm²")
                st.metric(f"Provided As ({nb}-{main_bar})", f"{nb*bar_area:.2f} cm²")

        with col_cal:
            with st.expander("📝 รายการคำนวณละเอียด (WSD/EIT 1007)", expanded=True):
                st.markdown("**1. Allowable Stresses & Constants**")
                st.latex(f"f_c = 0.45 f'_c = {fc_allow:.2f} \\text{{ ksc}}")
                st.latex(f"f_s = 0.5 f_y = {fs_allow:.0f} \\text{{ ksc}}")
                st.latex(f"n = E_s/E_c \\approx {n}")
                st.latex(f"k = \\frac{{1}}{{1 + f_s/(n f_c)}} = {k:.3f}, \\quad j = 1 - k/3 = {j:.3f}")
                
                st.markdown("**2. Moment Calculation**")
                st.write(f"ตรวจสอบโมเมนต์ต้านทานของคอนกรีต ($M_c$):")
                st.latex(f"M_c = \\frac{{1}}{{2}} f_c k j b d^2 = {Mc_kgm:.0f} \\text{{ kg-m}}")
                st.latex(f"M_{{design}} = {M_chk_kgm:.0f} \\text{{ kg-m}}")
                
                st.markdown("**3. Steel Area Calculation**")
                st.latex(f"A_s = \\frac{{M}}{{f_s j d}} = \\frac{{{M_chk_kgm:.0f} \\times 100}}{{{fs_allow:.0f} \\times {j:.3f} \\times {d_val:.1f}}} = \\mathbf{{{As_req_cm2:.2f} \\text{{ cm}}^2}}")

                st.markdown("**4. Shear Check (Brief)**")
                vc_allow = 0.29 * np.sqrt(fc) # ksc
                st.latex(f"v_c = 0.29\\sqrt{{f'_c}} = {vc_allow:.2f} \\text{{ ksc}}")
                
                # V actual
                V_chk_kg = V_design_N / 9.80665
                v_act = V_chk_kg / (b_val * d_val) # ksc
                st.latex(f"v = \\frac{{V}}{{bd}} = \\frac{{{V_chk_kg:.0f}}}{{{b_val}\\times{d_val}}} = {v_act:.2f} \\text{{ ksc}}")
                
                if v_act > vc_allow:
                    st.write("🔴 **Stirrups Required** ($v > v_c$)")
                else:
                    st.write("🟢 **Concrete Shear OK**")
