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
# ⚙️ CONFIG & UNITS
# ==========================================
st.set_page_config(page_title="Beam Analysis & Design (Full)", layout="wide")

# Sidebar Settings
st.sidebar.header("⚙️ ตั้งค่า (Settings)")
unit_opt = st.sidebar.radio("ระบบหน่วย (Unit System)", ["SI Units (kN, m)", "MKS Units (kg, m)"])

# Define Unit Factors
if "kN" in unit_opt:
    UNIT_F = "kN"
    UNIT_L = "kN/m"
    UNIT_M = "kN-m"
    TO_NEWTON = 1000.0  # Input to Engine (N)
    FROM_NEWTON = 1/1000.0 # Output from Engine (kN)
else:
    UNIT_F = "kg"
    UNIT_L = "kg/m"
    UNIT_M = "kg-m"
    TO_NEWTON = 9.80665 # Input to Engine (N)
    FROM_NEWTON = 1/9.80665 # Output from Engine (kg)

# ==========================================
# 🎨 VISUALIZATION FUNCTIONS
# ==========================================
def add_peak_labels(fig, x_data, y_data, inverted=False):
    """Add labels to peaks without arrows"""
    max_idx = np.argmax(y_data)
    min_idx = np.argmin(y_data)
    
    peaks = [(x_data[max_idx], y_data[max_idx]), (x_data[min_idx], y_data[min_idx])]
    
    font_style = dict(color="black", size=11, family="Arial")
    bg_color = "rgba(255,255,255,0.7)"

    for x, y in peaks:
        # Shift text away from the line
        shift = 15 if (y >= 0 and not inverted) or (y < 0 and inverted) else -15
        fig.add_annotation(
            x=x, y=y, text=f"{y:.2f}",
            showarrow=False, yshift=shift,
            font=font_style, bgcolor=bg_color
        )

def draw_beam_diagram(spans, supports, loads):
    """Draw Beam, Dimensions, Supports, and Loads"""
    fig = go.Figure()
    
    total_len = sum(spans)
    # 1. Main Beam
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0],
        mode='lines', line=dict(color='black', width=5), hoverinfo='skip'
    ))
    
    curr_x = 0
    supp_x = 0
    
    # 2. Dimensions & Grid
    for i, L in enumerate(spans):
        # Text Dimension
        fig.add_annotation(
            x=curr_x + L/2, y=-0.6, text=f"{L} m",
            showarrow=False, font=dict(color="blue", size=12)
        )
        # Vertical Grid
        fig.add_shape(type="line", x0=curr_x+L, y0=-0.3, x1=curr_x+L, y1=0.3, line=dict(color="gray", dash="dot"))
        curr_x += L
        
    # 3. Supports
    for i, s in enumerate(supports):
        sym = "triangle-up"
        col = "green"
        if s == "Fix":
            sym = "square"
            col = "red"
        elif s == "Roller":
            sym = "circle"
            col = "orange"
            
        fig.add_trace(go.Scatter(
            x=[supp_x], y=[-0.2], mode='markers',
            marker=dict(symbol=sym, size=15, color=col),
            showlegend=False, hoverinfo='text', text=f"{s}"
        ))
        if i < len(spans): supp_x += spans[i]

    # 4. Loads
    for ld in loads:
        sx = sum(spans[:ld['span_idx']])
        val_disp = ld['display_val']
        
        if ld['type'] == 'Point':
            px = sx + ld['pos']
            fig.add_annotation(
                x=px, y=0.1, ax=0, ay=-40,
                text=f"{val_disp:.2f} {UNIT_F}",
                showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="red"
            )
        elif ld['type'] == 'Uniform':
            ex = sx + spans[ld['span_idx']]
            fig.add_shape(
                type="rect", x0=sx, y0=0, x1=ex, y1=0.3,
                fillcolor="rgba(255,0,0,0.15)", line_width=0
            )
            fig.add_annotation(
                x=(sx+ex)/2, y=0.35,
                text=f"{val_disp:.2f} {UNIT_L}",
                showarrow=False, font=dict(color="red", size=10)
            )

    fig.update_layout(
        title="Physical Model & Loading",
        height=300,
        xaxis=dict(showgrid=False, visible=True, title="Distance (m)"),
        yaxis=dict(visible=False, range=[-1, 1.5]),
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor="white"
    )
    return fig

# ==========================================
# 🖥️ MAIN UI
# ==========================================
st.title(f"🏗️ RC Beam Analysis & Design")
st.caption(f"Unit System: {unit_opt}")

# --- 1. GEOMETRY INPUT ---
with st.expander("1. Geometry & Supports", expanded=True):
    col1, col2 = st.columns([1, 3])
    with col1:
        n_span = st.number_input("จำนวนช่วงคาน (Spans)", 1, 6, 2)
    
    spans = []
    with col2:
        cols = st.columns(n_span)
        for i in range(n_span):
            spans.append(cols[i].number_input(f"L{i+1} (m)", 1.0, 20.0, 4.0))
            
    st.markdown("**Supports Condition:**")
    supports = []
    cols_s = st.columns(n_span+1)
    opts = ['Pin', 'Roller', 'Fix']
    for i in range(n_span+1):
        def_idx = 0 if i==0 else 1
        supports.append(cols_s[i].selectbox(f"Sup {i+1}", opts, index=def_idx))

# --- 2. LOADS INPUT ---
st.subheader("2. Loads Configuration (Factored: 1.4DL + 1.7LL)")
loads_input = []
cols_load = st.columns(n_span)

for i in range(n_span):
    with cols_load[i]:
        st.info(f"📍 **Span {i+1}**")
        # Uniform
        udl = st.number_input(f"UDL-DL ({UNIT_L})", 0.0, key=f"udl_{i}")
        ull = st.number_input(f"UDL-LL ({UNIT_L})", 0.0, key=f"ull_{i}")
        
        # Points
        n_pt = st.number_input(f"Point Loads (Points)", 0, 5, 0, key=f"npt_{i}")
        points = []
        for j in range(n_pt):
            p_dl = st.number_input(f"P{j+1}-DL ({UNIT_F})", 0.0, key=f"pdl_{i}_{j}")
            p_ll = st.number_input(f"P{j+1}-LL ({UNIT_F})", 0.0, key=f"pll_{i}_{j}")
            p_pos = st.number_input(f"Pos (m)", 0.0, spans[i], spans[i]/2, key=f"pp_{i}_{j}")
            points.append((p_dl, p_ll, p_pos))

        # Store Data
        if (udl + ull) > 0:
            loads_input.append({
                'span_idx': i, 'type': 'Uniform',
                'total_w': (1.4*udl + 1.7*ull) * TO_NEWTON, # Convert to N/m
                'display_val': (udl + ull)
            })
        for (pd, pl, pp) in points:
            if (pd + pl) > 0:
                loads_input.append({
                    'span_idx': i, 'type': 'Point',
                    'total_w': (1.4*pd + 1.7*pl) * TO_NEWTON, # Convert to N
                    'pos': pp,
                    'display_val': (pd + pl)
                })

# --- ANALYSIS ACTION ---
if st.button("🚀 Calculate Analysis", type="primary", use_container_width=True):
    solver = SimpleBeamSolver(spans, supports, loads_input)
    u, err = solver.solve()
    
    if err:
        st.error(f"Error: {err}")
    else:
        # Get Results & Convert Units
        df = solver.get_internal_forces(num_points=100)
        df['shear_disp'] = df['shear'] * FROM_NEWTON
        df['moment_disp'] = df['moment'] * FROM_NEWTON
        
        st.session_state['analysis_done'] = True
        st.session_state['df_res'] = df
        st.session_state['fig_beam'] = draw_beam_diagram(spans, supports, loads_input)

# ==========================================
# 📊 RESULTS SECTION
# ==========================================
if st.session_state.get('analysis_done', False):
    df = st.session_state['df_res']
    
    # 1. Visualization
    st.plotly_chart(st.session_state['fig_beam'], use_container_width=True)
    
    # 2. Key Values
    m_max = df['moment_disp'].max()
    m_min = df['moment_disp'].min()
    v_max = df['shear_disp'].abs().max()
    
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Max Moment (+)", f"{m_max:.2f} {UNIT_M}")
    c2.metric(f"Max Moment (-)", f"{m_min:.2f} {UNIT_M}")
    c3.metric(f"Max Shear (Abs)", f"{v_max:.2f} {UNIT_F}")
    
    # 3. Graphs
    col_g1, col_g2 = st.columns(2)
    
    # SFD
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=df['x'], y=df['shear_disp'], fill='tozeroy', line=dict(color='#D32F2F')))
    add_peak_labels(fig_v, df['x'].values, df['shear_disp'].values)
    fig_v.update_layout(title=f"Shear Force Diagram (SFD) [{UNIT_F}]", hovermode="x")
    col_g1.plotly_chart(fig_v, use_container_width=True)
    
    # BMD
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(x=df['x'], y=df['moment_disp'], fill='tozeroy', line=dict(color='#1976D2')))
    add_peak_labels(fig_m, df['x'].values, df['moment_disp'].values, inverted=True)
    fig_m.update_layout(title=f"Bending Moment Diagram (BMD) [{UNIT_M}]", yaxis=dict(autorange="reversed"))
    col_g2.plotly_chart(fig_m, use_container_width=True)

    # ==========================================
    # 🛠️ INTERACTIVE DESIGN (DETAILED)
    # ==========================================
    st.markdown("---")
    st.header("🛠️ Design & Calculation")
    
    # -- Design Inputs --
    with st.form("design_panel"):
        c_mat, c_sect, c_bar = st.columns(3)
        with c_mat:
            st.markdown("##### Material")
            fc = st.number_input("f'c (ksc)", value=240.0)
            fy = st.number_input("fy (ksc)", value=4000.0)
        with c_sect:
            st.markdown("##### Section")
            b_val = st.number_input("Width b (cm)", 15.0, 100.0, 25.0)
            h_val = st.number_input("Depth h (cm)", 20.0, 200.0, 50.0)
            cov_val = st.number_input("Covering (cm)", 2.0, 5.0, 3.0)
        with c_bar:
            st.markdown("##### Reinforcement")
            bar_opts = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
            main_bar = st.selectbox("Main Bar", list(bar_opts.keys()), index=1)
            
        submitted = st.form_submit_button("🔄 Recalculate Design")
        
    # -- Calculation Logic --
    # Convert Material to MPa (Standard Calculation)
    fc_mpa = fc * 0.0980665
    fy_mpa = fy * 0.0980665
    b_mm = b_val * 10
    d_mm = (h_val - cov_val) * 10
    
    # Design Moment (Mu) -> Needs to be in N-mm for formula
    # If display is kN-m -> * 1e6
    # If display is kg-m -> * 9.80665 * 1000
    Mu_disp = max(abs(m_max), abs(m_min))
    if "kN" in unit_opt:
        Mu_Nmm = Mu_disp * 1e6
    else:
        Mu_Nmm = Mu_disp * 9.80665 * 1000
        
    # --- 1. Flexural Design ---
    st.subheader("1. Flexural Design (Moment)")
    col_flex_res, col_flex_cal = st.columns([1, 1.5])
    
    with col_flex_res:
        phi_b = 0.9
        Rn = (Mu_Nmm / phi_b) / (b_mm * d_mm**2) # MPa
        m_rat = fy_mpa / (0.85 * fc_mpa)
        
        term = 1 - (2 * m_rat * Rn) / fy_mpa
        
        if term < 0:
            st.error("❌ **FAIL**: Section too small")
            st.write(f"Rn required ({Rn:.2f}) exceeds limit.")
        else:
            rho_req = (1/m_rat)*(1 - np.sqrt(term))
            rho_min = max(np.sqrt(fc_mpa)/(4*fy_mpa), 1.4/fy_mpa)
            rho_des = max(rho_req, rho_min)
            
            As_req = rho_des * b_mm * d_mm # mm2
            As_cm2 = As_req / 100
            
            # Bar Selection
            bar_area = bar_opts[main_bar]
            num_bars = max(2, int(np.ceil(As_cm2 / bar_area)))
            real_As = num_bars * bar_area
            
            st.success(f"✅ **PASS** | Use {num_bars} - {main_bar}")
            st.metric("Required As", f"{As_cm2:.2f} cm²")
            st.metric(f"Provided As ({num_bars}-{main_bar})", f"{real_As:.2f} cm²")

    with col_flex_cal:
        with st.expander("Show Calculation Sheet (Flexure)", expanded=True):
            st.latex(f"M_u = {Mu_disp:.2f} \\text{{ {UNIT_M}}}")
            st.latex(f"R_n = \\frac{{M_u}}{{\\phi b d^2}} = {Rn:.2f} \\text{{ MPa}}")
            st.latex(f"\\rho_{{req}} = {rho_req:.5f}, \\quad \\rho_{{min}} = {rho_min:.5f}")
            st.latex(f"A_s = \\rho \\cdot b \\cdot d = {As_cm2:.2f} \\text{{ cm}}^2")

    # --- 2. Shear Design ---
    st.markdown("---")
    st.subheader("2. Shear Design (Stirrups)")
    col_shear_res, col_shear_cal = st.columns([1, 1.5])
    
    # Prepare Vc
    # Vc = 0.17 * sqrt(fc') * b * d (in N)
    Vc_N = 0.17 * np.sqrt(fc_mpa) * b_mm * d_mm
    phi_v = 0.85
    phiVc_N = phi_v * Vc_N
    
    # Vu
    Vu_disp = v_max
    if "kN" in unit_opt:
        Vu_N = Vu_disp * 1000
    else:
        Vu_N = Vu_disp * 9.80665
    
    # Convert Vc to Display Unit
    phiVc_disp = phiVc_N * FROM_NEWTON

    with col_shear_res:
        st.metric("Max Shear (Vu)", f"{Vu_disp:.2f} {UNIT_F}")
        st.metric("Concrete Capacity (phi Vc)", f"{phiVc_disp:.2f} {UNIT_F}")
        
        status_shear = ""
        stirrup_txt = ""
        
        if Vu_N <= phiVc_N / 2:
            st.info("✅ No Stirrups Required (Theoretically)")
            stirrup_txt = "Min Stirrups Recommendation: RB6 @ 20 cm"
        elif Vu_N <= phiVc_N:
            st.warning("⚠️ Min Stirrups Required")
            stirrup_txt = "Use RB6 @ 20 cm (Min Spec)"
        else:
            st.error("❗ Stirrups Required (Calculation needed)")
            Vs_N = (Vu_N - phiVc_N) / phi_v
            # Assume RB6 (2 legs) -> Av = 56 mm2
            Av = 2 * 28.27
            # s = (Av * fy * d) / Vs
            s_req_mm = (Av * fy_mpa * d_mm) / Vs_N
            
            max_s = d_mm / 2
            s_final = min(s_req_mm, max_s, 300) # Cap at 30cm
            
            stirrup_txt = f"Use RB6 @ {int(s_final/10)} cm"
            
        st.write(f"**Result:** {stirrup_txt}")

    with col_shear_cal:
         with st.expander("Show Calculation Sheet (Shear)", expanded=True):
            st.latex(f"V_u = {Vu_disp:.2f} \\text{{ {UNIT_F}}}")
            st.latex(f"\\phi V_c = 0.85 \\times 0.17\\sqrt{{f'_c}} b d = {phiVc_disp:.2f} \\text{{ {UNIT_F}}}")
            
            if Vu_N > phiVc_N:
                Vs_disp = (Vs_N * FROM_NEWTON)
                st.latex(f"V_s = \\frac{{V_u - \\phi V_c}}{{\\phi}} = {Vs_disp:.2f} \\text{{ {UNIT_F}}}")
                st.latex(f"s = \\frac{{A_v f_y d}}{{V_s}} \\text{{ (Use RB6)}}")
            else:
                st.write("$V_u < \\phi V_c$: Use Minimum Reinforcement")
