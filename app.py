import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- IMPORT CUSTOM ENGINE ---
try:
    from beam_engine import SimpleBeamSolver
except ImportError:
    st.error("⚠️ ไม่พบไฟล์ 'beam_engine.py' กรุณาวางไฟล์ engine ไว้ในโฟลเดอร์เดียวกัน")
    st.stop()

# ==========================================
# 🎨 UI CONFIG & HELPER FUNCTIONS
# ==========================================
st.set_page_config(page_title="Beam Pro V8", layout="wide")

def add_peak_labels(fig, x_data, y_data, inverted=False):
    """Helper to add Max/Min labels WITHOUT arrows"""
    # Find Max Positive
    max_idx = np.argmax(y_data)
    max_val = y_data[max_idx]
    max_x = x_data[max_idx]
    
    # Find Max Negative
    min_idx = np.argmin(y_data)
    min_val = y_data[min_idx]
    min_x = x_data[min_idx]

    font_style = dict(color="black", size=12, family="Arial")
    bg_color = "rgba(255,255,255,0.8)"

    # Label for Max
    fig.add_annotation(
        x=max_x, y=max_val, text=f"{max_val:.2f}",
        showarrow=False,  # <--- เอาลูกศรออกตามขอ
        yshift=15 if not inverted else -15,
        font=font_style, bgcolor=bg_color
    )
    # Label for Min
    fig.add_annotation(
        x=min_x, y=min_val, text=f"{min_val:.2f}",
        showarrow=False, # <--- เอาลูกศรออกตามขอ
        yshift=-15 if not inverted else 15,
        font=font_style, bgcolor=bg_color
    )

def draw_beam_diagram(spans, supports, loads):
    """ฟังก์ชันวาดรูปคานจำลอง (FBD) พร้อม Loads และ Dimensions"""
    fig = go.Figure()
    
    current_x = 0
    max_load_h = 1.0 # Scale factor for visualization
    
    # 1. Draw Beam Line
    total_len = sum(spans)
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0],
        mode='lines', line=dict(color='black', width=5),
        name='Beam'
    ))
    
    # 2. Draw Supports & Dimensions
    current_x = 0
    for i, L in enumerate(spans):
        # Dimension Line
        mid_x = current_x + L/2
        fig.add_annotation(
            x=mid_x, y=-0.5, text=f"{L} m",
            showarrow=False, font=dict(size=12, color="blue")
        )
        # Vertical grid lines for spans
        fig.add_shape(type="line", x0=current_x+L, y0=-0.2, x1=current_x+L, y1=0.2, 
                      line=dict(color="gray", dash="dot"))
        
        current_x += L

    # Draw Supports Markers
    supp_x = 0
    for i, s_type in enumerate(supports):
        symbol = "triangle-up"
        color = "green"
        if s_type == 'Fix': 
            symbol = "square"
            color = "red"
        elif s_type == 'Roller': 
            symbol = "circle"
            color = "orange"
            
        fig.add_trace(go.Scatter(
            x=[supp_x], y=[-0.15], # Offset slightly below beam
            mode='markers+text',
            marker=dict(symbol=symbol, size=15, color=color),
            text=[s_type], textposition="bottom center",
            showlegend=False
        ))
        
        if i < len(spans):
            supp_x += spans[i]

    # 3. Draw Loads
    for load in loads:
        span_idx = load['span_idx']
        start_x_span = sum(spans[:span_idx])
        
        if load['type'] == 'Point':
            pos_x = start_x_span + load['pos']
            w_val = load['total_w'] / 1000.0 # to kN
            
            # Arrow
            fig.add_annotation(
                x=pos_x, y=0,
                ax=0, ay=-40, # Vector length
                text=f"P={w_val:.1f} kN",
                showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="red"
            )
            
        elif load['type'] == 'Uniform':
            w_val = load['total_w'] / 1000.0 # to kN/m
            end_x_span = start_x_span + spans[span_idx]
            
            # Draw Rectangle representing Uniform Load
            fig.add_shape(
                type="rect",
                x0=start_x_span, y0=0, x1=end_x_span, y1=0.3,
                fillcolor="rgba(255, 0, 0, 0.2)", line=dict(width=0),
            )
            fig.add_annotation(
                x=(start_x_span+end_x_span)/2, y=0.35,
                text=f"w={w_val:.1f} kN/m", showarrow=False, font=dict(color="red")
            )

    fig.update_layout(
        title="Physical Beam Model & Loading",
        height=300,
        yaxis=dict(visible=False, range=[-1, 1.5]),
        xaxis=dict(title="Distance (m)", showgrid=False),
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ==========================================
# 🖥️ MAIN APPLICATION
# ==========================================
st.title("🏗️ RC Beam Analysis & Design (Pro V8)")

# --- INPUT SECTION ---
with st.sidebar:
    st.header("1. Geometry & Supports")
    n_span = st.number_input("จำนวนช่วงคาน (Spans)", 1, 6, 2)
    
    spans = []
    for i in range(n_span):
        spans.append(st.number_input(f"ความยาวช่วงที่ {i+1} (m)", 1.0, 20.0, 4.0))
        
    st.divider()
    supports = []
    opts = ['Pin', 'Roller', 'Fix']
    for i in range(n_span+1):
        def_idx = 0 if i==0 else 1
        supports.append(st.selectbox(f"จุดรองรับที่ {i+1}", opts, index=def_idx))

    st.header("2. Material Properties")
    fc = st.number_input("f'c (ksc)", value=240.0) # Input as ksc usually in TH
    fy = st.number_input("fy (ksc)", value=4000.0) # RB=2400, DB=4000
    b = st.number_input("ความกว้าง b (cm)", value=25.0)
    h = st.number_input("ความลึก h (cm)", value=50.0)
    covering = st.number_input("Covering (cm)", value=4.0)

# --- LOAD INPUT SECTION (MAIN AREA) ---
st.subheader("3. Loads Configuration (Factored: 1.4DL + 1.7LL)")
loads_input = []
cols = st.columns(n_span)

for i in range(n_span):
    with cols[i]:
        st.info(f"📍 **Span {i+1}**")
        
        # Uniform
        u_dl = st.number_input(f"DL (kN/m)", 0.0, key=f"udl_{i}")
        u_ll = st.number_input(f"LL (kN/m)", 0.0, key=f"ull_{i}")
        if (u_dl + u_ll) > 0:
            loads_input.append({
                'span_idx': i, 'type': 'Uniform', 
                'total_w': (1.4*u_dl + 1.7*u_ll)*1000
            })
            
        # Point Loads
        st.markdown("---")
        num_pt = st.number_input(f"Point Loads (จุด)", 0, 5, 0, key=f"npt_{i}")
        for j in range(num_pt):
            c1, c2 = st.columns(2)
            p_dl = c1.number_input(f"P{j+1} DL", 0.0, key=f"pdl_{i}_{j}")
            p_ll = c1.number_input(f"P{j+1} LL", 0.0, key=f"pll_{i}_{j}")
            p_pos = c2.number_input(f"Pos (m)", 0.0, spans[i], spans[i]/2, key=f"ppos_{i}_{j}")
            
            if (p_dl + p_ll) > 0:
                loads_input.append({
                    'span_idx': i, 'type': 'Point',
                    'total_w': (1.4*p_dl + 1.7*p_ll)*1000,
                    'pos': p_pos
                })

# --- CALCULATION ---
if st.button("🚀 Calculate & Design", type="primary", use_container_width=True):
    
    # 1. Analyze
    solver = SimpleBeamSolver(spans, supports, loads_input)
    u, err = solver.solve()
    
    if err:
        st.error(f"Analysis Error: {err}")
    else:
        # Get Data
        df = solver.get_internal_forces(num_points=100)
        m_max = df['moment'].max()
        m_min = df['moment'].min()
        v_max = df['shear'].abs().max()

        # ==========================
        # 1. VISUALIZATION
        # ==========================
        st.markdown("### 🛠️ Model Visualization")
        fig_beam = draw_beam_diagram(spans, supports, loads_input)
        st.plotly_chart(fig_beam, use_container_width=True)

        # ==========================
        # 2. ANALYSIS RESULTS
        # ==========================
        st.markdown("### 📊 Analysis Diagrams")
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Moment (+)", f"{m_max:.2f} kN-m")
        c2.metric("Max Moment (-)", f"{m_min:.2f} kN-m")
        c3.metric("Max Shear", f"{v_max:.2f} kN")
        
        # SFD
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D32F2F'), name='Shear'))
        add_peak_labels(fig_v, df['x'].values, df['shear'].values)
        fig_v.update_layout(title="Shear Force Diagram (SFD) [kN]", hovermode="x")
        st.plotly_chart(fig_v, use_container_width=True)
        
        # BMD
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1976D2'), name='Moment'))
        add_peak_labels(fig_m, df['x'].values, df['moment'].values, inverted=True)
        fig_m.update_layout(
            title="Bending Moment Diagram (BMD) [kN-m] (Tension Side)", 
            yaxis=dict(autorange="reversed"), hovermode="x"
        )
        st.plotly_chart(fig_m, use_container_width=True)

        # ==========================
        # 3. DETAILED DESIGN CALC
        # ==========================
        st.markdown("---")
        st.header("📝 Design Calculation (SDM)")
        
        # แปลงหน่วย input (ksc -> MPa สำหรับคำนวณในสูตรมาตรฐาน หรือแปลงกลับตอนแสดงผล)
        # ปกติสูตร ACI Metric ใช้ MPa แต่เพื่อให้เข้าใจง่ายกับ input ksc เราจะแปลงค่าให้ถูกต้อง
        # 1 ksc ≈ 0.098 MPa, 1 MPa ≈ 10.2 ksc
        # เพื่อความง่ายในการแสดงผล จะใช้หน่วย SI (MPa) ในการคำนวณ แล้วเทียบเคียง
        
        fc_mpa = fc * 0.0980665
        fy_mpa = fy * 0.0980665
        d = h - covering
        
        # Design Moment (Max absolute)
        Mu = max(abs(m_max), abs(m_min))
        
        st.markdown(f"#### 1. Design Parameters")
        st.latex(f"f'_c = {fc:.0f} \\text{{ ksc}} (\\approx {fc_mpa:.2f} \\text{{ MPa}})")
        st.latex(f"f_y = {fy:.0f} \\text{{ ksc}} (\\approx {fy_mpa:.2f} \\text{{ MPa}})")
        st.latex(f"b = {b:.0f} \\text{{ cm}}, \\quad h = {h:.0f} \\text{{ cm}}, \\quad d = {d:.0f} \\text{{ cm}}")
        
        st.markdown(f"#### 2. Flexural Design (Strength Design Method)")
        st.write("ตรวจสอบค่าโมเมนต์ดัดสูงสุด (Max Factored Moment):")
        st.latex(f"M_u = {Mu:.2f} \\text{{ kN-m}}")
        
        phi_b = 0.90
        Mn_req = Mu / phi_b
        st.write(f"ต้องการกำลังระบุ (Nominal Moment Strength), $\\phi = {phi_b}$:")
        st.latex(f"M_n = \\frac{{M_u}}{{\\phi}} = \\frac{{{Mu:.2f}}}{{{phi_b}}} = {Mn_req:.2f} \\text{{ kN-m}}")
        
        # Check Rn
        # Rn = Mn / (b * d^2)
        # Mn in N-m, b, d in m
        Mn_Nm = Mn_req * 1000 * 1000 # to N-mm
        b_mm = b * 10
        d_mm = d * 10
        
        Rn = Mn_Nm / (b_mm * d_mm**2) # MPa
        
        st.write("คำนวณค่า $R_n$ (Coefficient of Resistance):")
        st.latex(f"R_n = \\frac{{M_n \\times 10^6}}{{b \\cdot d^2}} = \\frac{{{Mn_req:.2f} \\times 10^6}}{{{b_mm:.0f} \\cdot {d_mm:.0f}^2}} = {Rn:.4f} \\text{{ MPa}}")
        
        # Calculate Rho
        m_ratio = fy_mpa / (0.85 * fc_mpa)
        st.write("คำนวณสัดส่วนเหล็กเสริม (Reinforcement Ratio):")
        st.latex(f"m = \\frac{{f_y}}{{0.85 f'_c}} = \\frac{{{fy_mpa:.2f}}}{{0.85 \\times {fc_mpa:.2f}}} = {m_ratio:.2f}")
        
        term = 1 - (2 * m_ratio * Rn) / fy_mpa
        
        if term < 0:
            st.error(f"❌ **Section Fail!** หน้าตัดเล็กเกินไป รับแรงอัดไม่ไหว ($2mR_n/f_y > 1$)")
            st.stop()
            
        rho = (1 / m_ratio) * (1 - np.sqrt(term))
        st.latex(f"\\rho = \\frac{{1}}{{m}} \\left( 1 - \\sqrt{{1 - \\frac{{2 m R_n}}{{f_y}}}} \\right)")
        st.latex(f"\\rho = \\frac{{1}}{{{m_ratio:.2f}}} \\left( 1 - \\sqrt{{1 - \\frac{{2({m_ratio:.2f})({Rn:.4f})}}{{{fy_mpa:.2f}}}}} \\right) = {rho:.5f}")
        
        # Check Min Steel
        # As_min equations for MPa: max( sqrt(fc')/4fy, 1.4/fy ) * b*d
        rho_min1 = np.sqrt(fc_mpa) / (4 * fy_mpa)
        rho_min2 = 1.4 / fy_mpa
        rho_min = max(rho_min1, rho_min2)
        
        rho_final = max(rho, rho_min)
        is_min_gov = rho < rho_min
        
        st.write("ตรวจสอบ $\\rho_{min}$:")
        st.latex(f"\\rho_{{min}} = \\max \\left( \\frac{{\\sqrt{{f'_c}}}}{{4 f_y}}, \\frac{{1.4}}{{f_y}} \\right) = {rho_min:.5f}")
        
        remark = "(Use $\\rho_{min}$)" if is_min_gov else "(Use Calculated $\\rho$)"
        st.write(f"**Selected $\\rho$:** {rho_final:.5f} {remark}")
        
        # Calculate As
        As_req = rho_final * b * d # cm2
        st.markdown(f"#### 3. Required Steel Area ($A_s$)")
        st.latex(f"A_s = \\rho \\cdot b \\cdot d = {rho_final:.5f} \\times {b:.0f} \\times {d:.0f} = \\mathbf{{{As_req:.2f} \\text{{ cm}}^2}}")

        # Shear Design Brief
        st.markdown("---")
        st.markdown(f"#### 4. Shear Check (Brief)")
        # Vc = 0.53 * sqrt(fc_ksc) * b * d  (Simplified ACI metric in kg, convert to kN)
        # Using SI: Vc = 0.17 * sqrt(fc_mpa) * b_mm * d_mm / 1000 (kN)
        
        Vc = 0.17 * np.sqrt(fc_mpa) * b_mm * d_mm / 1000.0 # kN
        phi_v = 0.85
        phiVc = phi_v * Vc
        
        st.latex(f"V_u = {v_max:.2f} \\text{{ kN}}")
        st.latex(f"\\phi V_c = 0.85 \\times 0.17\\sqrt{{{fc_mpa:.2f}}} \\times {b_mm} \\times {d_mm} / 1000 = {phiVc:.2f} \\text{{ kN}}")
        
        if v_max <= phiVc / 2:
            st.success(f"✅ Concrete Capacity OK ($V_u < 0.5\\phi V_c$). No stirrups required theoretically.")
        elif v_max <= phiVc:
            st.warning(f"⚠️ Minimum Stirrups Required ($0.5\\phi V_c < V_u < \\phi V_c$).")
        else:
            Vs_req = (v_max - phiVc) / phi_v
            st.error(f"❗ **Design Stirrups Required!** ($V_u > \\phi V_c$)")
            st.latex(f"V_s = \\frac{{V_u - \\phi V_c}}{{\\phi}} = \\frac{{{v_max:.2f} - {phiVc:.2f}}}{{0.85}} = {Vs_req:.2f} \\text{{ kN}}")
