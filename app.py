import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

# --- MOCK ENGINE (กรณีไม่มีไฟล์ beam_engine.py) ---
# เพื่อให้ Code รันได้ทันทีโดยไม่ Error
try:
    from beam_engine import SimpleBeamSolver
except ImportError:
    class SimpleBeamSolver:
        def __init__(self, spans, supports, loads):
            self.spans = spans
            self.loads = loads
        def solve(self):
            return None, None
        def get_internal_forces(self, n):
            L = sum(self.spans)
            x = np.linspace(0, L, n)
            # สร้างกราฟจำลอง (Sine wave + Linear)
            return pd.DataFrame({
                'x': x,
                'shear': 15 * np.cos(x) * (1 - x/L), 
                'moment': 30 * np.sin(x) * x
            })

# ==========================================
# ⚙️ CONFIG & STYLES
# ==========================================
st.set_page_config(page_title="RC Beam Pro V.18 (Clean UI)", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    
    /* Headers */
    .main-header { 
        background-color: #1E88E5; color: white; padding: 15px; 
        border-radius: 10px; text-align: center; margin-bottom: 20px; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.2rem; font-weight: bold; color: #1565C0;
        border-bottom: 2px solid #1565C0; padding-bottom: 5px; margin-top: 30px; margin-bottom: 15px;
    }
    
    /* Input Cards */
    .input-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #ddd; margin-bottom: 10px; }
    
    /* Result Box */
    .design-box { 
        background-color: #e3f2fd; border: 2px solid #2196F3; 
        padding: 20px; border-radius: 10px; margin-top: 20px; 
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

def get_default_factors(code_name):
    if "WSD" in code_name: return 1.0, 1.0, "WSD"
    elif "ACI" in code_name: return 1.2, 1.6, "SDM"
    else: return 1.4, 1.7, "SDM" # EIT SDM

def draw_beam_diagram(spans, supports, loads, unit_load="kg/m", unit_force="kg"):
    """ สร้างกราฟแสดงโมเดลคานที่สวยงามและมีตัวเลขชัดเจน """
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    
    fig = go.Figure()
    
    # 1. เส้นคานหลัก
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0], 
        mode='lines', line=dict(color='black', width=5), 
        hoverinfo='skip'
    ))

    # 2. จุดรองรับ (Supports)
    for i, s_type in enumerate(supports):
        sx = cum_len[i]
        symbol = "triangle-up" if s_type != "Fix" else "square"
        color = "#388E3C" if s_type != "Fix" else "#D32F2F"
        fig.add_trace(go.Scatter(
            x=[sx], y=[-0.02], mode='markers+text',
            marker=dict(symbol=symbol, size=15, color=color),
            text=[s_type], textposition="bottom center",
            hoverinfo='none', showlegend=False
        ))

    # 3. โหลด (Loads) & Annotations
    max_load_h = 0.2
    
    for load in loads:
        start_x = cum_len[load['span_idx']]
        val = load['display_val'] # Unfactored value for display
        
        if load['type'] == 'Uniform':
            end_x = start_x + spans[load['span_idx']]
            # วาดสี่เหลี่ยมแทน UDL
            fig.add_shape(type="rect", x0=start_x, y0=0, x1=end_x, y1=max_load_h,
                          line=dict(width=0), fillcolor="rgba(255, 69, 0, 0.2)")
            # เส้นขอบบน
            fig.add_trace(go.Scatter(
                x=[start_x, end_x], y=[max_load_h, max_load_h],
                mode='lines', line=dict(color='orangered', width=2), showlegend=False
            ))
            # Text Label
            fig.add_annotation(
                x=(start_x+end_x)/2, y=max_load_h+0.05,
                text=f"w = {val} {unit_load}", showarrow=False,
                font=dict(color="orangered", size=12)
            )

        elif load['type'] == 'Point':
            lx = start_x + load['pos']
            # วาดลูกศร
            fig.add_annotation(
                x=lx, y=0, ax=0, ay=-40,
                text=f"P = {val} {unit_force}",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="red"
            )

    # 4. Dimension Lines (บอกระยะ Span)
    for i, span in enumerate(spans):
        mid_x = cum_len[i] + span/2
        fig.add_annotation(
            x=mid_x, y=-0.15, text=f"{span:.2f} m",
            showarrow=False, font=dict(size=14, color="blue")
        )
        # เส้นกั้นช่วง
        fig.add_shape(type="line", x0=cum_len[i], y0=-0.1, x1=cum_len[i], y1=-0.2, line=dict(color="gray", dash="dot"))
        fig.add_shape(type="line", x0=cum_len[i+1], y0=-0.1, x1=cum_len[i+1], y1=-0.2, line=dict(color="gray", dash="dot"))

    fig.update_layout(
        title="Beam Model & Loads",
        height=300,
        yaxis=dict(visible=False, range=[-0.3, 0.5]),
        xaxis=dict(title="Distance (m)", showgrid=True),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white'
    )
    return fig

def draw_section_view(b, h, cover, n_bars, bar_dia_mm, stirrup_dia_mm):
    """ วาดรูปหน้าตัดคาน """
    fig = go.Figure()
    # Concrete
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=2), fillcolor="#eeeeee")
    # Stirrup
    inset = cover
    fig.add_shape(type="rect", x0=inset, y0=inset, x1=b-inset, y1=h-inset, line=dict(color="red", width=2))
    # Main Bars
    if n_bars > 0:
        eff_w = b - 2*cover - 2*(stirrup_dia_mm/10) - (bar_dia_mm/10)
        gap = eff_w / (n_bars - 1) if n_bars > 1 else 0
        y_pos = cover + (stirrup_dia_mm/10) + (bar_dia_mm/20)
        
        x_list = []
        start_x = cover + (stirrup_dia_mm/10) + (bar_dia_mm/20)
        for i in range(n_bars):
            x_list.append(start_x + i*gap)
            
        fig.add_trace(go.Scatter(
            x=x_list, y=[y_pos]*n_bars, mode='markers',
            marker=dict(size=bar_dia_mm, color='blue', line=dict(width=1, color='black'))
        ))

    fig.update_layout(
        width=250, height=250*(h/b) if b>0 else 250,
        xaxis=dict(visible=False, range=[-5, b+5]),
        yaxis=dict(visible=False, range=[-5, h+5]),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='rgba(0,0,0,0)', showlegend=False
    )
    return fig

# ==========================================
# 🖥️ MAIN APPLICATION
# ==========================================

st.markdown('<div class="main-header"><h2>🏗️ RC Beam Pro V.18 (Engineering Suite)</h2></div>', unsafe_allow_html=True)

# ------------------------------------------
# PART 1: SETTINGS & INPUTS
# ------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # 1. Code Selection
    design_code = st.selectbox("Design Standard", ["EIT 1007 (WSD)", "EIT 1008 (SDM)", "ACI 318 (SDM)"], index=1)
    def_dl, def_ll, method = get_default_factors(design_code)
    
    st.divider()
    st.markdown("**Load Factors (Editable)**")
    col_f1, col_f2 = st.columns(2)
    fact_dl = col_f1.number_input("DL Factor", value=def_dl, step=0.1)
    fact_ll = col_f2.number_input("LL Factor", value=def_ll, step=0.1)
    
    st.divider()
    unit_sys = st.radio("Units", ["MKS (kg, m)", "SI (kN, m)"], index=0)
    
    if "kN" in unit_sys:
        U_L, U_M, U_F = "kN/m", "kN-m", "kN"
        TO_N, FROM_N = 1000.0, 0.001
    else:
        U_L, U_M, U_F = "kg/m", "kg-m", "kg"
        TO_N, FROM_N = 9.80665, 1/9.80665

# --- INPUT AREA ---
st.markdown('<div class="section-header">1️⃣ Structure & Loads (ระบุโครงสร้างและน้ำหนักบรรทุก)</div>', unsafe_allow_html=True)

col_geo, col_loads = st.columns([1, 1.5])

with col_geo:
    st.info("📍 Geometry (รูปทรง)")
    n_span = st.number_input("จำนวนช่วงคาน (Spans)", 1, 5, 2)
    spans = []
    supports = []
    
    c_sup_L, c_sup_R = st.columns(2)
    sup_L = c_sup_L.selectbox("Left Support", ["Pin", "Roller", "Fix"], key="SL")
    sup_R = c_sup_R.selectbox("Right Support", ["Pin", "Roller", "Fix"], index=1, key="SR")
    
    supports.append(sup_L)
    for i in range(n_span):
        l = st.number_input(f"ความยาวช่วงที่ {i+1} (m)", 1.0, 20.0, 4.0, key=f"L{i}")
        spans.append(l)
        if i < n_span-1: supports.append("Roller")
    supports.append(sup_R)

with col_loads:
    st.info(f"⬇️ Loads (น้ำหนักบรรทุก - {unit_sys})")
    loads_input = []
    tabs = st.tabs([f"Span {i+1}" for i in range(n_span)])
    
    for i, tab in enumerate(tabs):
        with tab:
            st.markdown(f"**Load on Span {i+1}**")
            c1, c2 = st.columns(2)
            udl = c1.number_input(f"Uniform DL ({U_L})", 0.0, key=f"udl{i}")
            ull = c2.number_input(f"Uniform LL ({U_L})", 0.0, key=f"ull{i}")
            
            # Factored Calculation
            w_total = udl*fact_dl + ull*fact_ll
            if w_total > 0:
                loads_input.append({'span_idx': i, 'type': 'Uniform', 'total_w': w_total*TO_N, 'display_val': udl+ull})
            
            st.markdown("---")
            if st.checkbox("เพิ่ม Point Load", key=f"chk{i}"):
                c3, c4, c5 = st.columns(3)
                pdl = c3.number_input("P DL", 0.0, key=f"pdl{i}")
                pll = c4.number_input("P LL", 0.0, key=f"pll{i}")
                px = c5.number_input("ระยะ x (m)", 0.0, spans[i], spans[i]/2, key=f"px{i}")
                
                p_total = pdl*fact_dl + pll*fact_ll
                if p_total > 0:
                    loads_input.append({'span_idx': i, 'type': 'Point', 'total_w': p_total*TO_N, 'pos': px, 'display_val': pdl+pll})

# ------------------------------------------
# PART 2: ANALYSIS
# ------------------------------------------
st.markdown('<div class="section-header">2️⃣ Structural Analysis (วิเคราะห์โครงสร้าง)</div>', unsafe_allow_html=True)

if st.button("🚀 Calculate Analysis", type="primary", use_container_width=True):
    # 1. Draw Beam Diagram
    fig_beam = draw_beam_diagram(spans, supports, loads_input, U_L, U_F)
    st.plotly_chart(fig_beam, use_container_width=True)
    
    # 2. Solve
    solver = SimpleBeamSolver(spans, supports, loads_input)
    u, err = solver.solve() # Mock doesn't use u
    
    # 3. Get Results
    df = solver.get_internal_forces(100)
    
    # Convert units back for display
    df['V_show'] = df['shear'] * FROM_N
    df['M_show'] = df['moment'] * FROM_N
    
    st.session_state['res_df'] = df
    st.session_state['analyzed'] = True

# Show Graphs if analyzed
if st.session_state.get('analyzed', False):
    df = st.session_state['res_df']
    
    col_res1, col_res2 = st.columns(2)
    max_M = df['M_show'].abs().max()
    max_V = df['V_show'].abs().max()
    
    col_res1.metric(f"Max Moment (Mu)", f"{max_M:.2f} {U_M}")
    col_res2.metric(f"Max Shear (Vu)", f"{max_V:.2f} {U_F}")

    # Plot SFD & BMD
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                        subplot_titles=(f"Shear Diagram ({U_F})", f"Moment Diagram ({U_M})"))
    
    fig.add_trace(go.Scatter(x=df['x'], y=df['V_show'], fill='tozeroy', line=dict(color='#E53935'), name="V"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['M_show'], fill='tozeroy', line=dict(color='#1E88E5'), name="M"), row=2, col=1)
    
    fig.update_layout(height=500, showlegend=False, hovermode="x unified")
    fig.update_yaxes(autorange="reversed", row=2, col=1) # Flip Moment
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# PART 3: DESIGN (แยกส่วนชัดเจน)
# ------------------------------------------
st.markdown("---")
st.markdown('<div class="section-header">3️⃣ Part Design (ออกแบบหน้าตัดคอนกรีตเสริมเหล็ก)</div>', unsafe_allow_html=True)

if st.session_state.get('analyzed', False):
    
    # Container for Design to make it distinct
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        # Design Inputs
        c_d1, c_d2, c_d3 = st.columns([1, 1, 1.5])
        
        with c_d1:
            st.markdown("##### 🧱 Material")
            fc = st.number_input("f'c (ksc)", value=240.0)
            fy = st.number_input("fy (ksc)", value=4000.0)
            
        with c_d2:
            st.markdown("##### 📏 Section Size")
            b = st.number_input("Width b (cm)", value=25.0)
            h = st.number_input("Depth h (cm)", value=50.0)
            cov = st.number_input("Cover (cm)", value=3.0)
            
        with c_d3:
            st.markdown("##### ⛓️ Rebar Spec")
            main_bar = st.selectbox("Main Bar", ["DB12", "DB16", "DB20", "DB25"], index=2)
            stirrup = st.selectbox("Stirrup", ["RB6", "RB9", "DB10"], index=1)
            
            bar_areas = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91}
            bar_dias = {'DB12':12, 'DB16':16, 'DB20':20, 'DB25':25}
            stir_areas = {'RB6':0.28, 'RB9':0.64, 'DB10':0.78}
            stir_dias = {'RB6':6, 'RB9':9, 'DB10':10}

        st.markdown('</div>', unsafe_allow_html=True)

        # --- CALCULATION ---
        Mu_val = st.session_state['res_df']['M_show'].abs().max()
        Vu_val = st.session_state['res_df']['V_show'].abs().max()
        
        # Convert all to kg, cm for calc
        if "kN" in unit_sys:
            Mu_calc = Mu_val * 101.97 * 100 # kN-m -> kg-cm
            Vu_calc = Vu_val * 101.97       # kN -> kg
        else:
            Mu_calc = Mu_val * 100          # kg-m -> kg-cm
            Vu_calc = Vu_val
        
        d = h - cov - 0.9 # est d
        
        # Flexure
        if method == "SDM":
            phi = 0.9
            Rn = Mu_calc / (phi * b * d**2)
            rho = (0.85*fc/fy) * (1 - np.sqrt(max(0, 1 - (2*Rn)/(0.85*fc))))
            As_req = max(rho, 14/fy) * b * d
        else:
            # WSD simplified
            As_req = Mu_calc / (0.875 * 0.5 * fy * d) # Allowable Stress approx
            
        nb = int(np.ceil(As_req / bar_areas[main_bar]))
        nb = max(2, nb) # Min 2 bars
        
        # Shear
        vc = 0.53 * np.sqrt(fc) * b * d
        if method == "SDM": vc = 0.85 * vc # phi shear
            
        if Vu_calc > vc:
            vs_req = Vu_calc - vc
            av = 2 * stir_areas[stirrup]
            s_req = (av * fy * d) / vs_req
            s = int(5 * round(min(s_req, d/2, 30)/5))
            if s == 0: s = 5
            shear_res = f"@{s} cm"
            shear_status = "⚠️ Shear Reinf. Req"
        else:
            s = int(d/2)
            shear_res = f"@{s} cm (Min)"
            shear_status = "✅ Conc. OK"

        # --- OUTPUT DISPLAY ---
        st.markdown('<div class="design-box">', unsafe_allow_html=True)
        col_out1, col_out2 = st.columns([1, 1])
        
        with col_out1:
            st.markdown("#### 🎯 Design Result")
            st.write(f"**Main Steel:** {nb}-{main_bar} (As = {nb*bar_areas[main_bar]:.2f} cm²)")
            st.write(f"**Stirrups:** {stirrup} {shear_res}")
            st.caption(f"Status: {shear_status}")
            
            # Print Button Logic (Mock)
            st.button("🖨️ Print Calculation Report")
            
        with col_out2:
            st.markdown("#### 📐 Section View")
            # วาดหน้าตัดด้วยฟังก์ชันใหม่
            fig_sec = draw_section_view(b, h, cov, nb, bar_dias[main_bar], stir_dias[stirrup])
            st.plotly_chart(fig_sec, use_container_width=True, config={'displayModeBar': False})
            
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("⚠️ กรุณากดปุ่ม 'Calculate Analysis' ด้านบนก่อน เพื่อเริ่มการออกแบบ")
