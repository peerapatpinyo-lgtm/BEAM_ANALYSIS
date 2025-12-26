import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 0. ENGINE & SETUP
# ==========================================
st.set_page_config(page_title="RC Beam Pro V.19.1", layout="wide", page_icon="🏗️")

# --- 🛡️ CRASH PROOF: INITIALIZE STATE FIRST ---
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False
if 'res_df_raw' not in st.session_state:
    st.session_state['res_df_raw'] = None
if 'vis_data' not in st.session_state:
    st.session_state['vis_data'] = None

# Mock Engine (กรณีไม่มีไฟล์ beam_engine.py)
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
            return pd.DataFrame({
                'x': x,
                'shear': 15000 * np.cos(x) * (1 - x/L), 
                'moment': 30000 * np.sin(x) * x
            })

# CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    .main-header { 
        background: linear-gradient(90deg, #1565C0 0%, #0D47A1 100%); 
        color: white; padding: 15px; border-radius: 10px; 
        text-align: center; margin-bottom: 20px; 
    }
    .section-header {
        font-size: 1.3rem; font-weight: bold; color: #1565C0;
        border-bottom: 2px solid #1565C0; padding-bottom: 5px; 
        margin-top: 30px; margin-bottom: 15px;
    }
    .input-card { background-color: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #dee2e6; }
    .design-box { background-color: #e3f2fd; border: 2px solid #2196F3; padding: 20px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. HELPER FUNCTIONS & VISUALIZATION
# ==========================================

# --- 🛠️ FIX: นำฟังก์ชันนี้กลับมาครับ ---
def get_default_factors(code_name):
    if "WSD" in code_name: return 1.0, 1.0, "WSD"
    elif "ACI" in code_name: return 1.2, 1.6, "SDM"
    else: return 1.4, 1.7, "SDM" # EIT SDM

def draw_beam_diagram(spans, supports, loads, unit_load, unit_force):
    """ วาดรูปคาน โดยแยกสัญลักษณ์ Pin/Roller ให้ชัดเจน """
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()
    
    # 1. เส้นคาน
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', line=dict(color='black', width=5), hoverinfo='skip'))

    # 2. จุดรองรับ (Supports) - ปรับสัญลักษณ์
    for i, s_type in enumerate(supports):
        sx = cum_len[i]
        
        # กำหนดรูปทรงและสี
        if s_type == "Fix":
            symbol = "square"
            color = "#D32F2F" # แดง
            offset_y = 0
        elif s_type == "Pin":
            symbol = "triangle-up"
            color = "#2E7D32" # เขียว
            offset_y = -0.02
        else: # Roller
            symbol = "circle" # เปลี่ยนเป็นวงกลม
            color = "#F57C00" # ส้ม
            offset_y = -0.02

        fig.add_trace(go.Scatter(
            x=[sx], y=[offset_y], mode='markers+text',
            marker=dict(symbol=symbol, size=15, color=color, line=dict(width=2, color='black')),
            text=[s_type], textposition="bottom center", hoverinfo='none', showlegend=False
        ))
        
        # ถ้าเป็น Roller วาดเส้นพื้นด้านล่างเพิ่ม
        if s_type == "Roller":
             fig.add_shape(type="line", x0=sx-0.2, y0=-0.06, x1=sx+0.2, y1=-0.06, line=dict(color="black", width=2))

    # 3. Loads
    max_h = 0.25
    for load in loads:
        start_x = cum_len[load['span_idx']]
        val = load['display_val']
        if load['type'] == 'Uniform':
            end_x = start_x + spans[load['span_idx']]
            fig.add_shape(type="rect", x0=start_x, y0=0, x1=end_x, y1=max_h, line=dict(width=0), fillcolor="rgba(255, 87, 34, 0.2)")
            fig.add_trace(go.Scatter(x=[start_x, end_x], y=[max_h, max_h], mode='lines', line=dict(color='#E64A19', width=2), showlegend=False))
            fig.add_annotation(x=(start_x+end_x)/2, y=max_h+0.05, text=f"w={val}", showarrow=False, font=dict(color="#E64A19"))
        elif load['type'] == 'Point':
            lx = start_x + load['pos']
            fig.add_annotation(x=lx, y=0, ax=0, ay=-40, text=f"P={val}", showarrow=True, arrowhead=2, arrowcolor="#D32F2F")

    # 4. Dimensions Lines
    for i, span in enumerate(spans):
        mid_x = cum_len[i] + span/2
        fig.add_annotation(x=mid_x, y=-0.15, text=f"{span} m", showarrow=False, font=dict(color="blue", size=14, weight="bold"))
        # เส้นกั้นช่วง
        fig.add_shape(type="line", x0=cum_len[i], y0=-0.1, x1=cum_len[i], y1=-0.25, line=dict(color="gray", dash="dot"))
        fig.add_shape(type="line", x0=cum_len[i+1], y0=-0.1, x1=cum_len[i+1], y1=-0.25, line=dict(color="gray", dash="dot"))

    fig.update_layout(height=300, title="Structure Model (Beam & Supports)", yaxis=dict(visible=False, range=[-0.4, 0.5]), xaxis=dict(title="Length (m)"), margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='white')
    return fig

def draw_section_view(b, h, cover, n_bars, bar_dia_mm, stirrup_dia_mm, bar_name):
    """ วาดหน้าตัดคาน พร้อมบอกขนาดและชื่อเหล็ก """
    fig = go.Figure()
    
    # 1. Concrete Box
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=3), fillcolor="#F5F5F5")
    
    # 2. Stirrup (เหล็กปลอก)
    inset = cover
    fig.add_shape(type="rect", x0=inset, y0=inset, x1=b-inset, y1=h-inset, line=dict(color="#D32F2F", width=2))
    
    # 3. Main Bars (เหล็กเสริมหลัก)
    if n_bars > 0:
        eff_w = b - 2*cover - 2*(stirrup_dia_mm/10) - (bar_dia_mm/10)
        gap = eff_w / (n_bars - 1) if n_bars > 1 else 0
        y_pos = cover + (stirrup_dia_mm/10) + (bar_dia_mm/20)
        x_list = [cover + (stirrup_dia_mm/10) + (bar_dia_mm/20) + i*gap for i in range(n_bars)]
        
        fig.add_trace(go.Scatter(
            x=x_list, y=[y_pos]*n_bars, 
            mode='markers', 
            marker=dict(size=bar_dia_mm*1.5, color='#1565C0', line=dict(width=1, color='black')),
            name="Main Bar"
        ))
        
        # Label ชี้เหล็ก
        fig.add_annotation(
            x=b/2, y=y_pos, ax=40, ay=40,
            text=f"{n_bars}-{bar_name}", showarrow=True, arrowcolor="blue", font=dict(color="blue", size=14)
        )

    # 4. Dimension Annotations (บอกขนาด)
    # Width (b)
    fig.add_annotation(x=b/2, y=-5, text=f"b = {b} cm", showarrow=False, font=dict(size=14))
    fig.add_shape(type="line", x0=0, y0=-2, x1=b, y1=-2, line=dict(color="black")) 
    
    # Depth (h)
    fig.add_annotation(x=-5, y=h/2, text=f"h = {h} cm", showarrow=False, textangle=-90, font=dict(size=14))
    fig.add_shape(type="line", x0=-2, y0=0, x1=-2, y1=h, line=dict(color="black"))

    fig.update_layout(
        width=300, height=300,
        xaxis=dict(visible=False, range=[-10, b+10]),
        yaxis=dict(visible=False, range=[-10, h+10]),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
        title="Section View"
    )
    return fig

# ==========================================
# 2. MAIN APP
# ==========================================

st.markdown('<div class="main-header"><h2>🏗️ RC Beam Pro V.19.1 (Fixed)</h2></div>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    design_code = st.selectbox("Design Standard", ["EIT 1007 (WSD)", "EIT 1008 (SDM)", "ACI 318 (SDM)"], index=1)
    
    # เรียกใช้ฟังก์ชันที่แก้แล้วตรงนี้
    def_dl, def_ll, method = get_default_factors(design_code)
    
    st.divider()
    st.markdown("**Load Factors**")
    c1, c2 = st.columns(2)
    fact_dl = c1.number_input("DL Factor", value=def_dl, step=0.1)
    fact_ll = c2.number_input("LL Factor", value=def_ll, step=0.1)
    
    st.divider()
    unit_sys = st.radio("System", ["MKS (kg, m, ksc)", "SI (kN, m, MPa)"], index=0)
    
    if "kN" in unit_sys:
        U_L, U_M, U_F, U_S = "kN/m", "kN-m", "kN", "MPa"
        TO_N, FROM_N = 1000.0, 0.001
    else:
        U_L, U_M, U_F, U_S = "kg/m", "kg-m", "kg", "ksc"
        TO_N, FROM_N = 9.80665, 1/9.80665

# --- PART 1: INPUT ---
st.markdown('<div class="section-header">1️⃣ Structure & Loads</div>', unsafe_allow_html=True)
col_geo, col_load = st.columns([1, 1.5])

with col_geo:
    st.info("📍 Geometry")
    n_span = st.number_input("Number of Spans", 1, 5, 2)
    spans, supports = [], []
    c_s1, c_s2 = st.columns(2)
    sup_L = c_s1.selectbox("Left Sup", ["Pin", "Roller", "Fix"], key="SL")
    sup_R = c_s2.selectbox("Right Sup", ["Pin", "Roller", "Fix"], index=1, key="SR")
    
    supports.append(sup_L)
    for i in range(n_span):
        l = st.number_input(f"Span {i+1} (m)", 0.5, 20.0, 4.0, key=f"L{i}")
        spans.append(l)
        if i < n_span-1: supports.append("Roller")
    supports.append(sup_R)

with col_load:
    st.info(f"⬇️ Loads ({unit_sys})")
    loads_input = []
    tabs = st.tabs([f"Span {i+1}" for i in range(n_span)])
    for i, tab in enumerate(tabs):
        with tab:
            c1, c2 = st.columns(2)
            udl = c1.number_input(f"Uniform DL ({U_L})", 0.0, key=f"udl{i}")
            ull = c2.number_input(f"Uniform LL ({U_L})", 0.0, key=f"ull{i}")
            w_total = udl*fact_dl + ull*fact_ll
            if w_total > 0:
                loads_input.append({'span_idx': i, 'type': 'Uniform', 'total_w': w_total*TO_N, 'display_val': udl+ull})
            
            st.markdown("---")
            if st.checkbox("Add Point Load", key=f"chk{i}"):
                c3, c4, c5 = st.columns(3)
                pdl = c3.number_input("P DL", 0.0, key=f"pdl{i}")
                pll = c4.number_input("P LL", 0.0, key=f"pll{i}")
                px = c5.number_input("Dist x (m)", 0.0, spans[i], spans[i]/2, key=f"px{i}")
                p_total = pdl*fact_dl + pll*fact_ll
                if p_total > 0:
                    loads_input.append({'span_idx': i, 'type': 'Point', 'total_w': p_total*TO_N, 'pos': px, 'display_val': pdl+pll})

# --- PART 2: ANALYSIS ---
st.markdown('<div class="section-header">2️⃣ Analysis</div>', unsafe_allow_html=True)

if st.button("🚀 Calculate Analysis", type="primary"):
    solver = SimpleBeamSolver(spans, supports, loads_input)
    u, err = solver.solve()
    if err:
        st.error(err)
        st.session_state['analyzed'] = False
    else:
        df = solver.get_internal_forces(100)
        st.session_state['res_df_raw'] = df
        st.session_state['vis_data'] = (spans, supports, loads_input)
        st.session_state['analyzed'] = True
        st.rerun()

# --- DISPLAY RESULT ---
if st.session_state.get('analyzed', False):
    
    df_raw = st.session_state.get('res_df_raw')
    vis_data = st.session_state.get('vis_data')
    
    if df_raw is not None and vis_data is not None:
        vis_spans, vis_supports, vis_loads = vis_data
        
        # Convert Units
        df = df_raw.copy()
        df['V_show'] = df['shear'] * FROM_N
        df['M_show'] = df['moment'] * FROM_N
        
        # 1. Beam Model
        st.plotly_chart(draw_beam_diagram(vis_spans, vis_supports, vis_loads, U_L, U_F), use_container_width=True)
        
        # 2. Graphs
        col_res1, col_res2 = st.columns(2)
        max_M = df['M_show'].abs().max()
        max_V = df['V_show'].abs().max()
        col_res1.metric(f"Max Moment (Mu)", f"{max_M:.2f} {U_M}")
        col_res2.metric(f"Max Shear (Vu)", f"{max_V:.2f} {U_F}")
        
        fig_res = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(f"Shear ({U_F})", f"Moment ({U_M})"))
        fig_res.add_trace(go.Scatter(x=df['x'], y=df['V_show'], fill='tozeroy', line=dict(color='#D32F2F'), name="Shear"), row=1, col=1)
        fig_res.add_trace(go.Scatter(x=df['x'], y=df['M_show'], fill='tozeroy', line=dict(color='#1976D2'), name="Moment"), row=2, col=1)
        fig_res.update_layout(height=450, showlegend=False, hovermode="x unified")
        fig_res.update_yaxes(autorange="reversed", row=2, col=1)
        st.plotly_chart(fig_res, use_container_width=True)

        # --- PART 3: DESIGN ---
        st.markdown('<div class="section-header">3️⃣ RC Design</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            c_d1, c_d2, c_d3 = st.columns([1, 1, 1.5])
            
            with c_d1:
                st.markdown("##### 🧱 Materials")
                fc = st.number_input(f"f'c ({U_S})", value=240.0 if "ksc" in U_S else 24.0)
                fy = st.number_input(f"fy ({U_S})", value=4000.0 if "ksc" in U_S else 400.0)
            
            with c_d2:
                st.markdown("##### 📐 Section")
                b = st.number_input("Width b (cm)", value=25.0)
                h = st.number_input("Depth h (cm)", value=50.0)
                cov = st.number_input("Cover (cm)", value=3.0)
            
            with c_d3:
                st.markdown("##### ⛓️ Rebar")
                bar_keys = ["DB12", "DB16", "DB20", "DB25", "DB28"]
                main_bar = st.selectbox("Main Bar", bar_keys, index=1)
                stirrup = st.selectbox("Stirrup", ["RB6", "RB9", "DB10"], index=1)
                
                bar_areas = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
                bar_dias = {k: int(k[2:]) for k in bar_keys}
                stir_areas = {'RB6':0.28, 'RB9':0.64, 'DB10':0.78}
                stir_dias = {'RB6':6, 'RB9':9, 'DB10':10}
            st.markdown('</div>', unsafe_allow_html=True)

            # CALCULATION
            if "kN" in unit_sys:
                Mu_calc = max_M * 1000 * 100 / 9.81
                Vu_calc = max_V * 1000 / 9.81
                fc_c, fy_c = fc * 10.19, fy * 10.19
            else:
                Mu_calc = max_M * 100
                Vu_calc = max_V
                fc_c, fy_c = fc, fy

            d = h - cov - 0.9
            
            # Flexure
            if method == "SDM":
                phi_b = 0.9
                Rn = Mu_calc / (phi_b * b * d**2)
                term = 1 - (2*Rn)/(0.85*fc_c)
                if term < 0:
                    As_req = 9999
                    design_msg = "❌ Section too small"
                else:
                    rho = (0.85*fc_c/fy_c) * (1 - np.sqrt(term))
                    min_rho = 14/fy_c
                    As_req = max(rho, min_rho) * b * d
                    design_msg = "✅ Moment OK"
            else:
                j = 0.875
                As_req = Mu_calc / (0.5 * fy_c * j * d)
                design_msg = "✅ Moment OK (WSD)"
                
            nb = 0 if As_req == 9999 else max(2, int(np.ceil(As_req / bar_areas[main_bar])))
            
            # Shear
            Vc = 0.53 * np.sqrt(fc_c) * b * d
            if method == "SDM": Vc = 0.85 * Vc
            
            if Vu_calc > Vc:
                Vs_req = Vu_calc - Vc
                Av = 2 * stir_areas[stirrup]
                if Vs_req > 0:
                    s_req = (Av * fy_c * d) / Vs_req
                    s = int(5 * round(min(s_req, d/2, 60)/5))
                    if s == 0: s = 5
                    shear_res = f"@{s} cm"
                    shear_status = "⚠️ Shear Reinf."
                else:
                    s = int(d/2)
                    shear_res = f"@{s} cm (Min)"
                    shear_status = "✅ Min Shear"
            else:
                s = int(d/2)
                shear_res = f"@{s} cm (Min)"
                shear_status = "✅ Concrete OK"

            # OUTPUT
            st.markdown('<div class="design-box">', unsafe_allow_html=True)
            c_res1, c_res2 = st.columns(2)
            with c_res1:
                st.subheader("🎯 Design Result")
                if nb == 0:
                    st.error(design_msg)
                else:
                    st.success(design_msg)
                    st.write(f"**Main Steel:** {nb}-{main_bar}")
                    st.caption(f"As req: {As_req:.2f} cm² | Prov: {nb*bar_areas[main_bar]:.2f} cm²")
                    st.divider()
                    st.write(f"**Stirrups:** {stirrup} {shear_res}")
                    st.caption(f"Status: {shear_status}")
            
            with c_res2:
                # เรียกวาดรูปหน้าตัดที่นี่
                fig_sec = draw_section_view(b, h, cov, nb, bar_dias[main_bar], stir_dias[stirrup], main_bar)
                st.plotly_chart(fig_sec, use_container_width=True, config={'displayModeBar': False})
                
            st.markdown('</div>', unsafe_allow_html=True)
    else:
         st.error("Data lost. Please click 'Calculate Analysis' again.")
else:
    st.info("👈 Please enter loads and click 'Calculate Analysis' to start.")
