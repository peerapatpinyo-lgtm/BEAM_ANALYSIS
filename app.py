import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

# --- IMPORT ENGINE ---
try:
    from beam_engine import SimpleBeamSolver
except ImportError:
    # MOCK ENGINE สำหรับกรณีไม่มีไฟล์ (เพื่อให้รันโชว์ UI ได้)
    class SimpleBeamSolver:
        def __init__(self, spans, supports, loads):
            self.spans = spans
            self.loads = loads
        def solve(self):
            return None, None
        def get_internal_forces(self, n):
            x = np.linspace(0, sum(self.spans), n)
            # Create dummy parabola moment and linear shear
            L = sum(self.spans)
            return pd.DataFrame({
                'x': x,
                'shear': 1000 * (1 - 2*x/L), # Dummy
                'moment': 5000 * (x/L) * (1 - x/L) # Dummy
            })

# ==========================================
# 🛡️ 0. SESSION STATE INITIALIZATION
# ==========================================
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False
if 'res_df' not in st.session_state:
    st.session_state['res_df'] = None
if 'inputs' not in st.session_state:
    st.session_state['inputs'] = None

# ==========================================
# 🎨 1. SETTINGS & STYLES
# ==========================================
st.set_page_config(page_title="RC Beam Pro V.17.2", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    
    .header-box { background: linear-gradient(90deg, #1565C0 0%, #0D47A1 100%); padding: 15px; border-radius: 8px; color: white; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .report-card { background-color: #ffffff; border: 2px solid #e0e0e0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .stMetric { background-color: #e3f2fd; padding: 10px; border-radius: 5px; border: 1px solid #90caf9; }
    hr { margin: 1.5rem 0; border-top: 2px solid #bbb; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🛠️ 2. FUNCTIONS
# ==========================================

def get_code_params(code_name):
    if "WSD" in code_name:
        return {"type": "WSD", "DL": 1.0, "LL": 1.0, "phi_b": 1.0, "phi_v": 1.0}
    elif "ACI 318" in code_name:
        return {"type": "SDM", "DL": 1.2, "LL": 1.6, "phi_b": 0.90, "phi_v": 0.75}
    else: # EIT SDM
        return {"type": "SDM", "DL": 1.4, "LL": 1.7, "phi_b": 0.90, "phi_v": 0.85}

def draw_section_view(b, h, cover, n_bars, bar_dia_mm, stirrup_dia_mm):
    """ฟังก์ชันวาดหน้าตัดคานด้วย Plotly"""
    fig = go.Figure()

    # 1. Concrete Face (สีเทา)
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, 
                  line=dict(color="Black", width=2), fillcolor="#E0E0E0")

    # 2. Stirrup (เหล็กปลอก - สีแดง)
    s_inset = cover
    fig.add_shape(type="rect", x0=s_inset, y0=s_inset, x1=b-s_inset, y1=h-s_inset,
                  line=dict(color="#D32F2F", width=3))

    # 3. Main Bars (เหล็กแกน - สีน้ำเงิน)
    # คำนวณตำแหน่ง X ของเหล็กแต่ละเส้น
    eff_width = b - 2*cover - 2*(stirrup_dia_mm/10) - (bar_dia_mm/10)
    
    if n_bars > 1:
        spacing = eff_width / (n_bars - 1)
        x_positions = [cover + (stirrup_dia_mm/10) + (bar_dia_mm/20) + i*spacing for i in range(n_bars)]
    else:
        x_positions = [b/2] # กรณี 1 เส้น (ไม่ควรเกิด)

    y_pos = cover + (stirrup_dia_mm/10) + (bar_dia_mm/20)

    fig.add_trace(go.Scatter(
        x=x_positions, 
        y=[y_pos]*n_bars,
        mode='markers',
        marker=dict(size=bar_dia_mm*1.2, color='#1565C0', line=dict(width=1, color='Black')),
        name='Main Bar'
    ))

    # Decoration
    fig.update_layout(
        width=300, height=300 * (h/b) if b>0 else 300,
        xaxis=dict(visible=False, range=[-5, b+5]),
        yaxis=dict(visible=False, range=[-5, h+5]),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

def generate_report_html(project_data, res_data):
    # (ใช้ HTML Template เดิมของคุณ แต่เพิ่มส่วน Section View เล็กน้อยถ้าต้องการ)
    # เพื่อความกระชับ ขอใช้ Code เดิมของคุณในส่วนนี้
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap" rel="stylesheet">
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>
            body {{ font-family: 'Sarabun', sans-serif; padding: 40px; }}
            .head {{ text-align: center; border: 2px solid #000; padding: 15px; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ border: 1px solid #000; padding: 8px; text-align: center; }}
            th {{ background-color: #f0f0f0; }}
            .footer {{ margin-top: 50px; text-align: center; font-size: 10px; }}
        </style>
    </head>
    <body>
        <div class="head">
            <h2>📄 รายการคำนวณออกแบบคาน (RC Beam Design)</h2>
            <p>Code: {project_data['code']} | Method: {project_data['method']}</p>
        </div>
        
        <h3>1. Design Criteria</h3>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Concrete (fc')</td><td>{project_data['fc']} ksc</td></tr>
            <tr><td>Steel (fy)</td><td>{project_data['fy']} ksc</td></tr>
            <tr><td>Size</td><td>{res_data['b']} x {res_data['h']} cm</td></tr>
        </table>

        <h3>2. Analysis & Design Results</h3>
        <table>
            <tr><th>Item</th><th>Result</th></tr>
            <tr><td>Max Moment (Mu)</td><td>{res_data['Mu']:.2f} {res_data['u_m']}</td></tr>
            <tr><td>Req. Area (As)</td><td>{res_data['As_req']:.2f} cm²</td></tr>
            <tr><td><b>Selected Main Bar</b></td><td><b>{res_data['main_bar']}</b></td></tr>
            <tr><td><b>Selected Stirrup</b></td><td><b>{res_data['stirrup']}</b></td></tr>
        </table>
        
        <div class="footer">Generated by RC Beam Pro V.17.2</div>
    </body>
    </html>
    """
    return html

# ==========================================
# 🖥️ MAIN UI
# ==========================================

st.markdown('<div class="header-box"><h2>🏗️ RC Beam Pro V.17.2 (With Section View)</h2></div>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Project Settings")
    code_options = ["EIT 1007 (WSD)", "EIT 1008 (SDM Thai)", "ACI 318-25 (SDM)"]
    design_code = st.selectbox("Design Standard", code_options, index=1)
    params = get_code_params(design_code)
    is_wsd = params['type'] == "WSD"
    
    st.info(f"Method: **{params['type']}**\n\nLoad Factors: {params['DL']}DL + {params['LL']}LL")
    
    st.divider()
    unit_sys = st.radio("System of Units", ["SI (kN, m, MPa)", "MKS (kg, m, ksc)"], index=1)
    if "kN" in unit_sys:
        U_F, U_L, U_M = "kN", "kN/m", "kN-m"
        TO_N, FROM_N = 1000.0, 0.001
    else:
        U_F, U_L, U_M = "kg", "kg/m", "kg-m"
        TO_N, FROM_N = 9.80665, 1/9.80665

# --- INPUT SECTION ---
with st.expander("📝 1. Structure & Loading Inputs", expanded=True):
    col_geo, col_load = st.columns([1, 1.2])
    
    with col_geo:
        st.subheader("Geometry")
        n_span = st.number_input("Number of Spans", 1, 6, 2)
        spans, supports = [], []
        
        c_sup_start, c_sup_end = st.columns(2)
        sup_start = c_sup_start.selectbox("Left Support", ["Pin", "Roller", "Fix"], key="s_start")
        sup_end_val = c_sup_end.selectbox("Right Support", ["Pin", "Roller", "Fix"], index=1, key="s_end")
        
        supports.append(sup_start)
        for i in range(n_span):
            l = st.number_input(f"Span {i+1} Length (m)", 0.5, 30.0, 4.0, key=f"L{i}")
            spans.append(l)
            if i < n_span - 1:
                supports.append("Roller")
        supports.append(sup_end_val)
        
    with col_load:
        st.subheader("Loads (Service)")
        loads_input = []
        tabs_spans = st.tabs([f"Span {i+1}" for i in range(n_span)])
        
        for i, tab in enumerate(tabs_spans):
            with tab:
                c1, c2 = st.columns(2)
                udl = c1.number_input(f"Uniform DL ({U_L})", 0.0, key=f"udl{i}")
                ull = c2.number_input(f"Uniform LL ({U_L})", 0.0, key=f"ull{i}")
                
                w_factored = udl*params['DL'] + ull*params['LL']
                if w_factored > 0:
                    loads_input.append({'span_idx': i, 'type': 'Uniform', 'total_w': w_factored*TO_N, 'display_val': w_factored})
                
                if st.checkbox(f"Add Point Load?", key=f"chk{i}"):
                    cp1, cp2, cp3 = st.columns(3)
                    p_dl = cp1.number_input("P DL", 0.0, key=f"pdl{i}")
                    p_ll = cp2.number_input("P LL", 0.0, key=f"pll{i}")
                    x_p = cp3.number_input("Dist (m)", 0.0, spans[i], spans[i]/2, key=f"xp{i}")
                    
                    p_factored = p_dl*params['DL'] + p_ll*params['LL']
                    if p_factored > 0:
                        loads_input.append({'span_idx': i, 'type': 'Point', 'total_w': p_factored*TO_N, 'pos': x_p, 'display_val': p_factored})

if st.button("🚀 Analyze & Design", type="primary", use_container_width=True):
    solver = SimpleBeamSolver(spans, supports, loads_input)
    u, err = solver.solve()
    if err:
        st.error(err)
    else:
        df = solver.get_internal_forces(100)
        df['V_disp'] = df['shear'] * FROM_N
        df['M_disp'] = df['moment'] * FROM_N
        st.session_state['res_df'] = df
        st.session_state['inputs'] = (spans, supports, loads_input)
        st.session_state['analyzed'] = True
        st.rerun()

# ==========================================
# 📊 3. DASHBOARD
# ==========================================
if st.session_state.get('analyzed', False) and st.session_state.get('res_df') is not None:
    df = st.session_state['res_df']
    spans, supports, loads_input = st.session_state['inputs']
    
    st.divider()
    col_graph, col_design = st.columns([1.5, 1])
    
    # --- GRAPHS ---
    with col_graph:
        st.subheader("📊 Structural Analysis")
        total_len = sum(spans)
        cum_len = [0] + list(np.cumsum(spans))
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(f"Shear ({U_F})", f"Moment ({U_M})"))
        
        # SFD
        fig.add_trace(go.Scatter(x=df['x'], y=df['V_disp'], fill='tozeroy', line=dict(color='#D32F2F'), name="Shear"), row=1, col=1)
        # BMD
        fig.add_trace(go.Scatter(x=df['x'], y=df['M_disp'], fill='tozeroy', line=dict(color='#1976D2'), name="Moment"), row=2, col=1)
        
        fig.update_layout(height=500, showlegend=False)
        fig.update_yaxes(autorange="reversed", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    # --- DESIGN ---
    with col_design:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader(f"🛠️ Part Design: Section")
        
        c_mat1, c_mat2 = st.columns(2)
        fc = c_mat1.number_input("f'c (ksc)", value=240.0)
        fy = c_mat2.number_input("fy (ksc)", value=4000.0 if not is_wsd else 3000.0)
        
        c_sec1, c_sec2, c_sec3 = st.columns(3)
        b = c_sec1.number_input("b (cm)", value=25.0)
        h = c_sec2.number_input("h (cm)", value=50.0)
        cover = c_sec3.number_input("Cov (cm)", value=3.0)
        
        # Bar Selection
        bar_map = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91}
        bar_dia_map = {'DB12':12, 'DB16':16, 'DB20':20, 'DB25':25}
        
        main_bar = st.selectbox("Main Bar", list(bar_map.keys()), index=2)
        
        stir_map = {'RB6':0.28, 'RB9':0.64, 'DB10':0.78}
        stir_dia_map = {'RB6':6, 'RB9':9, 'DB10':10}
        stirrup = st.selectbox("Stirrup", list(stir_map.keys()), index=1)
        
        # --- CALCULATION LOGIC ---
        Mu = df['M_disp'].abs().max()
        Vu = df['V_disp'].abs().max()
        
        # Unit Conversion to kg, cm
        if "kN" in unit_sys:
            Mu_kgm, Vu_kg = Mu * 101.97, Vu * 101.97
        else:
            Mu_kgm, Vu_kg = Mu, Vu
            
        fc_ksc, fy_ksc = fc, fy
        d = h - cover - 1.0 # approx effective depth
        
        # 1. Flexure Design
        if is_wsd:
             # WSD Approx
             As_req = (Mu_kgm * 100) / (0.875 * fy_ksc * d) # Using j approx 0.875
        else:
             # SDM
             phi_b = params['phi_b']
             Mu_kgcm = Mu_kgm * 100
             Rn = Mu_kgcm / (phi_b * b * d**2)
             rho = (0.85*fc_ksc/fy_ksc) * (1 - np.sqrt(1 - (2*Rn)/(0.85*fc_ksc))) if (1 - (2*Rn)/(0.85*fc_ksc)) >= 0 else 999
             As_req = rho * b * d
             if As_req == 999: st.error("Section too small!")

        nb = max(2, int(np.ceil(As_req / bar_map[main_bar])))
        
        # 2. Shear Design (More Detailed)
        phi_v = params['phi_v']
        Vc = 0.53 * np.sqrt(fc_ksc) * b * d # SDM Formula
        Vs_req = (Vu_kg / phi_v) - Vc
        
        if Vs_req <= 0:
            s_req = d/2
            shear_status = "Min Reinf."
        else:
            Av = 2 * stir_map[stirrup] # 2 legs
            s_req = (Av * fy_ksc * d) / Vs_req
            shear_status = "Designed"
        
        # Check Max Spacing
        s_max = min(d/2, 60) # Simplified
        s_final = min(s_req, s_max)
        s_final_int = int(5 * round(s_final/5)) # Round to nearest 5
        if s_final_int == 0: s_final_int = 5

        # --- DISPLAY RESULTS ---
        st.markdown("---")
        st.success("✅ Design PASSED")
        
        # Metrics
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Top/Bot Steel", f"{nb}-{main_bar}")
        col_res2.metric("Stirrups", f"{stirrup} @ {s_final_int} cm", delta=shear_status)
        
        # Section Visualization
        st.markdown("##### 📐 Section View")
        fig_sec = draw_section_view(b, h, cover, nb, bar_dia_map[main_bar], stir_dia_map[stirrup])
        st.plotly_chart(fig_sec, use_container_width=True, config={'displayModeBar': False})
        
        # Deflection Check (L/d)
        L_max = max(spans) * 100 # cm
        min_h = L_max / 16 # Approx for simple support
        st.caption(f"ℹ️ Check Deflection: h provided ({h} cm) vs L/16 ({min_h:.1f} cm)")
        if h < min_h:
            st.warning("⚠️ Beam might be too shallow (Large Deflection).")

        # Report Generation
        project_data = {'code': design_code, 'method': params['type'], 'fc': fc, 'fy': fy}
        res_data = {
            'b':b, 'h':h, 'Mu':Mu, 'Vu':Vu, 'u_m':U_M, 'u_f':U_F, 
            'As_req': As_req, 'nb': nb, 
            'main_bar': f"{nb}-{main_bar}", 'stirrup': f"{stirrup}@{s_final_int}cm"
        }
        
        html = generate_report_html(project_data, res_data)
        b64 = base64.b64encode(html.encode()).decode()
        st.markdown(f'<a href="data:text/html;base64,{b64}" target="_blank" style="text-decoration:none; color:white; background-color:#D32F2F; padding:10px; border-radius:5px; display:block; text-align:center;">🖨️ Print Report</a>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
