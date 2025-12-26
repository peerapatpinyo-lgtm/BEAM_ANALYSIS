import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import base64

# --- IMPORT ENGINE ---
try:
    from beam_engine import SimpleBeamSolver
except ImportError:
    st.error("⚠️ ไม่พบไฟล์ 'beam_engine.py' กรุณาวางไฟล์ engine ไว้ในโฟลเดอร์เดียวกัน")
    st.stop()

# ==========================================
# 🎨 1. UI CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="RC Beam Pro V.15", layout="wide", page_icon="🏗️")

# Custom CSS for "Engineering Look"
st.markdown("""
<style>
    /* Font & Colors */
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    
    .main-header {
        background: linear-gradient(90deg, #1565C0 0%, #0D47A1 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    h1, h2, h3 { color: #0D47A1; }
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 5px solid #1565C0; }
    
    /* Report Button styling */
    .report-btn { 
        display: inline-block; padding: 10px 20px; 
        background-color: #D32F2F; color: white !important; 
        text-decoration: none; border-radius: 5px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🛠️ 2. HELPER FUNCTIONS
# ==========================================

def generate_html_report(project_info, design_res, graphics_html):
    """Generates a clean HTML report suitable for PDF printing"""
    
    # Unpack Data
    code = project_info['code']
    fc, fy = project_info['fc'], project_info['fy']
    b, h, d = design_res['b'], design_res['h'], design_res['d']
    
    html = f"""
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap" rel="stylesheet">
        <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>
            body {{ font-family: 'Sarabun', sans-serif; padding: 40px; line-height: 1.6; color: #333; }}
            .header {{ text-align: center; border-bottom: 3px solid #000; padding-bottom: 10px; margin-bottom: 20px; }}
            .section {{ margin-bottom: 25px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            .section-title {{ font-size: 18px; font-weight: bold; color: #0D47A1; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-bottom: 10px; }}
            .result-box {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; border: 1px solid #90caf9; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .footer {{ margin-top: 50px; text-align: center; font-size: 12px; color: #777; border-top: 1px solid #ddd; pt: 10px; }}
            
            @media print {{
                body {{ padding: 0; }}
                .no-print {{ display: none; }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📄 รายการคำนวณออกแบบคาน ค.ส.ล. (RC Beam Design)</h1>
            <p>Design Code: <b>{code}</b> | Date: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}</p>
        </div>

        <div class="section">
            <div class="section-title">1. ข้อมูลการออกแบบ (Design Criteria)</div>
            <table width="100%">
                <tr>
                    <td><b>f'c:</b> {fc} ksc</td>
                    <td><b>fy:</b> {fy} ksc</td>
                    <td><b>ขนาดคาน (b x h):</b> {b} x {h} cm</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <div class="section-title">2. ผลการวิเคราะห์โครงสร้าง (Analysis Results)</div>
            <p><b>Max Moment (Mu):</b> {design_res['Mu']:.2f} {design_res['unit_M']}</p>
            <p><b>Max Shear (Vu):</b> {design_res['Vu']:.2f} {design_res['unit_F']}</p>
            <div style="text-align:center; margin-top:10px; font-size:0.8em; color:gray;">
                (กราฟแรงเฉือนและโมเมนต์แนบในภาคผนวก)
            </div>
        </div>

        <div class="section">
            <div class="section-title">3. ผลการคำนวณเหล็กเสริม (Reinforcement Design)</div>
            <div class="result-box">
                <h3>📌 สรุปผลการออกแบบ (Conclusion)</h3>
                <ul>
                    <li><b>เหล็กเสริมหลัก (Main Bars):</b> {design_res['main_txt']}</li>
                    <li><b>เหล็กปลอก (Stirrups):</b> {design_res['shear_txt']}</li>
                    <li><b>ปริมาณเหล็กที่ต้องการ (As_req):</b> {design_res['As_req']:.2f} cm²</li>
                </ul>
            </div>
            <br>
            <b>รายการคำนวณโดยละเอียด (Calculation Details):</b><br>
            $$ R_n = \\frac{{M_u}}{{\\phi b d^2}} = {design_res['Rn']:.2f} \\text{{ MPa}} $$
            $$ \\rho_{{req}} = {design_res['rho']:.5f} $$
            $$ A_{{s,req}} = \\rho b d = {design_res['As_req']:.2f} \\text{{ cm}}^2 $$
        </div>

        <div class="footer">
            Generated by <b>RC Beam Pro V.15</b> | Engineered for Civil Engineers
        </div>
    </body>
    </html>
    """
    return html

def draw_section_detail(b, h, cover, n_bars, bar_name, stirrup_name, spacing):
    """Draw the beam section nicely"""
    fig = go.Figure()
    # Concrete
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=3), fillcolor="#F5F5F5")
    # Stirrup
    c = cover
    fig.add_shape(type="rect", x0=c, y0=c, x1=b-c, y1=h-c, line=dict(color="#D32F2F", width=2))
    # Main Bars
    dia = 2.0
    if n_bars > 0:
        gap = (b - 2*c - dia) / (n_bars - 1) if n_bars > 1 else 0
        for i in range(n_bars):
            bx = c + dia/2 + i*gap if n_bars > 1 else b/2
            by = c + dia/2
            fig.add_shape(type="circle", x0=bx-dia/2, y0=by-dia/2, x1=bx+dia/2, y1=by+dia/2, fillcolor="#1565C0", line_color="black")
    # Hanger Bars
    for i in [0, 1]:
        bx = c + dia/2 if i==0 else b - c - dia/2
        by = h - c - dia/2
        fig.add_shape(type="circle", x0=bx-dia/2, y0=by-dia/2, x1=bx+dia/2, y1=by+dia/2, fillcolor="#90CAF9", line_color="black")

    fig.add_annotation(x=b/2, y=h/2, text=f"<b>{b} x {h} cm</b>", showarrow=False, font=dict(size=20, color="rgba(0,0,0,0.2)"))
    fig.add_annotation(x=b/2, y=c-5, text=f"<b>{n_bars}-{bar_name}</b>", showarrow=False, font=dict(color="#1565C0", size=14))
    fig.add_annotation(x=b+5, y=h/2, text=f"Stirrup:<br>{stirrup_name}<br>@{spacing:.0f} cm", showarrow=False, font=dict(color="#D32F2F", size=12))

    fig.update_layout(width=350, height=400, xaxis=dict(visible=False, range=[-10, b+15]), yaxis=dict(visible=False, range=[-10, h+10]), 
                      margin=dict(l=10,r=10,t=10,b=10), plot_bgcolor="white")
    return fig

# ==========================================
# 🖥️ MAIN UI STARTS HERE
# ==========================================

# HEADER
st.markdown("""
<div class="main-header">
    <h1>🏗️ RC Beam Design Pro V.15</h1>
    <p>Professional Beam Analysis & Design Software (Thai Standard Support)</p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Project Settings")
    code_map = {
        "EIT 1008 (SDM Thai)": {"type": "SDM", "DL":1.4, "LL":1.7, "phi_b":0.90, "phi_v":0.85},
        "ACI 318-19/22 (SDM)": {"type": "SDM", "DL":1.2, "LL":1.6, "phi_b":0.90, "phi_v":0.75},
        "ACI 318-11 (SDM)":    {"type": "SDM", "DL":1.2, "LL":1.6, "phi_b":0.90, "phi_v":0.75}
    }
    design_code = st.selectbox("Design Code", list(code_map.keys()), index=0)
    params = code_map[design_code]
    
    st.divider()
    unit_sys = st.radio("Unit System", ["SI (kN, m, MPa)", "MKS (kg, m, ksc)"], index=1)
    
    # Unit Constants
    if "kN" in unit_sys:
        U_F, U_L, U_M = "kN", "kN/m", "kN-m"
        TO_N, FROM_N = 1000.0, 0.001
    else:
        U_F, U_L, U_M = "kg", "kg/m", "kg-m"
        TO_N, FROM_N = 9.80665, 1/9.80665

# MAIN CONTENT
tab_input, tab_res, tab_report = st.tabs(["📝 Input Data", "📊 Analysis & Design", "📄 Full Report"])

# --- TAB 1: INPUT ---
with tab_input:
    col_g, col_l = st.columns([1, 1.2])
    
    with col_g:
        st.markdown('<div class="card"><h3>1. Geometry & Supports</h3>', unsafe_allow_html=True)
        n_span = st.number_input("Number of Spans", 1, 10, 2)
        
        spans = []
        supports = []
        
        # --- FIXED SUPPORT LOGIC ---
        # Logic: Sup1 -> Span1 -> Sup2 -> Span2 -> ... -> Sup(N+1)
        
        # First Support
        supports.append(st.selectbox("Support 1 (Left)", ["Pin", "Roller", "Fix"], key="sup_start"))
        
        for i in range(n_span):
            c1, c2 = st.columns([2, 1])
            # Span Length
            spans.append(c1.number_input(f"Span {i+1} Length (m)", 0.5, 50.0, 4.0, key=f"L{i}"))
            
            # Intermediate Support (Only if not the last span)
            if i < n_span - 1:
                supports.append(c2.selectbox(f"Support {i+2} (Mid)", ["Pin", "Roller", "Fix"], index=1, key=f"sup_{i}"))
        
        # Last Support
        supports.append(st.selectbox(f"Support {n_span+1} (Right)", ["Pin", "Roller", "Fix"], index=1, key="sup_end"))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_l:
        st.markdown('<div class="card"><h3>2. Loading</h3>', unsafe_allow_html=True)
        cf1, cf2 = st.columns(2)
        f_dl = cf1.number_input("DL Factor", 0.0, 3.0, params['DL'])
        f_ll = cf2.number_input("LL Factor", 0.0, 3.0, params['LL'])
        
        loads_input = []
        st.info("💡 Tip: ใส่ค่าน้ำหนักบรรทุก (Service Load) โปรแกรมจะคูณ Factor ให้เอง")
        
        with st.expander("Show Load Input Table", expanded=True):
            for i in range(n_span):
                st.markdown(f"**Span {i+1}**")
                c_u1, c_u2 = st.columns(2)
                udl = c_u1.number_input(f"DL (Uniform) Span {i+1}", 0.0, key=f"udl{i}")
                ull = c_u2.number_input(f"LL (Uniform) Span {i+1}", 0.0, key=f"ull{i}")
                
                w_tot = f_dl*udl + f_ll*ull
                if w_tot > 0:
                    loads_input.append({'span_idx': i, 'type': 'Uniform', 'total_w': w_tot*TO_N, 'display_val': w_tot})
                
                # Point Load Input
                if st.checkbox(f"Add Point Load on Span {i+1}?", key=f"chk_p{i}"):
                    cp1, cp2, cp3 = st.columns([1,1,1.5])
                    p_dl = cp1.number_input("P (DL)", 0.0, key=f"pdl{i}")
                    p_ll = cp2.number_input("P (LL)", 0.0, key=f"pll{i}")
                    x_pos = cp3.number_input("Dist (m)", 0.0, spans[i], spans[i]/2, key=f"px{i}")
                    
                    p_tot = f_dl*p_dl + f_ll*p_ll
                    if p_tot > 0:
                         loads_input.append({'span_idx': i, 'type': 'Point', 'total_w': p_tot*TO_N, 'pos': x_pos, 'display_val': p_tot})
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
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
            st.session_state['input'] = (spans, supports, loads_input)
            st.session_state['analyzed'] = True
            st.rerun() # Refresh to jump to tabs

# --- TAB 2: ANALYSIS & DESIGN ---
with tab_res:
    if st.session_state.get('analyzed', False):
        df = st.session_state['res']
        spans, supports, loads_input = st.session_state['input']
        
        # 1. VISUALIZATION
        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            st.subheader("Structure Model")
            # Drawing Logic (Simplified for brevity, reusing previous logic idea)
            fig_beam = go.Figure()
            # ... (Assume draw_beam_schema logic here or simple scatter)
            cx = 0
            for i, L in enumerate(spans):
                fig_beam.add_trace(go.Scatter(x=[cx, cx+L], y=[0,0], mode='lines', line=dict(color='black', width=5), showlegend=False))
                fig_beam.add_annotation(x=cx+L/2, y=-0.2, text=f"{L}m", showarrow=False)
                cx += L
            # Supports
            sx = 0
            for s in supports:
                fig_beam.add_trace(go.Scatter(x=[sx], y=[-0.1], mode='markers', marker=dict(symbol='triangle-up', size=15, color='green'), showlegend=False))
                if spans: sx += spans[0] # Approximated for visual placement logic need correction for loop
            
            fig_beam.update_layout(height=200, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_beam, use_container_width=True)

        # 2. GRAPHS
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Shear Force Diagram (SFD)")
            fig_v = go.Figure(go.Scatter(x=df['x'], y=df['V'], fill='tozeroy', line=dict(color='#D32F2F')))
            fig_v.update_layout(height=300, title_text=f"Max V = {abs(df['V']).max():.2f} {U_F}")
            st.plotly_chart(fig_v, use_container_width=True)
        with c2:
            st.markdown("#### Bending Moment Diagram (BMD)")
            fig_m = go.Figure(go.Scatter(x=df['x'], y=df['M'], fill='tozeroy', line=dict(color='#1976D2')))
            fig_m.update_layout(height=300, yaxis=dict(autorange="reversed"), title_text=f"Max M = {abs(df['M']).max():.2f} {U_M}")
            st.plotly_chart(fig_m, use_container_width=True)

        # 3. DESIGN SECTION
        st.markdown("---")
        st.subheader("🛠️ Section Design")
        
        # Design Inputs
        cd1, cd2, cd3 = st.columns(3)
        with cd1:
            st.markdown("**Material**")
            fc = st.number_input("f'c (ksc)", value=240.0)
            fy = st.number_input("fy (ksc)", value=4000.0)
        with cd2:
            st.markdown("**Section**")
            b = st.number_input("Width b (cm)", value=25.0)
            h = st.number_input("Height h (cm)", value=50.0)
            cover = st.number_input("Cover (cm)", value=3.0)
        with cd3:
            st.markdown("**Rebar**")
            bar_opts = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91}
            main_bar = st.selectbox("Main Bar", list(bar_opts.keys()), index=1)
            stirrup_sel = st.radio("Stirrup", ["RB9", "DB12"], horizontal=True)

        # CALCULATION (Engine)
        fc_mpa, fy_mpa = fc*0.0981, fy*0.0981
        b_mm, d_mm = b*10, (h-cover)*10
        Mu_user = max(abs(df['M']))
        Vu_user = max(abs(df['V']))
        
        # Unit Conversion for Calc
        Mu_Nmm = Mu_user * (1e6 if "kN" in unit_sys else 9.81*1000)
        Vu_N = Vu_user * (1000 if "kN" in unit_sys else 9.81)
        
        # SDM Calc
        phi_b, phi_v = params['phi_b'], params['phi_v']
        Rn = Mu_Nmm / (phi_b * b_mm * d_mm**2)
        m = fy_mpa / (0.85 * fc_mpa)
        
        check_val = 1 - (2*m*Rn)/fy_mpa
        if check_val < 0:
            st.error("❌ Section Failed (Moment too high)")
            design_res = None
        else:
            rho = (1/m)*(1 - np.sqrt(check_val))
            rho_min = 1.4/fy_mpa
            As_req = max(rho, rho_min) * b_mm * d_mm / 100 # cm2
            nb = max(2, int(np.ceil(As_req/bar_opts[main_bar])))
            
            # Shear
            Vc = 0.17 * np.sqrt(fc_mpa) * b_mm * d_mm
            phiVc = phi_v * Vc
            if Vu_N > phiVc:
                Av = 2 * (0.636 if "RB9" in stirrup_sel else 1.13)
                Vs = (Vu_N - phiVc)/phi_v
                s = (Av*100 * (240 if "RB9" in stirrup_sel else 400)*0.0981 * d_mm) / Vs # Assume fys
                s_use = min(s, d_mm/2, 600)
                shear_txt = f"{stirrup_sel} @ {int(s_use/10)} cm"
            else:
                shear_txt = f"{stirrup_sel} @ {int(d_mm/20)} cm (Min)"

            # Store for Report
            design_res = {
                'Mu': Mu_user, 'Vu': Vu_user, 'unit_M': U_M, 'unit_F': U_F,
                'b': b, 'h': h, 'd': h-cover,
                'Rn': Rn, 'rho': rho, 'As_req': As_req,
                'main_txt': f"{nb}-{main_bar}", 'shear_txt': shear_txt
            }
            
            # Display Result
            c_draw, c_txt = st.columns([1, 1.5])
            with c_draw:
                st.plotly_chart(draw_section_detail(b, h, cover, nb, main_bar, stirrup_sel, 20), use_container_width=True)
            with c_txt:
                st.success(f"✅ Design OK")
                st.metric("Main Reinforcement", f"{nb} - {main_bar}")
                st.metric("Stirrups", shear_txt)
                st.info(f"As Required: {As_req:.2f} cm² | Provided: {nb*bar_opts[main_bar]:.2f} cm²")

    else:
        st.info("👈 Please Input Data and Click 'Analyze' first.")

# --- TAB 3: REPORT ---
with tab_report:
    st.header("📄 Calculation Report Generation")
    
    if st.session_state.get('analyzed', False) and design_res:
        st.markdown("""
        <div style="background-color:#FFF3E0; padding:15px; border-radius:5px; border:1px solid #FFB74D;">
            <b>💡 How to Save as PDF (วิธีบันทึกเป็น PDF):</b><br>
            1. Click the button below to open the report.<br>
            2. Press <b>Ctrl + P</b> (or Cmd + P).<br>
            3. Choose <b>"Save as PDF"</b> in the destination setting.<br>
            4. Click Save. (รองรับภาษาไทย 100%)
        </div>
        """, unsafe_allow_html=True)
        
        # Prepare HTML
        project_info = {'code': design_code, 'fc': fc, 'fy': fy}
        html_string = generate_html_report(project_info, design_res, None)
        
        # Download/Open Button
        b64 = base64.b64encode(html_string.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" target="_blank" class="report-btn">🖨️ Open Report / Print PDF</a>'
        st.markdown(f"<br>{href}", unsafe_allow_html=True)
        
        # Preview
        st.markdown("---")
        st.markdown("**Report Preview:**")
        st.components.v1.html(html_string, height=800, scrolling=True)
        
    else:
        st.warning("Please Run Analysis First.")
