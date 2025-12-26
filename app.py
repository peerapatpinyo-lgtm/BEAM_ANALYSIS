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
    st.error("⚠️ ไม่พบไฟล์ 'beam_engine.py' กรุณาวางไฟล์ engine ไว้ในโฟลเดอร์เดียวกัน")
    st.stop()

# ==========================================
# 🛡️ 0. SESSION STATE INITIALIZATION (FIX BUG)
# ==========================================
# ป้องกัน Error เวลา Refesh หน้าจอ หรือ State หาย
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False
if 'res_df' not in st.session_state:
    st.session_state['res_df'] = None
if 'inputs' not in st.session_state:
    st.session_state['inputs'] = None

# ==========================================
# 🎨 1. SETTINGS & STYLES
# ==========================================
st.set_page_config(page_title="RC Beam Pro V.17.1", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    
    /* UI Decoration */
    .header-box { background: linear-gradient(90deg, #1565C0 0%, #0D47A1 100%); padding: 15px; border-radius: 8px; color: white; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .report-card { background-color: #ffffff; border: 2px solid #e0e0e0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .stMetric { background-color: #e3f2fd; padding: 10px; border-radius: 5px; border: 1px solid #90caf9; }
    
    /* Stronger Divider */
    hr { margin: 1.5rem 0; border-top: 2px solid #bbb; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🛠️ 2. FUNCTIONS
# ==========================================

def get_code_params(code_name):
    """Return parameters based on selected code"""
    if "WSD" in code_name:
        return {"type": "WSD", "DL": 1.0, "LL": 1.0, "phi_b": 1.0, "phi_v": 1.0}
    elif "ACI 318-25" in code_name:
        return {"type": "SDM", "DL": 1.2, "LL": 1.6, "phi_b": 0.90, "phi_v": 0.75}
    elif "EIT 1008" in code_name:
        return {"type": "SDM", "DL": 1.4, "LL": 1.7, "phi_b": 0.90, "phi_v": 0.85}
    else: # Fallback
        return {"type": "SDM", "DL": 1.2, "LL": 1.6, "phi_b": 0.90, "phi_v": 0.75}

def generate_report_html(project_data, res_data):
    """Generate HTML Report with STRONG borders for printing"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap" rel="stylesheet">
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>
            body {{ font-family: 'Sarabun', sans-serif; padding: 40px; line-height: 1.5; color: #000; }}
            
            /* Header Style */
            .head {{ text-align: center; border: 2px solid #000; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
            h2 {{ margin: 0; color: #000; }}
            h3 {{ border-bottom: 2px solid #000; padding-bottom: 5px; margin-top: 25px; color: #000; }}
            
            /* Table Style - FIXED VISIBILITY */
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ border: 1px solid #000 !important; padding: 10px; text-align: center; font-size: 14px; }}
            th {{ background-color: #f0f0f0; font-weight: bold; }}
            
            /* Results Box */
            .res-box {{ border: 2px solid #000; padding: 15px; margin-top: 10px; background-color: #f9f9f9; }}
            
            /* Footer */
            .footer {{ margin-top: 50px; text-align: center; font-size: 10px; color: #555; border-top: 1px solid #000; padding-top: 10px; }}
            
            @media print {{
                body {{ -webkit-print-color-adjust: exact; }}
            }}
        </style>
    </head>
    <body>
        <div class="head">
            <h2>📄 รายการคำนวณออกแบบคาน (RC Beam Design)</h2>
            <p style="margin-top:5px;">Standard: <b>{project_data['code']}</b> | Method: <b>{project_data['method']}</b></p>
        </div>
        
        <h3>1. ข้อมูลการออกแบบ (Design Criteria)</h3>
        <table>
            <tr>
                <th>Parameters</th>
                <th>Value</th>
                <th>Unit</th>
            </tr>
            <tr>
                <td>Concrete Strength (f'c)</td>
                <td>{project_data['fc']}</td>
                <td>ksc</td>
            </tr>
            <tr>
                <td>Rebar Yield Strength (fy)</td>
                <td>{project_data['fy']}</td>
                <td>ksc</td>
            </tr>
            <tr>
                <td>Beam Size (b x h)</td>
                <td>{res_data['b']} x {res_data['h']}</td>
                <td>cm</td>
            </tr>
        </table>

        <h3>2. ผลการวิเคราะห์โครงสร้าง (Structural Analysis)</h3>
        <table>
            <tr>
                <th>Type</th>
                <th>Maximum Value (Factored)</th>
            </tr>
            <tr>
                <td>Max Moment (Mu)</td>
                <td>{res_data['Mu']:.2f} {res_data['u_m']}</td>
            </tr>
            <tr>
                <td>Max Shear (Vu)</td>
                <td>{res_data['Vu']:.2f} {res_data['u_f']}</td>
            </tr>
        </table>

        <h3>3. ผลการออกแบบหน้าตัด (Section Design Results)</h3>
        <div class="res-box">
            <p><b>📌 บทสรุป (Conclusion):</b></p>
            <ul>
                <li><b>เหล็กเสริมหลัก (Main Rebar):</b> <span style="font-size:1.2em; font-weight:bold;">{res_data['main_bar']}</span></li>
                <li><b>เหล็กปลอก (Stirrups):</b> <span style="font-size:1.2em; font-weight:bold;">{res_data['stirrup']}</span></li>
            </ul>
        </div>
        
        <br>
        <b>รายละเอียดการคำนวณ (Calculation Details):</b>
        <div style="padding: 10px; border: 1px dashed #000; margin-top: 5px;">
             $$ M_u = {res_data['Mu']:.2f} \\; {res_data['u_m']} $$
             $$ A_{{s,req}} = {res_data['As_req']:.2f} \\; \\text{{cm}}^2 $$
             $$ \\text{{Selected: }} {res_data['nb']} - {project_data['bar_type']} \\; (A_s = {res_data['As_prov']:.2f} \\; \\text{{cm}}^2) $$
        </div>

        <div class="footer">
            Generated by RC Beam Pro V.17 | Date: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
        </div>
    </body>
    </html>
    """
    return html

# ==========================================
# 🖥️ MAIN UI
# ==========================================

st.markdown('<div class="header-box"><h2>🏗️ RC Beam Pro V.17.1 (Stable)</h2></div>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Project Settings")
    code_options = ["EIT 1007 (WSD)", "EIT 1008 (SDM Thai)", "ACI 318-25 (SDM)", "ACI 318-11 (SDM)"]
    design_code = st.selectbox("Design Standard", code_options, index=0)
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
with st.expander("📝 1. Structure & Loading Inputs (คลิกเพื่อแก้ไข)", expanded=True):
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
                
                st.markdown("---")
                if st.checkbox(f"Add Point Load?", key=f"chk{i}"):
                    cp1, cp2, cp3 = st.columns(3)
                    p_dl = cp1.number_input("P DL", 0.0, key=f"pdl{i}")
                    p_ll = cp2.number_input("P LL", 0.0, key=f"pll{i}")
                    x_p = cp3.number_input("Dist (m)", 0.0, spans[i], spans[i]/2, key=f"xp{i}")
                    
                    p_factored = p_dl*params['DL'] + p_ll*params['LL']
                    if p_factored > 0:
                        loads_input.append({'span_idx': i, 'type': 'Point', 'total_w': p_factored*TO_N, 'pos': x_p, 'display_val': p_factored})

if st.button("🚀 Analyze & Design (คำนวณ)", type="primary", use_container_width=True):
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
        st.rerun() # Force rerun to refresh state

# ==========================================
# 📊 3. DASHBOARD (Safety Checked)
# ==========================================
# เพิ่มการเช็ค: ต้องมี flag analyze และตัวแปร res_df ต้องไม่เป็น None
if st.session_state.get('analyzed', False) and st.session_state.get('res_df') is not None:
    df = st.session_state['res_df']
    spans, supports, loads_input = st.session_state['inputs']
    
    st.divider()
    
    col_graph, col_design = st.columns([1.8, 1])
    
    # --- GRAPHS ---
    with col_graph:
        st.subheader("📊 Structural Analysis")
        total_len = sum(spans)
        cum_len = [0] + list(np.cumsum(spans))
        
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.2, 0.4, 0.4],
            subplot_titles=("Beam Model", f"Shear Force ({U_F})", f"Bending Moment ({U_M})")
        )

        # 1. Model
        fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', line=dict(color='black', width=4), hoverinfo='skip'), row=1, col=1)
        for i, s_type in enumerate(supports):
            sx = cum_len[i]
            sym = "triangle-up" if s_type != "Fix" else "square"
            col = "green" if s_type != "Fix" else "red"
            fig.add_trace(go.Scatter(x=[sx], y=[-0.05], mode='markers', marker=dict(symbol=sym, size=12, color=col), hoverinfo='text', text=f"Sup: {s_type}"), row=1, col=1)

        for load in loads_input:
            start_x = cum_len[load['span_idx']]
            val = load['display_val']
            if load['type'] == 'Uniform':
                fig.add_shape(type="rect", x0=start_x, y0=0, x1=start_x+spans[load['span_idx']], y1=0.2, line_width=0, fillcolor="rgba(255, 0, 0, 0.2)", row=1, col=1)
                fig.add_annotation(x=start_x + spans[load['span_idx']]/2, y=0.25, text=f"w={val:.1f}", showarrow=False, row=1, col=1)
            elif load['type'] == 'Point':
                lx = start_x + load['pos']
                fig.add_annotation(x=lx, y=0, ax=0, ay=-40, text=f"P={val:.1f}", showarrow=True, arrowhead=2, arrowcolor="red", row=1, col=1)

        # 2. SFD
        fig.add_trace(go.Scatter(x=df['x'], y=df['V_disp'], fill='tozeroy', line=dict(color='#D32F2F'), name="Shear"), row=2, col=1)
        
        # 3. BMD
        fig.add_trace(go.Scatter(x=df['x'], y=df['M_disp'], fill='tozeroy', line=dict(color='#1976D2'), name="Moment"), row=3, col=1)

        fig.update_layout(height=700, showlegend=False, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
        fig.update_yaxes(visible=False, row=1, col=1, range=[-0.5, 0.5])
        fig.update_yaxes(autorange="reversed", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)

    # --- DESIGN ---
    with col_design:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader(f"🛠️ Design: {design_code}")
        
        c_mat1, c_mat2 = st.columns(2)
        fc = c_mat1.number_input("f'c (ksc)", value=240.0)
        fy = c_mat2.number_input("fy (ksc)", value=4000.0 if not is_wsd else 3000.0)
        
        c_sec1, c_sec2, c_sec3 = st.columns(3)
        b = c_sec1.number_input("b (cm)", value=25.0)
        h = c_sec2.number_input("h (cm)", value=50.0)
        cover = c_sec3.number_input("Cov (cm)", value=3.0)
        
        bar_map = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
        main_bar = st.selectbox("Main Bar", list(bar_map.keys()), index=1)
        stirrup = st.selectbox("Stirrup", ["RB6", "RB9", "DB10"], index=1)
        
        # Calc
        Mu = df['M_disp'].abs().max()
        Vu = df['V_disp'].abs().max()
        
        fc_mpa, fy_mpa = fc * 0.0981, fy * 0.0981
        b_mm, d_mm = b*10, (h-cover)*10
        
        if "kN" in unit_sys:
            Mu_calc, Vu_calc = Mu*1e6, Vu*1000
        else:
            Mu_calc, Vu_calc = Mu*9.80665*1000, Vu*9.80665
            
        # -- Logic --
        if is_wsd:
            n_ratio = 135 / np.sqrt(fc)
            k = 1 / (1 + (fy*0.5 / (0.45*fc * n_ratio)))
            j = 1 - k/3
            As_req = Mu_calc / (0.5*fy_mpa * j * d_mm)
            shear_res = f"{stirrup} @ 20 cm (Est.)"
        else:
            phi_b, phi_v = params['phi_b'], params['phi_v']
            Rn = Mu_calc / (phi_b * b_mm * d_mm**2)
            m = fy_mpa / (0.85 * fc_mpa)
            term = 1 - (2*m*Rn)/fy_mpa
            if term < 0:
                As_req = 9999
            else:
                rho_req = (1/m)*(1 - np.sqrt(term))
                rho_min = 1.4/fy_mpa
                As_req = max(rho_req, rho_min) * b_mm * d_mm / 100
                
            Vc = 0.17 * np.sqrt(fc_mpa) * b_mm * d_mm
            if Vu_calc > phi_v * Vc: shear_res = f"{stirrup} @ 15 cm"
            else: shear_res = f"{stirrup} @ {int(d_mm/20)} cm (Min)"

        nb = max(2, int(np.ceil(As_req / bar_map[main_bar]))) if As_req != 9999 else 0
        
        st.markdown("---")
        if As_req == 9999:
            st.error("❌ Section Failed! Increase Size.")
        else:
            st.success("✅ Design PASSED")
            c_r1, c_r2 = st.columns(2)
            c_r1.metric("Main Rebar", f"{nb} - {main_bar}")
            c_r2.metric("Stirrups", shear_res)
            
            project_data = {'code': design_code, 'method': params['type'], 'fc': fc, 'fy': fy, 'bar_type': main_bar}
            res_data = {
                'b':b, 'h':h, 'Mu':Mu, 'Vu':Vu, 'u_m':U_M, 'u_f':U_F, 
                'As_req': As_req, 'nb': nb, 'As_prov': nb*bar_map[main_bar], 
                'main_bar': f"{nb}-{main_bar}", 'stirrup': shear_res
            }
            
            html = generate_report_html(project_data, res_data)
            b64 = base64.b64encode(html.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" target="_blank" style="text-decoration:none; color:white; background-color:#D32F2F; padding:12px 20px; border-radius:5px; display:block; text-align:center; font-weight:bold; margin-top:10px;">🖨️ Print Report (PDF)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
