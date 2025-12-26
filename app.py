import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- IMPORT MODULES ---
try:
    from beam_analysis import BeamFiniteElement
    from rc_design import calculate_rc_design
    import input_handler as ui
except ImportError:
    st.error("⚠️ Missing required files. Please ensure beam_analysis.py, rc_design.py, and input_handler.py are in the folder.")
    st.stop()

# ==========================================
# 0. SETUP & GRAPHICS FUNCTIONS
# ==========================================
st.set_page_config(page_title="RC Beam Pro V.Real", layout="wide", page_icon="🏗️")

if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False
if 'res_df' not in st.session_state: st.session_state['res_df'] = None
if 'vis_data' not in st.session_state: st.session_state['vis_data'] = None

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    .main-header { background: linear-gradient(90deg, #1565C0 0%, #0D47A1 100%); color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    .section-header { font-size: 1.3rem; font-weight: bold; color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 5px; margin-top: 30px; margin-bottom: 15px; }
    .input-card { background-color: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #e9ecef; }
    .design-box { background-color: #e3f2fd; border: 2px solid #90caf9; padding: 20px; border-radius: 10px; }
    .calc-log { font-family: 'Courier New', monospace; font-size: 0.9rem; background-color: #f1f1f1; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

def draw_beam_diagram(spans, supports, loads):
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()
    # Beam
    fig.add_trace(go.Scatter(x=[0, sum(spans)], y=[0, 0], mode='lines', line=dict(color='black', width=5), hoverinfo='skip'))
    # Supports
    for i, s in enumerate(supports):
        sx = cum_len[i]
        sym, col, off = ("square", "#D32F2F", 0) if s=="Fix" else (("triangle-up", "#2E7D32", -0.02) if s=="Pin" else ("circle", "#F57C00", -0.02))
        fig.add_trace(go.Scatter(x=[sx], y=[off], mode='markers+text', marker=dict(symbol=sym, size=15, color=col, line=dict(width=2,color='black')), showlegend=False))
        if s == "Roller": fig.add_shape(type="line", x0=sx-0.2, y0=-0.06, x1=sx+0.2, y1=-0.06, line=dict(color="black", width=2))
    # Dimensions
    for i, sp in enumerate(spans):
        fig.add_annotation(x=cum_len[i]+sp/2, y=-0.15, text=f"{sp}m", showarrow=False, font=dict(color="blue", size=14))
        
    fig.update_layout(height=250, margin=dict(l=20,r=20,t=20,b=20), yaxis=dict(visible=False, range=[-0.4, 0.4]))
    return fig

def draw_section(b, h, cov, nb, bd, sd, name):
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=3), fillcolor="#F5F5F5")
    fig.add_shape(type="rect", x0=cov, y0=cov, x1=b-cov, y1=h-cov, line=dict(color="red", width=2)) # Stirrup
    if nb > 0:
        gap = (b - 2*cov - 2*sd/10 - bd/10)/(nb-1) if nb>1 else 0
        y = cov + sd/10 + bd/20
        xs = [cov + sd/10 + bd/20 + i*gap for i in range(nb)]
        fig.add_trace(go.Scatter(x=xs, y=[y]*nb, mode='markers', marker=dict(size=bd*1.5, color='blue', line=dict(width=1,color='black'))))
        fig.add_annotation(x=b/2, y=y, ax=0, ay=40, text=f"{nb}-{name}", showarrow=True, arrowcolor="blue")
    
    fig.add_annotation(x=b/2, y=-3, text=f"b={b}", showarrow=False)
    fig.add_annotation(x=-3, y=h/2, text=f"h={h}", textangle=-90, showarrow=False)
    fig.update_layout(width=300, height=300, xaxis=dict(visible=False, range=[-5, b+5]), yaxis=dict(visible=False, range=[-5, h+5]), margin=dict(l=10,r=10,t=10,b=10))
    return fig

# ==========================================
# 1. MAIN APPLICATION
# ==========================================
st.markdown('<div class="main-header"><h2>🏗️ RC Beam Pro V.Real Analysis</h2></div>', unsafe_allow_html=True)

# 1.1 Input Handling
design_code, method, fact_dl, fact_ll, unit_sys = ui.render_sidebar()

st.markdown('<div class="section-header">1️⃣ Structure & Loads</div>', unsafe_allow_html=True)
c_geo, c_load = st.columns([1, 1.5])
with c_geo:
    n_span, spans, supports = ui.render_geometry_input()
with c_load:
    loads_input = ui.render_loads_input(n_span, spans, fact_dl, fact_ll, unit_sys)

# 1.2 Analysis
st.markdown('<div class="section-header">2️⃣ Analysis (Finite Element)</div>', unsafe_allow_html=True)
if st.button("🚀 Calculate Analysis", type="primary"):
    # ใช้ BeamFiniteElement ตัวใหม่
    solver = BeamFiniteElement(spans, supports, loads_input)
    success, msg = solver.solve()
    
    if success:
        st.session_state['res_df'] = solver.get_internal_forces()
        st.session_state['vis_data'] = (spans, supports, loads_input)
        st.session_state['analyzed'] = True
        st.rerun()
    else:
        st.error(f"Analysis Failed: {msg}")

# 1.3 Result & Design
if st.session_state['analyzed']:
    vis_data = st.session_state['vis_data']
    df = st.session_state['res_df'].copy()
    
    # Check Units for Display (Engine is unitless/consistent, we map to UI unit)
    # Note: Engine คำนวณตาม Unit ที่ใส่เข้าไป (ถ้าใส่ kN, m ผลลัพธ์คือ kN, kN-m)
    # แต่ต้องระวังเรื่อง MKS (ใส่ kg, m ผลลัพธ์คือ kg, kg-m)
    
    u_f, u_m = ("kN", "kN-m") if "kN" in unit_sys else ("kg", "kg-m")
    
    st.plotly_chart(draw_beam_diagram(*vis_data), use_container_width=True)
    
    c1, c2 = st.columns(2)
    max_M = df['moment'].abs().max()
    max_V = df['shear'].abs().max()
    c1.metric(f"Max Mu ({u_m})", f"{max_M:.2f}")
    c2.metric(f"Max Vu ({u_f})", f"{max_V:.2f}")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f"Shear ({u_f})", f"Moment ({u_m})"))
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='red'), name="V"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='blue'), name="M"), row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # 1.4 Design Section
    st.markdown('<div class="section-header">3️⃣ RC Design & Calculation</div>', unsafe_allow_html=True)
    fc, fy, b, h, cov, m_bar, s_bar = ui.render_design_input(unit_sys)
    
    bar_areas = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
    bar_dias = {k: int(k[2:]) for k in bar_areas}
    stir_areas = {'RB6':0.28, 'RB9':0.64, 'DB10':0.78}
    stir_dias = {'RB6':6, 'RB9':9, 'DB10':10}

    res = calculate_rc_design(max_M, max_V, fc, fy, b, h, cov, method, unit_sys, bar_areas[m_bar], stir_areas[s_bar])
    
    st.markdown('<div class="design-box">', unsafe_allow_html=True)
    c_res, c_draw = st.columns(2)
    with c_res:
        st.subheader("🎯 Result")
        if res['nb'] == 0: st.error(res['msg_flex'])
        else:
            st.success(res['msg_flex'])
            st.write(f"**Main:** {res['nb']}-{m_bar} (As: {res['As_req']:.2f} cm²)")
            st.divider()
            st.write(f"**Stirrup:** {s_bar} {res['stirrup_text']}")
            st.caption(res['msg_shear'])
            
            # Show Detailed Calculations
            with st.expander("📝 View Detailed Calculation"):
                st.markdown("```text\n" + "\n".join(res['logs']) + "\n```")
                
    with c_draw:
        st.plotly_chart(draw_section(b, h, cov, res['nb'], bar_dias[m_bar], stir_dias[s_bar], m_bar), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
