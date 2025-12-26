import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- IMPORT MODULES ---
try:
    from beam_analysis import BeamFiniteElement
    from rc_design import calculate_rc_design
    import input_handler as ui
except ImportError:
    st.error("⚠️ Missing required files.")
    st.stop()

# ==========================================
# SETUP
# ==========================================
st.set_page_config(page_title="RC Beam Pro Ultimate", layout="wide", page_icon="🏗️")

if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False
if 'res_df' not in st.session_state: st.session_state['res_df'] = None
if 'vis_data' not in st.session_state: st.session_state['vis_data'] = None

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    .main-header { background: #1565C0; color: white; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 20px; }
    .section-header { border-left: 5px solid #1565C0; padding-left: 10px; font-size: 1.3rem; font-weight: bold; color: #1565C0; margin-top: 30px; margin-bottom: 15px; background-color: #E3F2FD; padding: 10px; border-radius: 0 5px 5px 0; }
</style>
""", unsafe_allow_html=True)

# --- DRAWING FUNCTIONS ---

def draw_reinforcement_profile(spans, design_results, m_bar, s_bar):
    """ วาดรูปด้านข้างคาน (Side View) แยกเหล็กตามช่วง Span จริง """
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()

    # 1. Concrete Body (Gray Box)
    fig.add_shape(type="rect", x0=0, y0=-0.5, x1=total_len, y1=0.5, 
                  line=dict(color="black", width=2), fillcolor="#F5F5F5", layer="below")
    
    # 2. Iterate Spans to draw Rebars
    for i in range(len(spans)):
        x0 = cum_len[i]
        x1 = cum_len[i+1]
        res = design_results[i]
        
        # Bottom Bar (Main) - Blue Line (Inset slightly from support)
        # วาดเส้นแยกช่วงชัดเจน ไม่ลากยาว
        fig.add_trace(go.Scatter(
            x=[x0 + 0.1, x1 - 0.1], 
            y=[-0.35, -0.35],
            mode='lines',
            line=dict(color='#1565C0', width=5), # เส้นหนา สีน้ำเงิน
            name=f"Bott. Span {i+1}", showlegend=False, hoverinfo='text',
            text=f"Span {i+1}: {res['nb']}-{m_bar}"
        ))

        # Top Bar (Hanger/Negative) - Red Line
        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[0.35, 0.35],
            mode='lines',
            line=dict(color='#D32F2F', width=3), # เส้นบางกว่า สีแดง
            name=f"Top Span {i+1}", showlegend=False, hoverinfo='skip'
        ))
        
        # Label Box (Center of Span)
        mid_x = (x0 + x1) / 2
        fig.add_annotation(
            x=mid_x, y=0,
            text=f"<b>SPAN {i+1}</b><br><span style='color:#1565C0'>{res['nb']}-{m_bar}</span><br><span style='color:#D32F2F'>{s_bar} {res['stirrup_text']}</span>",
            showarrow=False,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#999"
        )

    # 3. Supports (Triangles)
    for x in cum_len:
         fig.add_trace(go.Scatter(x=[x], y=[-0.6], mode='markers', 
                                  marker=dict(symbol="triangle-up", size=15, color="#333"), 
                                  showlegend=False, hoverinfo='skip'))

    fig.update_layout(
        title="🏗️ Reinforcement Profile (Side View)",
        height=280,
        xaxis=dict(title="Distance (m)", range=[-0.5, total_len+0.5], showgrid=True, dtick=1),
        yaxis=dict(visible=False, range=[-1, 1]),
        margin=dict(l=20,r=20,t=40,b=20), plot_bgcolor='white'
    )
    return fig

def draw_section_engineering(b_cm, h_cm, cov_cm, nb, bd_mm, stir_d_mm, main_name, stir_name, s_val_mm, title):
    """ วาด Cross Section แบบวิศวกรรม (วงกลมตามจำนวนจริง) """
    fig = go.Figure()
    
    # Scale variables
    bd_cm = bd_mm / 10.0
    sd_cm = stir_d_mm / 10.0
    
    # 1. Concrete Outline
    fig.add_shape(type="rect", x0=0, y0=0, x1=b_cm, y1=h_cm, 
                  line=dict(color="black", width=3), fillcolor="#FAFAFA")
    
    # 2. Stirrup (Red Loop)
    sx0, sy0 = cov_cm, cov_cm
    sx1, sy1 = b_cm - cov_cm, h_cm - cov_cm
    fig.add_shape(type="rect", x0=sx0, y0=sy0, x1=sx1, y1=sy1,
                  line=dict(color="#C62828", width=3), fillcolor="rgba(0,0,0,0)")
    
    # 3. Main Bars (Blue Circles) - Loop วาดตามจำนวนจริง
    if nb > 0:
        # หาพื้นที่วางเหล็ก (Effective Width)
        start_x = cov_cm + sd_cm + bd_cm/2
        end_x = b_cm - (cov_cm + sd_cm + bd_cm/2)
        y_pos = cov_cm + sd_cm + bd_cm/2
        
        if nb == 1:
            x_positions = [(start_x + end_x)/2]
        else:
            x_positions = np.linspace(start_x, end_x, nb)
            
        # วาดทีละวง
        for xp in x_positions:
            fig.add_shape(type="circle",
                x0=xp - bd_cm/2, y0=y_pos - bd_cm/2,
                x1=xp + bd_cm/2, y1=y_pos + bd_cm/2,
                line_color="black", fillcolor="#1565C0"
            )
            
    # 4. Hanger Bars (Top) - Dummy 2 bars
    y_top = h_cm - (cov_cm + sd_cm + bd_cm/2)
    # Left Top
    fig.add_shape(type="circle",
        x0=start_x - bd_cm/2, y0=y_top - bd_cm/2,
        x1=start_x + bd_cm/2, y1=y_top + bd_cm/2,
        line_color="black", fillcolor="#B0BEC5"
    )
    # Right Top
    fig.add_shape(type="circle",
        x0=end_x - bd_cm/2, y0=y_top - bd_cm/2,
        x1=end_x + bd_cm/2, y1=y_top + bd_cm/2,
        line_color="black", fillcolor="#B0BEC5"
    )

    # Annotations
    fig.add_annotation(x=b_cm/2, y=-h_cm*0.1, text=f"b = {b_cm*10:.0f} mm", showarrow=False)
    fig.add_annotation(x=-b_cm*0.15, y=h_cm/2, text=f"h = {h_cm*10:.0f} mm", textangle=-90, showarrow=False)
    
    # Legend Text below
    fig.add_annotation(x=b_cm/2, y=h_cm/2, text=f"Main: {nb}-{main_name}", font=dict(color="#1565C0", size=14, weight="bold"), showarrow=False, yshift=-15)
    fig.add_annotation(x=b_cm/2, y=h_cm/2, text=f"Stir: {stir_name}@{s_val_mm}mm", font=dict(color="#C62828", size=12), showarrow=False, yshift=15)

    fig.update_layout(
        title=dict(text=title, x=0.5), width=300, height=350,
        xaxis=dict(visible=False, range=[-b_cm*0.4, b_cm*1.4]), 
        yaxis=dict(visible=False, range=[-h_cm*0.2, h_cm*1.2]),
        margin=dict(l=10,r=10,t=40,b=10), plot_bgcolor='white'
    )
    return fig

# ==========================================
# MAIN APP
# ==========================================
st.markdown('<div class="main-header"><h2>🏗️ RC Beam Pro Ultimate</h2></div>', unsafe_allow_html=True)

# 1. Inputs
design_code, method, fact_dl, fact_ll, unit_sys = ui.render_sidebar()
c_geo, c_load = st.columns([1, 1.5])
with c_geo: n_span, spans, supports = ui.render_geometry_input()
with c_load: loads_input = ui.render_loads_input(n_span, spans, fact_dl, fact_ll, unit_sys)

# 2. Calculate Button
if st.button("🚀 Calculate Analysis & Design", type="primary"):
    solver = BeamFiniteElement(spans, supports, loads_input)
    success, msg = solver.solve()
    if success:
        st.session_state['res_df'] = solver.get_internal_forces()
        st.session_state['vis_data'] = (spans, supports, loads_input)
        st.session_state['analyzed'] = True
    else:
        st.error(f"Analysis Failed: {msg}")

# 3. Results Display
if st.session_state['analyzed'] and st.session_state['res_df'] is not None:
    vis_data = st.session_state['vis_data']
    df = st.session_state['res_df'].copy()
    vis_spans = vis_data[0]
    vis_supports = vis_data[1]
    total_len = sum(vis_spans)
    cum_len = [0] + list(np.cumsum(vis_spans))
    
    u_force = "kN" if "kN" in unit_sys else "kg"
    u_moment = "kN-m" if "kN" in unit_sys else "kg-m"

    # --- PART 1: ANALYSIS GRAPHS (FIXED) ---
    st.markdown('<div class="section-header">1️⃣ Analysis Diagrams</div>', unsafe_allow_html=True)
    
    # ใช้วิธี subplot แบบง่ายแต่ชัวร์ เพื่อไม่ให้กราฟหาย
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.08, 
                        row_heights=[0.15, 0.4, 0.4],
                        subplot_titles=("Beam Model", f"Shear Force ({u_force})", f"Bending Moment ({u_moment})"))

    # Row 1: Beam Model
    fig.add_shape(type="rect", x0=0, y0=-0.1, x1=total_len, y1=0.1, fillcolor="#E0E0E0", line=dict(color="black"), row=1, col=1)
    # Supports
    for i, s in enumerate(vis_supports):
        if s != "None":
            fig.add_trace(go.Scatter(x=[cum_len[i]], y=[-0.15], mode='markers', marker=dict(symbol="triangle-up", size=12, color="green"), hoverinfo='name', name=f"{s}"), row=1, col=1)

    # Row 2: Shear
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#E53935', width=2), name="Shear"), row=2, col=1)
    
    # Row 3: Moment
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1E88E5', width=2), name="Moment"), row=3, col=1)

    # Layout: Force X-Axis Range to match perfectly
    fig.update_layout(height=700, showlegend=False, hovermode="x unified", margin=dict(l=50, r=20, t=40, b=40))
    common_x_range = [-0.5, total_len + 0.5]
    
    fig.update_xaxes(range=common_x_range, row=1, col=1, visible=False) # Hide axis on beam model
    fig.update_xaxes(range=common_x_range, row=2, col=1, showgrid=True)
    fig.update_xaxes(range=common_x_range, row=3, col=1, showgrid=True, title_text="Distance (m)", tickmode='linear', dtick=1.0)
    
    fig.update_yaxes(visible=False, row=1, col=1) # Hide Y on beam model
    
    st.plotly_chart(fig, use_container_width=True)

    # --- PART 2: DESIGN ---
    st.markdown('<div class="section-header">2️⃣ Structural Design Results</div>', unsafe_allow_html=True)
    
    fc, fy, b_cm, h_cm, cov_cm, m_bar, s_bar, man_s = ui.render_design_input(unit_sys)
    
    # Database
    bar_areas = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
    stir_areas = {'RB6':0.28, 'RB9':0.64, 'DB10':0.78, 'DB12':1.13}
    
    # Mappings for drawing
    bd_map = {'DB12':12, 'DB16':16, 'DB20':20, 'DB25':25, 'DB28':28}
    sd_map = {'RB6':6, 'RB9':9, 'DB10':10, 'DB12':12}

    span_results = []
    for i in range(n_span):
        x_start, x_end = cum_len[i], cum_len[i+1]
        span_df = df[(df['x'] >= x_start) & (df['x'] <= x_end)]
        m_span = span_df['moment'].abs().max()
        v_span = span_df['shear'].abs().max()
        
        # Design Calculation
        res = calculate_rc_design(m_span, v_span, fc, fy, b_cm, h_cm, cov_cm, method, unit_sys, bar_areas[m_bar], stir_areas[s_bar], man_s)
        res['id'] = i+1
        span_results.append(res)

    # 2.1 Profile View
    st.plotly_chart(draw_reinforcement_profile(vis_spans, span_results, m_bar, s_bar), use_container_width=True)

    # 2.2 Section View
    st.markdown("#### 🔍 Section Details (per Span)")
    tabs = st.tabs([f"Span {res['id']}" for res in span_results])
    
    for i, tab in enumerate(tabs):
        res = span_results[i]
        with tab:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"**Section Span {res['id']}**")
                # เรียกฟังก์ชันวาดวงกลมเหล็ก (Pass ค่า s_value_mm ไปแสดง)
                fig_sec = draw_section_engineering(
                    b_cm, h_cm, cov_cm, res['nb'], 
                    bd_map[m_bar], sd_map[s_bar], 
                    m_bar, s_bar, res['s_value_mm'], 
                    ""
                )
                st.plotly_chart(fig_sec, use_container_width=True, key=f"sec_plot_{i}")
                
            with c2:
                with st.expander("Show Calculation Logs", expanded=True):
                    for line in res['logs']:
                        st.markdown(line)
