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
# 0. SETUP
# ==========================================
st.set_page_config(page_title="RC Beam Pro Ultimate", layout="wide", page_icon="🏗️")

if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False
if 'res_df' not in st.session_state: st.session_state['res_df'] = None
if 'vis_data' not in st.session_state: st.session_state['vis_data'] = None

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    .main-header { background: linear-gradient(90deg, #1565C0 0%, #0D47A1 100%); color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    .section-header { font-size: 1.3rem; font-weight: bold; color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 5px; margin-top: 30px; margin-bottom: 15px; }
    .design-box { background-color: #e3f2fd; border: 2px solid #90caf9; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- GRAPHIC FUNCTIONS ---

def draw_beam_model(spans, supports, loads):
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()
    
    # Beam
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', name='Beam', line=dict(color='black', width=6), hoverinfo='name+x'))

    # Supports
    for i, s in enumerate(supports):
        sx = cum_len[i]
        if s == "None": continue
        if s == "Fix": sym, col, off = "square", "#D32F2F", 0
        elif s == "Pin": sym, col, off = "triangle-up", "#2E7D32", -0.02
        else: sym, col, off = "circle", "#F57C00", -0.02
        fig.add_trace(go.Scatter(x=[sx], y=[off], mode='markers', marker=dict(symbol=sym, size=15, color=col, line=dict(width=2,color='black')), showlegend=False))
        
    # Loads
    for load in loads:
        start_x = cum_len[load['span_idx']]
        val = load['display_val']
        if load['type'] == 'Uniform':
            end_x = start_x + spans[load['span_idx']]
            fig.add_shape(type="rect", x0=start_x, y0=0, x1=end_x, y1=0.2, line=dict(width=0), fillcolor="rgba(255, 87, 34, 0.2)")
            fig.add_annotation(x=(start_x+end_x)/2, y=0.2, text=f"w={val:.2f}", showarrow=True, ay=-20)
        elif load['type'] == 'Point':
            lx = start_x + load['pos']
            fig.add_annotation(x=lx, y=0, text=f"P={val:.2f}", showarrow=True, ay=-40, arrowhead=2, arrowcolor="red")

    fig.update_layout(height=250, margin=dict(l=30,r=30,t=20,b=20), yaxis=dict(visible=False, range=[-0.3, 0.4]), xaxis=dict(range=[-0.5, total_len+0.5]))
    return fig

def draw_reinforcement_profile(spans, design_results, m_bar, s_bar):
    """ วาดรูปคานพร้อมเส้นเหล็กและ Label แยกแต่ละช่วง """
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()

    # Draw Beam Outline
    fig.add_shape(type="rect", x0=0, y0=-1, x1=total_len, y1=1, line=dict(color="black", width=2), fillcolor="white")
    
    # Draw Supports
    for x in cum_len:
        fig.add_shape(type="line", x0=x, y0=-1.5, x1=x, y1=-1, line=dict(color="black", width=2))
        fig.add_trace(go.Scatter(x=[x], y=[-1.5], mode='markers', marker=dict(symbol="triangle-up", size=10, color="grey"), showlegend=False))

    # Loop Spans and Add Labels
    for i, res in enumerate(design_results):
        start_x = cum_len[i]
        end_x = cum_len[i+1]
        mid_x = (start_x + end_x) / 2
        
        nb = res['nb']
        stir = res['stirrup_text']
        
        # Main Rebar Text (Bottom)
        txt_main = f"{nb}-{m_bar}" if nb > 0 else "N/A"
        fig.add_annotation(x=mid_x, y=0, text=f"<b>Span {i+1}</b><br>{txt_main}<br>Stir: {s_bar} {stir}", 
                           showarrow=False, font=dict(size=12, color="#1565C0"), bgcolor="#E3F2FD", bordercolor="#1565C0")
        
        # Illustrate Rebar Line (Schematic)
        fig.add_shape(type="line", x0=start_x+0.2, y0=-0.6, x1=end_x-0.2, y1=-0.6, line=dict(color="blue", width=3))

    fig.update_layout(title="🏗️ Beam Reinforcement Profile (Schematic)", height=200, 
                      yaxis=dict(visible=False, range=[-2, 2]), xaxis=dict(visible=True, title="Distance (m)"),
                      margin=dict(l=20,r=20,t=40,b=20), plot_bgcolor='white')
    return fig

def draw_section_beautiful(b, h, cov, nb, bd_mm, sd_mm, main_name, stir_name, s_val, title):
    fig = go.Figure()
    bd, sd = bd_mm/10, sd_mm/10
    
    # Concrete
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=3), fillcolor="#EEEEEE")
    # Stirrup
    inset = cov
    fig.add_shape(type="rect", x0=inset, y0=inset, x1=b-inset, y1=h-inset, line=dict(color="#C62828", width=3))
    
    # Main Bars
    if nb > 0:
        eff_w = b - 2*cov - 2*sd - bd
        gap = eff_w / (nb - 1) if nb > 1 else 0
        y_pos = cov + sd + bd/2
        xs = [cov + sd + bd/2 + i*gap for i in range(nb)]
        fig.add_trace(go.Scatter(x=xs, y=[y_pos]*nb, mode='markers', marker=dict(size=bd_mm*1.8, color='#1565C0', line=dict(width=1,color='black')), showlegend=False))
        fig.add_annotation(x=b/2, y=y_pos, text=f"<b>{nb}-{main_name}</b>", ay=40, arrowhead=2, arrowcolor="blue")

    # Labels
    fig.add_annotation(x=b/2, y=h/2, text=f"{stir_name}@{s_val}cm", font=dict(color="#C62828"), bgcolor="white")
    
    fig.update_layout(title=title, width=300, height=300, 
                      xaxis=dict(visible=False, range=[-5, b+5]), yaxis=dict(visible=False, range=[-5, h+5]),
                      margin=dict(l=10,r=10,t=40,b=10))
    return fig

# ==========================================
# 1. MAIN APPLICATION
# ==========================================
st.markdown('<div class="main-header"><h2>🏗️ RC Beam Pro Ultimate</h2></div>', unsafe_allow_html=True)

# 1.1 Sidebar & Inputs
design_code, method, fact_dl, fact_ll, unit_sys = ui.render_sidebar()
c_geo, c_load = st.columns([1, 1.5])
with c_geo: n_span, spans, supports = ui.render_geometry_input()
with c_load: loads_input = ui.render_loads_input(n_span, spans, fact_dl, fact_ll, unit_sys)

# 1.2 Analysis Button
if st.button("🚀 Run Analysis & Design", type="primary"):
    solver = BeamFiniteElement(spans, supports, loads_input)
    success, msg = solver.solve()
    if success:
        st.session_state['res_df'] = solver.get_internal_forces()
        st.session_state['vis_data'] = (spans, supports, loads_input)
        st.session_state['analyzed'] = True
    else:
        st.error(f"Analysis Failed: {msg}")

# 1.3 Results
if st.session_state['analyzed']:
    vis_data = st.session_state['vis_data']
    df = st.session_state['res_df'].copy()
    vis_spans = vis_data[0]
    total_len = sum(vis_spans)
    u_f, u_m = ("kN", "kN-m") if "kN" in unit_sys else ("kg", "kg-m")

    # --- PART 1: GRAPHS ---
    st.markdown('<div class="section-header">1️⃣ Analysis Results</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_beam_model(*vis_data), use_container_width=True)

    # Prepare Graph Data with Smart Padding for Labels (แก้ปัญหาเลขบัง)
    y_v_max, y_v_min = df['shear'].max(), df['shear'].min()
    y_m_max, y_m_min = df['moment'].max(), df['moment'].min()
    
    # Calculate Padding (20% of range)
    v_pad = (abs(y_v_max) + abs(y_v_min)) * 0.2
    m_pad = (abs(y_m_max) + abs(y_m_min)) * 0.2
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                        subplot_titles=(f"Shear ({u_f})", f"Moment ({u_m})"))
    
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#E53935'), name="Shear"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1E88E5'), name="Moment"), row=2, col=1)

    # Max/Min Labeling (Smart Positioning)
    for col, data, y_range in [(1, df['shear'], [y_v_min - v_pad, y_v_max + v_pad]), 
                               (2, df['moment'], [y_m_min - m_pad, y_m_max + m_pad])]:
        mx, mn = data.max(), data.min()
        mx_idx, mn_idx = data.idxmax(), data.idxmin()
        
        for val, idx in [(mx, mx_idx), (mn, mn_idx)]:
            x_pos = df.iloc[idx]['x']
            # Dynamic Text Position
            txt_pos = "top center" if val >= 0 else "bottom center"
            # If near edges, shift inside to avoid cutoff
            if x_pos < total_len * 0.05: txt_pos = "middle right"
            elif x_pos > total_len * 0.95: txt_pos = "middle left"
            
            fig.add_trace(go.Scatter(x=[x_pos], y=[val], mode='markers+text',
                                     text=[f"{val:.2f}"], textposition=txt_pos,
                                     marker=dict(color='black', size=6), showlegend=False), row=col, col=1)
        # Apply Manual Range Padding
        fig.update_yaxes(range=y_range, row=col, col=1)

    fig.update_xaxes(range=[-0.5, total_len + 0.5])
    fig.update_layout(height=600, margin=dict(l=50, r=50, t=50, b=50))
    st.plotly_chart(fig, use_container_width=True)

    # --- PART 2: DESIGN PER SPAN ---
    st.markdown('<div class="section-header">2️⃣ Design & Detailing (Per Span)</div>', unsafe_allow_html=True)
    fc, fy, b, h, cov, m_bar, s_bar, man_s = ui.render_design_input(unit_sys)
    
    bar_areas = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
    bar_dias = {k: int(k[2:]) for k in bar_areas}
    stir_areas = {'RB6':0.28, 'RB9':0.64, 'DB10':0.78, 'DB12':1.13} # Added DB12
    stir_dias = {k: (int(k[2:]) if 'DB' in k else int(k[2:])) for k in stir_areas} # RB6->6, DB12->12

    # Loop Each Span for Design
    cum_len = [0] + list(np.cumsum(vis_spans))
    span_results = []
    
    for i in range(n_span):
        x_start, x_end = cum_len[i], cum_len[i+1]
        # Filter forces in this span
        span_df = df[(df['x'] >= x_start) & (df['x'] <= x_end)]
        
        # Get Envelope Forces for this span
        m_span = span_df['moment'].abs().max()
        v_span = span_df['shear'].abs().max()
        
        res = calculate_rc_design(m_span, v_span, fc, fy, b, h, cov, method, unit_sys, 
                                  bar_areas[m_bar], stir_areas[s_bar], man_s)
        res['span_id'] = i + 1
        res['m_design'] = m_span
        res['v_design'] = v_span
        span_results.append(res)

    # 2.1 Visual Profile
    st.subheader("📍 Reinforcement Profile")
    st.plotly_chart(draw_reinforcement_profile(vis_spans, span_results, m_bar, s_bar), use_container_width=True)

    # 2.2 Detailed Cards
    st.subheader("📋 Span Details")
    cols = st.columns(n_span)
    
    for i, res in enumerate(span_results):
        with cols[i]:
            st.markdown(f'<div class="design-box"><h4>Span {res["span_id"]}</h4>', unsafe_allow_html=True)
            st.caption(f"Mu: {res['m_design']:.2f} | Vu: {res['v_design']:.2f}")
            
            if res['nb'] == 0:
                st.error("Fail: " + res['msg_flex'])
            else:
                st.write(f"**Main:** {res['nb']}-{m_bar}")
                st.write(f"**Stirrup:** {s_bar} {res['stirrup_text']}")
                st.caption(f"Shear: {res['msg_shear']}")
            
            # Draw Section for this span
            fig_sec = draw_section_beautiful(b, h, cov, res['nb'], bar_dias[m_bar], stir_dias[s_bar], m_bar, s_bar, res['s_value'], f"Sec-Span {i+1}")
            st.plotly_chart(fig_sec, use_container_width=True)
            
            with st.expander("Calc Log"):
                st.code("\n".join(res['logs']))
            st.markdown('</div>', unsafe_allow_html=True)
