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

# CSS for better readability
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    .main-header { background: linear-gradient(90deg, #1976D2 0%, #0D47A1 100%); color: white; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 20px; }
    .section-header { font-size: 1.4rem; font-weight: bold; color: #1565C0; border-bottom: 2px solid #BBDEFB; padding-bottom: 5px; margin-top: 30px; margin-bottom: 15px; }
    .design-card { background-color: #F5F7F9; border: 1px solid #E3E8EE; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .calc-log { background-color: #FFFFFF; border-left: 4px solid #1976D2; padding: 15px; font-size: 0.95rem; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# --- GRAPHIC FUNCTIONS ---

def draw_reinforcement_profile(spans, design_results, m_bar, s_bar):
    """ วาดรูปด้านข้างคาน พร้อมเส้นเหล็กวิ่งยาว (Top/Bottom Lines) """
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()

    # 1. Concrete Body (Outline)
    fig.add_shape(type="rect", x0=0, y0=-0.5, x1=total_len, y1=0.5, 
                  line=dict(color="black", width=2), fillcolor="white")
    
    # 2. Rebar Lines (Visual Representation)
    # เหล็กบน (Top Hanger) - สีเทา/เส้นประ หรือทึบถ้าจำเป็น (สมมติใส่ตลอด)
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0.35, 0.35], mode='lines', 
                             name="Top Bar", line=dict(color='#B0BEC5', width=3, dash='solid'), hoverinfo='skip'))
    
    # เหล็กล่าง (Main Bottom) - สีน้ำเงินเข้ม
    fig.add_trace(go.Scatter(x=[0, total_len], y=[-0.35, -0.35], mode='lines', 
                             name="Bottom Bar", line=dict(color='#1565C0', width=4), hoverinfo='skip'))

    # 3. Supports & Stirrup Indication (Schematic Vertical lines)
    for i in range(len(cum_len)-1):
        x_start, x_end = cum_len[i], cum_len[i+1]
        mid = (x_start + x_end)/2
        res = design_results[i]
        
        # Label Box
        fig.add_annotation(x=mid, y=0, 
                           text=f"<b>Span {i+1}</b><br><span style='color:blue'>{res['nb']}-{m_bar} (Bott)</span><br><span style='color:red'>{s_bar} {res['stirrup_text']}</span>", 
                           showarrow=False, bgcolor="#E3F2FD", bordercolor="#1565C0", borderwidth=1, font=dict(size=11))
    
    # Supports
    for x in cum_len:
         fig.add_trace(go.Scatter(x=[x], y=[-0.6], mode='markers', 
                                  marker=dict(symbol="triangle-up", size=12, color="#424242"), showlegend=False, hoverinfo='skip'))

    fig.update_layout(title="🏗️ Reinforcement Profile (Longitudinal)", height=220, 
                      yaxis=dict(visible=False, range=[-1, 1]), 
                      xaxis=dict(title="Distance (m)", showgrid=False, range=[-0.2, total_len+0.2]),
                      margin=dict(l=20,r=20,t=40,b=20), plot_bgcolor='white', showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def draw_section_beautiful(b, h, cov, nb, bd_mm, sd_mm, main_name, stir_name, s_val, title):
    fig = go.Figure()
    bd, sd = bd_mm/10, sd_mm/10
    
    # Concrete
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=2), fillcolor="#F5F5F5")
    
    # Stirrup (Red Loop)
    inset = cov
    fig.add_shape(type="rect", x0=inset, y0=inset, x1=b-inset, y1=h-inset, 
                  line=dict(color="#D32F2F", width=3), opacity=1)
    
    # Main Bars (Blue Circles)
    if nb > 0:
        eff_w = b - 2*cov - 2*sd - bd
        gap = eff_w / (nb - 1) if nb > 1 else 0
        y_pos = cov + sd + bd/2
        xs = [cov + sd + bd/2 + i*gap for i in range(nb)]
        
        fig.add_trace(go.Scatter(x=xs, y=[y_pos]*nb, mode='markers', 
                                 marker=dict(size=bd_mm*2, color='#1976D2', line=dict(width=1.5, color='black')), 
                                 name="Main Bar", showlegend=False))
        
    # Hanger Bars (Top - Dummy 2 bars)
    y_top = h - (cov + sd + bd/2)
    fig.add_trace(go.Scatter(x=[cov+sd+bd/2, b-(cov+sd+bd/2)], y=[y_top, y_top], mode='markers',
                             marker=dict(size=bd_mm*1.2, color='#B0BEC5', line=dict(width=1, color='black')),
                             name="Hanger Bar", showlegend=False))

    # Dimensions
    fig.add_annotation(x=b/2, y=-3, text=f"b={b}", showarrow=False, font=dict(size=12))
    fig.add_annotation(x=-3, y=h/2, text=f"h={h}", textangle=-90, showarrow=False, font=dict(size=12))
    fig.add_annotation(x=b/2, y=h/2, text=f"Stirrup<br>{stir_name}@{s_val}", font=dict(color="#C62828", size=12))

    fig.update_layout(title=dict(text=title, font=dict(size=14)), width=280, height=280, 
                      xaxis=dict(visible=False, range=[-8, b+8]), yaxis=dict(visible=False, range=[-8, h+8]),
                      margin=dict(l=10,r=10,t=40,b=10), plot_bgcolor='white')
    return fig

# ==========================================
# 1. MAIN APPLICATION
# ==========================================
st.markdown('<div class="main-header"><h2>🏗️ RC Beam Pro Ultimate</h2></div>', unsafe_allow_html=True)

# 1.1 Inputs
design_code, method, fact_dl, fact_ll, unit_sys = ui.render_sidebar()
c_geo, c_load = st.columns([1, 1.5])
with c_geo: n_span, spans, supports = ui.render_geometry_input()
with c_load: loads_input = ui.render_loads_input(n_span, spans, fact_dl, fact_ll, unit_sys)

# 1.2 Analysis Button
if st.button("🚀 Calculate Analysis & Design", type="primary"):
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
    
    # Unit Strings
    u_force = "kN" if "kN" in unit_sys else "kg"
    u_moment = "kN-m" if "kN" in unit_sys else "kg-m"
    u_dist = "m"

    # --- PART 1: ADVANCED GRAPHS (LINKED) ---
    st.markdown('<div class="section-header">1️⃣ Analysis Results (Diagrams)</div>', unsafe_allow_html=True)
    
    # Create Subplots with Shared X-Axis
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Beam Model", f"Shear Force ({u_force})", f"Bending Moment ({u_moment})"),
                        row_heights=[0.2, 0.4, 0.4])

    # 1. Beam Line (Hidden Y, just for reference)
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', line=dict(color='black', width=4), hoverinfo='skip'), row=1, col=1)
    # Add Supports to Graph 1
    cum_len = [0] + list(np.cumsum(vis_spans))
    for i, s in enumerate(supports):
        if s != "None":
            sx = cum_len[i]
            fig.add_trace(go.Scatter(x=[sx], y=[0], mode='markers', marker=dict(symbol="triangle-up", size=10, color="green"), showlegend=False, hoverinfo='skip'), row=1, col=1)

    # 2. Shear
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#E53935', width=2), name="Shear"), row=2, col=1)
    
    # 3. Moment
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1E88E5', width=2), name="Moment"), row=3, col=1)

    # Layout Updates: Hover Unified, Spikelines, Labels
    fig.update_layout(
        height=700, 
        hovermode="x unified",  # Key: Links all graphs vertically
        plot_bgcolor='white',
        margin=dict(l=50, r=50, t=30, b=30),
        showlegend=False
    )
    
    # Axis Styling (Grid, Spikes, Titles)
    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, mirror=True, gridcolor='#EEE', row=3, col=1, title_text=f"Distance ({u_dist})")
    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, mirror=True, gridcolor='#EEE', row=2, col=1)
    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, mirror=True, row=1, col=1)
    
    fig.update_yaxes(title_text=f"V ({u_force})", showgrid=True, gridcolor='#EEE', row=2, col=1)
    fig.update_yaxes(title_text=f"M ({u_moment})", showgrid=True, gridcolor='#EEE', row=3, col=1)
    fig.update_yaxes(visible=False, row=1, col=1) # Hide Y for Beam Model

    st.plotly_chart(fig, use_container_width=True)

    # --- PART 2: DESIGN & DETAILING ---
    st.markdown('<div class="section-header">2️⃣ Design & Detailing</div>', unsafe_allow_html=True)
    
    # Inputs Recap
    fc, fy, b, h, cov, m_bar, s_bar, man_s = ui.render_design_input(unit_sys)
    
    # Calculate Data
    bar_areas = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
    bar_dias = {k: int(k[2:]) for k in bar_areas}
    stir_areas = {'RB6':0.28, 'RB9':0.64, 'DB10':0.78, 'DB12':1.13}
    stir_dias = {k: (int(k[2:]) if 'DB' in k else int(k[2:])) for k in stir_areas}
    
    span_results = []
    for i in range(n_span):
        x_start, x_end = cum_len[i], cum_len[i+1]
        span_df = df[(df['x'] >= x_start) & (df['x'] <= x_end)]
        m_span = span_df['moment'].abs().max()
        v_span = span_df['shear'].abs().max()
        res = calculate_rc_design(m_span, v_span, fc, fy, b, h, cov, method, unit_sys, bar_areas[m_bar], stir_areas[s_bar], man_s)
        res['id'] = i+1
        res['M'] = m_span
        res['V'] = v_span
        span_results.append(res)

    # 2.1 Longitudinal Profile
    st.subheader("📍 Longitudinal Profile")
    st.plotly_chart(draw_reinforcement_profile(vis_spans, span_results, m_bar, s_bar), use_container_width=True)

    # 2.2 Detailed Calculation Cards
    st.subheader("📝 Calculation Details (Per Span)")
    
    tabs = st.tabs([f"Span {res['id']}" for res in span_results])
    
    for i, tab in enumerate(tabs):
        res = span_results[i]
        with tab:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"#### Cross Section: Span {res['id']}")
                fig_sec = draw_section_beautiful(b, h, cov, res['nb'], bar_dias[m_bar], stir_dias[s_bar], m_bar, s_bar, res['s_value'], "")
                st.plotly_chart(fig_sec, use_container_width=True)
                
                st.info(f"**Result Summary:**\n\n- Main: **{res['nb']}-{m_bar}**\n- Stirrup: **{s_bar} @ {res['s_value']} cm**")
            
            with col2:
                st.markdown("#### 🔢 Calculation Log")
                with st.container(height=500, border=True):
                    # Render List of Logs as nice Markdown
                    for log_item in res['logs']:
                        if log_item.startswith("###"): st.markdown(log_item)
                        elif log_item.startswith(">"): st.info(log_item.replace("> ", ""))
                        elif log_item.startswith("❌"): st.error(log_item)
                        else: st.markdown(log_item)
