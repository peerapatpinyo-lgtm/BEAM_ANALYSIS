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
st.set_page_config(page_title="RC Beam Pro", layout="wide", page_icon="🏗️")
if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    .main-header { background: #2c3e50; color: white; padding: 15px; border-radius: 5px; text-align: center; margin-bottom: 20px; }
    .section-header { border-bottom: 2px solid #3498db; color: #2c3e50; font-size: 1.2rem; font-weight: bold; margin-top: 20px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- DRAWING FUNCTIONS ---

def draw_reinforcement_profile(spans, design_results, m_bar, s_bar):
    """ วาดรูปด้านข้าง แยกเหล็กตาม Span จริง """
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()

    # 1. Concrete Body
    fig.add_shape(type="rect", x0=0, y0=-0.5, x1=total_len, y1=0.5, 
                  line=dict(color="black", width=2), fillcolor="#f0f3f4")

    # 2. Iterate Spans to draw Specific Rebar
    for i in range(len(spans)):
        x0 = cum_len[i]
        x1 = cum_len[i+1]
        res = design_results[i]
        
        # Bottom Bar (Blue) - วาดเฉพาะช่วงนี้
        fig.add_trace(go.Scatter(
            x=[x0 + 0.05, x1 - 0.05], # inset slightly
            y=[-0.35, -0.35],
            mode='lines+text',
            line=dict(color='#004aad', width=4),
            text=[f"{res['nb']}-{m_bar}", ""],
            textposition="bottom center",
            name=f"Span {i+1} Bottom", showlegend=False
        ))

        # Top Bar (Red) - วาดเฉพาะช่วงนี้ (สมมติว่าใส่ยาวตลอดช่วงเพื่อความง่ายในการวาด)
        fig.add_trace(go.Scatter(
            x=[x0 + 0.05, x1 - 0.05],
            y=[0.35, 0.35],
            mode='lines',
            line=dict(color='#c0392b', width=3, dash='solid'),
            name=f"Span {i+1} Top", showlegend=False
        ))
        
        # Stirrup Label (Center of span)
        mid_x = (x0 + x1) / 2
        fig.add_annotation(
            x=mid_x, y=0,
            text=f"Stir: {s_bar}<br>{res['stirrup_text']}",
            showarrow=False,
            font=dict(size=10, color="#d35400"),
            bgcolor="rgba(255,255,255,0.7)"
        )

    # Supports
    for x in cum_len:
         fig.add_trace(go.Scatter(x=[x], y=[-0.6], mode='markers', 
                                  marker=dict(symbol="triangle-up", size=15, color="#2c3e50"), 
                                  showlegend=False, hoverinfo='skip'))

    fig.update_layout(
        title="🏗️ Reinforcement Profile (As Designed)",
        height=250,
        xaxis=dict(title="Distance (m)", range=[-0.2, total_len+0.2], showgrid=True),
        yaxis=dict(visible=False, range=[-1, 1]),
        margin=dict(l=20,r=20,t=40,b=20), plot_bgcolor='white'
    )
    return fig

def draw_section_engineering(b_cm, h_cm, cov_cm, nb, bd_mm, stir_d_mm, main_name, stir_name, s_val_mm, title):
    """ วาด Cross Section แบบวิศวกรรม (วงกลมตามจำนวนจริง) """
    fig = go.Figure()
    
    # Scale variables for plotting (Use cm as base unit)
    bd_cm = bd_mm / 10.0
    sd_cm = stir_d_mm / 10.0
    
    # 1. Concrete Box
    fig.add_shape(type="rect", x0=0, y0=0, x1=b_cm, y1=h_cm, 
                  line=dict(color="black", width=3), fillcolor="#ffffff")
    
    # 2. Stirrup (Red Line) - Offset by cover
    sx0, sy0 = cov_cm, cov_cm
    sx1, sy1 = b_cm - cov_cm, h_cm - cov_cm
    fig.add_shape(type="rect", x0=sx0, y0=sy0, x1=sx1, y1=sy1,
                  line=dict(color="#c0392b", width=3), fillcolor="rgba(0,0,0,0)")
    
    # 3. Main Bars (Blue Circles) - Calculate exact positions
    if nb > 0:
        # Effective width inside stirrups to center of bars
        # Distribute 'nb' bars evenly along the bottom width
        start_x = cov_cm + sd_cm + bd_cm/2
        end_x = b_cm - (cov_cm + sd_cm + bd_cm/2)
        
        if nb == 1:
            xs = [(start_x + end_x)/2]
        else:
            # Create linear space
            xs = np.linspace(start_x, end_x, nb)
            
        y_pos = cov_cm + sd_cm + bd_cm/2
        
        # Plot each bar as a circle
        fig.add_trace(go.Scatter(
            x=xs, y=[y_pos]*nb, 
            mode='markers',
            marker=dict(size=bd_mm*1.8, color='#004aad', line=dict(width=2, color='black')), # Scale size for visibility
            name="Main Bar", showlegend=False
        ))
        
    # 4. Hanger Bars (Top - Dummy 2 bars)
    y_top = h_cm - (cov_cm + sd_cm + bd_cm/2)
    fig.add_trace(go.Scatter(
        x=[start_x, end_x], y=[y_top, y_top], 
        mode='markers',
        marker=dict(size=bd_mm*1.2, color='#95a5a6', line=dict(width=1, color='black')),
        name="Hanger", showlegend=False
    ))

    # Annotations
    fig.add_annotation(x=b_cm/2, y=-h_cm*0.15, text=f"b = {b_cm*10:.0f} mm", showarrow=False)
    fig.add_annotation(x=-b_cm*0.2, y=h_cm/2, text=f"h = {h_cm*10:.0f} mm", textangle=-90, showarrow=False)
    fig.add_annotation(x=b_cm/2, y=h_cm*0.5, text=f"{nb}-{main_name}", font=dict(color="blue", size=14), showarrow=False, yshift=-20)
    fig.add_annotation(x=b_cm/2, y=h_cm*0.5, text=f"{stir_name}@{s_val_mm}mm", font=dict(color="red", size=12), showarrow=False, yshift=20)

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)), width=280, height=320,
        xaxis=dict(visible=False, range=[-b_cm*0.4, b_cm*1.4]), 
        yaxis=dict(visible=False, range=[-h_cm*0.3, h_cm*1.3]),
        margin=dict(l=10,r=10,t=30,b=10), plot_bgcolor='white'
    )
    return fig

# ==========================================
# MAIN APP
# ==========================================
st.markdown('<div class="main-header"><h2>🏗️ RC Beam Pro Ultimate</h2></div>', unsafe_allow_html=True)

design_code, method, fact_dl, fact_ll, unit_sys = ui.render_sidebar()
c_geo, c_load = st.columns([1, 1.5])
with c_geo: n_span, spans, supports = ui.render_geometry_input()
with c_load: loads_input = ui.render_loads_input(n_span, spans, fact_dl, fact_ll, unit_sys)

if st.button("🚀 Calculate", type="primary"):
    solver = BeamFiniteElement(spans, supports, loads_input)
    success, msg = solver.solve()
    if success:
        st.session_state['res_df'] = solver.get_internal_forces()
        st.session_state['vis_data'] = (spans, supports, loads_input)
        st.session_state['analyzed'] = True
    else:
        st.error(f"Error: {msg}")

if st.session_state['analyzed']:
    vis_data = st.session_state['vis_data']
    df = st.session_state['res_df'].copy()
    vis_spans = vis_data[0]
    total_len = sum(vis_spans)
    cum_len = [0] + list(np.cumsum(vis_spans))
    
    # --- 1. LINKED ANALYSIS GRAPHS ---
    st.markdown('<div class="section-header">1️⃣ Analysis Results</div>', unsafe_allow_html=True)
    
    # ใช้ make_subplots จัด Layout 3 แถว (Model, Shear, Moment)
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.15, 0.4, 0.4],
        subplot_titles=("Beam Model", "Shear Force (V)", "Bending Moment (M)")
    )

    # 1.1 Beam Model (Row 1)
    # Draw Beam
    fig.add_shape(type="rect", x0=0, y0=-0.1, x1=total_len, y1=0.1, 
                  fillcolor="lightgray", line=dict(color="black"), row=1, col=1)
    # Draw Supports
    for i, s in enumerate(supports):
        if s != "None":
            sx = cum_len[i]
            fig.add_trace(go.Scatter(x=[sx], y=[-0.12], mode='markers', 
                                     marker=dict(symbol="triangle-up", size=12, color="green"),
                                     hoverinfo='text', text=f"{s} Support"), row=1, col=1)
    # Draw Loads (Simplified Arrow indication)
    for load in loads_input:
        lx_start = cum_len[load['span_idx']]
        lx_end = cum_len[load['span_idx']+1]
        mid = (lx_start+lx_end)/2
        fig.add_annotation(x=mid, y=0.15, text=f"w={load['w']:.1f}", showarrow=True, arrowhead=2, ax=0, ay=-20, row=1, col=1)

    # 1.2 Shear (Row 2)
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', 
                             line=dict(color='#e74c3c', width=2), name="Shear"), row=2, col=1)

    # 1.3 Moment (Row 3)
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', 
                             line=dict(color='#3498db', width=2), name="Moment"), row=3, col=1)

    # Layout Adjustments
    fig.update_layout(height=800, showlegend=False, hovermode="x unified")
    
    # Fix X-axis alignment
    fig.update_xaxes(range=[-0.5, total_len+0.5], showgrid=True, row=1, col=1)
    fig.update_xaxes(range=[-0.5, total_len+0.5], showgrid=True, row=2, col=1)
    fig.update_xaxes(range=[-0.5, total_len+0.5], title_text="Distance (m)", tickmode='linear', dtick=1.0, row=3, col=1)
    
    # Fix Y-axis
    fig.update_yaxes(visible=False, row=1, col=1) # Hide Y for beam model
    
    st.plotly_chart(fig, use_container_width=True)

    # --- 2. DESIGN ---
    st.markdown('<div class="section-header">2️⃣ Structural Design</div>', unsafe_allow_html=True)
    
    fc, fy, b_cm, h_cm, cov_cm, m_bar, s_bar, man_s = ui.render_design_input(unit_sys)
    
    # DB
    bar_areas = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
    bar_dias_mm = {k: int(k[2:]) for k in bar_areas}
    stir_areas = {'RB6':0.28, 'RB9':0.64, 'DB10':0.78, 'DB12':1.13}
    stir_dias_mm = {k: (int(k[2:]) if 'DB' in k else int(k[2:])) for k in stir_areas}
    
    span_results = []
    for i in range(n_span):
        x_start, x_end = cum_len[i], cum_len[i+1]
        span_df = df[(df['x'] >= x_start) & (df['x'] <= x_end)]
        m_span = span_df['moment'].abs().max()
        v_span = span_df['shear'].abs().max()
        
        res = calculate_rc_design(m_span, v_span, fc, fy, b_cm, h_cm, cov_cm, method, unit_sys, bar_areas[m_bar], stir_areas[s_bar], man_s)
        res['id'] = i+1
        span_results.append(res)

    # 2.1 Profile
    st.plotly_chart(draw_reinforcement_profile(vis_spans, span_results, m_bar, s_bar), use_container_width=True)

    # 2.2 Sections
    st.markdown("#### 🔍 Section Detail & Calculation")
    tabs = st.tabs([f"Span {res['id']}" for res in span_results])
    
    for i, tab in enumerate(tabs):
        res = span_results[i]
        with tab:
            c1, c2 = st.columns([1, 2])
            with c1:
                # วาด Section ที่มีวงกลมตามจริง
                fig_sec = draw_section_engineering(
                    b_cm, h_cm, cov_cm, res['nb'], 
                    bar_dias_mm[m_bar], stir_dias_mm[s_bar], 
                    m_bar, s_bar, res['s_value_mm'], # Send mm value
                    f"Section Span {res['id']}"
                )
                st.plotly_chart(fig_sec, use_container_width=True, key=f"sec_{i}")
                
            with c2:
                with st.expander("Show Calculation Log", expanded=True):
                    for line in res['logs']:
                        st.markdown(line)
