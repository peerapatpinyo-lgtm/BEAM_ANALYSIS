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
    st.error("⚠️ Error: Missing required files. Please ensure 'beam_analysis.py', 'rc_design.py', and 'input_handler.py' exist.")
    st.stop()

# ==========================================
# SETUP
# ==========================================
st.set_page_config(page_title="RC Beam Pro Ultimate", layout="wide", page_icon="🏗️")

if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    .header-box { background: #1565C0; color: white; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 20px; }
    .sub-header { border-left: 5px solid #1565C0; padding-left: 10px; font-size: 1.2rem; font-weight: bold; margin-top: 25px; margin-bottom: 10px; background: #E3F2FD; padding: 8px; border-radius: 0 5px 5px 0;}
</style>
""", unsafe_allow_html=True)

# --- PLOTTING FUNCTIONS ---

def draw_reinforcement_profile(spans, design_results, m_bar, s_bar):
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()

    # Concrete Body
    fig.add_shape(type="rect", x0=0, y0=-0.5, x1=total_len, y1=0.5, line=dict(color="black", width=2), fillcolor="#F5F5F5", layer="below")

    # Rebars per span
    for i in range(len(spans)):
        x0, x1 = cum_len[i], cum_len[i+1]
        res = design_results[i]
        
        # Bottom Bar (Blue) - Inset
        fig.add_trace(go.Scatter(
            x=[x0 + 0.1, x1 - 0.1], y=[-0.35, -0.35],
            mode='lines', line=dict(color='#1565C0', width=5),
            name=f"Bott {i+1}", showlegend=False, hoverinfo='text', text=f"Span {i+1}: {res['nb']}-{m_bar}"
        ))
        
        # Top Bar (Red)
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[0.35, 0.35],
            mode='lines', line=dict(color='#D32F2F', width=3),
            name=f"Top {i+1}", showlegend=False, hoverinfo='skip'
        ))

        mid = (x0+x1)/2
        fig.add_annotation(
            x=mid, y=0,
            text=f"<b>SPAN {i+1}</b><br>{res['nb']}-{m_bar}<br>{s_bar} {res['stirrup_text']}",
            showarrow=False, font=dict(size=10), bgcolor="rgba(255,255,255,0.8)"
        )

    # Supports
    for x in cum_len:
         fig.add_trace(go.Scatter(x=[x], y=[-0.6], mode='markers', marker=dict(symbol="triangle-up", size=15, color="#333"), showlegend=False, hoverinfo='skip'))

    fig.update_layout(
        title="🏗️ Reinforcement Profile (Side View)", height=250,
        xaxis=dict(range=[-0.5, total_len+0.5], showgrid=True),
        yaxis=dict(visible=False, range=[-1, 1]),
        margin=dict(l=20,r=20,t=40,b=20), plot_bgcolor='white'
    )
    return fig

def draw_section_real(b_cm, h_cm, cov_cm, nb, bd_mm, stir_d_mm, main_name, stir_name, s_val_mm, title):
    fig = go.Figure()
    bd_cm = bd_mm / 10.0
    sd_cm = stir_d_mm / 10.0
    
    # 1. Concrete
    fig.add_shape(type="rect", x0=0, y0=0, x1=b_cm, y1=h_cm, line=dict(color="black", width=3), fillcolor="#FAFAFA")
    
    # 2. Stirrup
    sx0, sy0 = cov_cm, cov_cm
    sx1, sy1 = b_cm - cov_cm, h_cm - cov_cm
    fig.add_shape(type="rect", x0=sx0, y0=sy0, x1=sx1, y1=sy1, line=dict(color="#C62828", width=3), fillcolor="rgba(0,0,0,0)")
    
    # 3. Main Bars (Circles)
    if nb > 0:
        start_x = cov_cm + sd_cm + bd_cm/2
        end_x = b_cm - (cov_cm + sd_cm + bd_cm/2)
        y_pos = cov_cm + sd_cm + bd_cm/2
        
        if nb == 1: x_pos = [(start_x+end_x)/2]
        else: x_pos = np.linspace(start_x, end_x, nb)
            
        for xp in x_pos:
            fig.add_shape(type="circle",
                x0=xp-bd_cm/2, y0=y_pos-bd_cm/2, x1=xp+bd_cm/2, y1=y_pos+bd_cm/2,
                line_color="black", fillcolor="#1565C0"
            )
            
    # 4. Hangers (Top Dummy)
    y_top = h_cm - (cov_cm + sd_cm + bd_cm/2)
    for xp in [start_x, end_x]:
        fig.add_shape(type="circle",
            x0=xp-bd_cm/2, y0=y_top-bd_cm/2, x1=xp+bd_cm/2, y1=y_top+bd_cm/2,
            line_color="black", fillcolor="#B0BEC5"
        )

    # Labels
    fig.add_annotation(x=b_cm/2, y=-h_cm*0.1, text=f"b={b_cm*10:.0f}mm", showarrow=False)
    fig.add_annotation(x=-b_cm*0.15, y=h_cm/2, text=f"h={h_cm*10:.0f}mm", textangle=-90, showarrow=False)
    fig.add_annotation(x=b_cm/2, y=h_cm/2, text=f"{nb}-{main_name}<br>{stir_name}@{s_val_mm}mm", showarrow=False, font=dict(size=14, color="#1565C0"))

    fig.update_layout(
        title=dict(text=title, x=0.5), width=250, height=300,
        xaxis=dict(visible=False, range=[-b_cm*0.5, b_cm*1.5]), 
        yaxis=dict(visible=False, range=[-h_cm*0.2, h_cm*1.2]),
        margin=dict(l=10,r=10,t=40,b=10), plot_bgcolor='white'
    )
    return fig

# ==========================================
# MAIN APP EXECUTION
# ==========================================
st.markdown('<div class="header-box"><h2>🏗️ RC Beam Pro Ultimate</h2></div>', unsafe_allow_html=True)

# 1. Inputs
design_code, method, fact_dl, fact_ll, unit_sys = ui.render_sidebar()
c_geo, c_load = st.columns([1, 1.5])
with c_geo: n_span, spans, supports = ui.render_geometry_input()
with c_load: loads_input = ui.render_loads_input(n_span, spans, fact_dl, fact_ll, unit_sys)

# 2. Calculation
if st.button("🚀 Calculate Analysis & Design", type="primary"):
    solver = BeamFiniteElement(spans, supports, loads_input)
    success, msg = solver.solve()
    if success:
        st.session_state['res_df'] = solver.get_internal_forces()
        st.session_state['vis_data'] = (spans, supports)
        st.session_state['analyzed'] = True
    else:
        st.error(f"Analysis Failed: {msg}")

# 3. Visualization
if st.session_state['analyzed'] and st.session_state.get('res_df') is not None:
    df = st.session_state['res_df']
    vis_spans, vis_supports = st.session_state['vis_data']
    total_len = sum(vis_spans)
    cum_len = [0] + list(np.cumsum(vis_spans))

    st.markdown('<div class="sub-header">1️⃣ Analysis Diagrams</div>', unsafe_allow_html=True)
    
    # --- PLOT ANALYSIS ---
    try:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                            subplot_titles=("Beam Model", "Shear Force", "Bending Moment"),
                            row_heights=[0.2, 0.4, 0.4])

        # Row 1: Beam
        fig.add_shape(type="rect", x0=0, y0=-0.1, x1=total_len, y1=0.1, fillcolor="#E0E0E0", line_color="black", row=1, col=1)
        for i, s in enumerate(vis_supports):
            if s != "None":
                fig.add_trace(go.Scatter(x=[cum_len[i]], y=[-0.15], mode='markers', marker=dict(symbol="triangle-up", size=12, color="green"), name=s), row=1, col=1)

        # Row 2: Shear
        fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#E53935'), name="Shear"), row=2, col=1)

        # Row 3: Moment
        fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1E88E5'), name="Moment"), row=3, col=1)

        fig.update_layout(height=700, showlegend=False, hovermode="x unified")
        fig.update_xaxes(range=[-0.5, total_len+0.5]) 
        fig.update_yaxes(visible=False, row=1, col=1)
        
        st.plotly_chart(fig, use_container_width=True, key="analysis_plot")
        
    except Exception as e:
        st.error(f"Error plotting graph: {e}")

    # --- DESIGN SECTION ---
    st.markdown('<div class="sub-header">2️⃣ Structural Design Results</div>', unsafe_allow_html=True)
    
    fc, fy, b_cm, h_cm, cov_cm, m_bar, s_bar, man_s_cm = ui.render_design_input(unit_sys)
    
    bar_areas = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
    stir_areas = {'RB6':0.28, 'RB9':0.64, 'DB10':0.78, 'DB12':1.13}
    bd_map = {k: int(k[2:]) for k in bar_areas}
    sd_map = {k: (int(k[2:]) if 'DB' in k else int(k[2:])) for k in stir_areas}

    span_results = []
    for i in range(n_span):
        x0, x1 = cum_len[i], cum_len[i+1]
        sub_df = df[(df['x'] >= x0) & (df['x'] <= x1)]
        
        res = calculate_rc_design(
            sub_df['moment'].abs().max(), sub_df['shear'].abs().max(),
            fc, fy, b_cm, h_cm, cov_cm, method, unit_sys, 
            bar_areas[m_bar], stir_areas[s_bar], man_s_cm
        )
        res['id'] = i+1
        span_results.append(res)

    # 1. Profile View
    st.plotly_chart(draw_reinforcement_profile(vis_spans, span_results, m_bar, s_bar), use_container_width=True, key="profile_plot")

    # 2. Section Details
    st.markdown("#### 🔍 Section Details")
    tabs = st.tabs([f"Span {r['id']}" for r in span_results])
    for i, tab in enumerate(tabs):
        r = span_results[i]
        with tab:
            c1, c2 = st.columns([1, 2])
            with c1:
                fig_sec = draw_section_real(
                    b_cm, h_cm, cov_cm, r['nb'], 
                    bd_map[m_bar], sd_map[s_bar], m_bar, s_bar, r['s_value_mm'], 
                    f"Section Span {r['id']}"
                )
                st.plotly_chart(fig_sec, use_container_width=True, key=f"sec_{i}")
            with c2:
                for line in r['logs']: st.markdown(line)
