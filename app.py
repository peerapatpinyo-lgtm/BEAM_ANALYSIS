import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- IMPORT MODULES ---
try:
    from beam_analysis import run_beam_analysis # ใช้ function wrapper ใหม่
    from rc_design import calculate_rc_design
    import input_handler as ui
except ImportError:
    st.error("⚠️ Error: Missing required files. Please ensure 'beam_analysis.py', 'rc_design.py', and 'input_handler.py' exist.")
    st.stop()

# ==========================================
# SETUP & STYLES
# ==========================================
st.set_page_config(page_title="RC Beam Pro Ultimate", layout="wide", page_icon="🏗️")

if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False
if 'loads_input' not in st.session_state: st.session_state['loads_input'] = []

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    .header-box { background: linear-gradient(90deg, #1565C0 0%, #0D47A1 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .sub-header { border-left: 5px solid #1565C0; padding-left: 15px; font-size: 1.25rem; font-weight: 600; margin-top: 30px; margin-bottom: 15px; background: #E3F2FD; padding: 10px; border-radius: 0 8px 8px 0; color: #0D47A1; }
    .metric-card { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 15px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# ADVANCED PLOTTING FUNCTIONS
# ==========================================

def create_engineering_plots(df, vis_spans, vis_supports, loads, unit_sys):
    """
    Creates professional structural engineering diagrams.
    1. Load Diagram (Visual representation of inputs)
    2. Shear Force Diagram (with local max/min per span)
    3. Bending Moment Diagram (with local max/min per span)
    """
    cum_len = [0] + list(np.cumsum(vis_spans))
    total_len = cum_len[-1]
    
    # Define Units
    force_unit = "kg" if unit_sys == "Metric (kg/cm)" else "kN"
    dist_unit = "m"
    moment_unit = "kg-m" if unit_sys == "Metric (kg/cm)" else "kN-m"
    
    # Create Subplots
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=("<b>Load Diagram</b>", "<b>Shear Force Diagram (SFD)</b>", "<b>Bending Moment Diagram (BMD)</b>"),
        row_heights=[0.25, 0.375, 0.375]
    )

    # --- 1. LOAD DIAGRAM (Row 1) ---
    # Draw Beam
    fig.add_shape(type="rect", x0=0, y0=-0.05, x1=total_len, y1=0.05, fillcolor="#CFD8DC", line_color="black", row=1, col=1)
    
    # Draw Supports
    for i, x in enumerate(cum_len):
        sup_type = vis_supports.iloc[i]['type'] if i < len(vis_supports) else "None" # Fix index logic if needed
        # Check actual supports logic from input. Assuming standard nodes here.
        # For simplicity, using the span endpoints
        if i < len(vis_supports): # Defensive
            stype = vis_supports.iloc[i]['type'] if 'type' in vis_supports.columns else "Pin"
            if stype != "None":
                symbol = "triangle-up" if stype == "Pin" else "circle"
                fig.add_trace(go.Scatter(x=[x], y=[-0.05], mode='markers', marker=dict(symbol=symbol, size=15, color="black"), hoverinfo='skip', showlegend=False), row=1, col=1)

    # Draw Loads
    max_load_h = 0
    for load in loads:
        span_idx = load['span_idx']
        w = load.get('w', 0)
        x_start = cum_len[span_idx]
        x_end = cum_len[span_idx+1]
        
        if w != 0: # Distributed Load
            # Draw a shaded block
            fig.add_trace(go.Scatter(
                x=[x_start, x_end, x_end, x_start], 
                y=[0, 0, w, w], 
                fill='toself', fillcolor='rgba(255, 179, 0, 0.3)', 
                line=dict(color='orange', width=0), 
                showlegend=False, hoverinfo='text', text=f"w = {w} {force_unit}/{dist_unit}"
            ), row=1, col=1)
            
            # Draw Arrows (Visual representation)
            steps = np.linspace(x_start, x_end, 5) # 5 arrows per span
            for sx in steps:
                fig.add_annotation(
                    x=sx, y=w/2, ax=0, ay=-20,
                    xref="x1", yref="y1",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="orange"
                )
            # Label
            fig.add_annotation(x=(x_start+x_end)/2, y=w, text=f"w={w:.0f}", showarrow=False, yshift=10, row=1, col=1)

    # --- 2. SHEAR FORCE (Row 2) ---
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'], 
        mode='lines', line=dict(color='#D32F2F', width=2), 
        fill='tozeroy', fillcolor='rgba(211, 47, 47, 0.1)',
        name="Shear"
    ), row=2, col=1)

    # Annotate Max/Min Shear per Span
    for i in range(len(vis_spans)):
        # Filter data for this span
        span_data = df[df['span_id'] == i]
        if not span_data.empty:
            max_v = span_data['shear'].max()
            min_v = span_data['shear'].min()
            
            # Label Positive Peak
            idxmax = span_data['shear'].idxmax()
            fig.add_annotation(
                x=span_data.loc[idxmax, 'x'], y=max_v,
                text=f"<b>{max_v:.2f}</b>", showarrow=False, yshift=10,
                font=dict(color="#D32F2F", size=10), bgcolor="rgba(255,255,255,0.7)", row=2, col=1
            )
            # Label Negative Peak
            idxmin = span_data['shear'].idxmin()
            fig.add_annotation(
                x=span_data.loc[idxmin, 'x'], y=min_v,
                text=f"<b>{min_v:.2f}</b>", showarrow=False, yshift=-10,
                font=dict(color="#D32F2F", size=10), bgcolor="rgba(255,255,255,0.7)", row=2, col=1
            )

    # --- 3. BENDING MOMENT (Row 3) ---
    # Note: Engineering convention usually flips moment, but for charts standard is fine, just label well.
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['moment'], 
        mode='lines', line=dict(color='#1976D2', width=2), 
        fill='tozeroy', fillcolor='rgba(25, 118, 210, 0.1)',
        name="Moment"
    ), row=3, col=1)

    # Annotate Max/Min Moment per Span
    for i in range(len(vis_spans)):
        span_data = df[df['span_id'] == i]
        if not span_data.empty:
            # Positive Moment (Sagging)
            max_m = span_data['moment'].max()
            if max_m > 0.1: # Threshold to avoid labeling zero noise
                idxmax = span_data['moment'].idxmax()
                fig.add_annotation(
                    x=span_data.loc[idxmax, 'x'], y=max_m,
                    text=f"<b>{max_m:.2f}</b>", showarrow=False, yshift=10,
                    font=dict(color="#1976D2", size=10), bgcolor="rgba(255,255,255,0.7)", row=3, col=1
                )
            
            # Negative Moment (Hogging - usually at supports)
            min_m = span_data['moment'].min()
            if min_m < -0.1:
                idxmin = span_data['moment'].idxmin()
                fig.add_annotation(
                    x=span_data.loc[idxmin, 'x'], y=min_m,
                    text=f"<b>{min_m:.2f}</b>", showarrow=False, yshift=-10,
                    font=dict(color="#1976D2", size=10), bgcolor="rgba(255,255,255,0.7)", row=3, col=1
                )

    # --- COMMON LAYOUT SETTINGS ---
    # Add vertical lines for supports across all subplots
    for x in cum_len:
        fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

    # Zero lines
    fig.add_hline(y=0, line_width=1.5, line_color="black", opacity=1, row=2, col=1)
    fig.add_hline(y=0, line_width=1.5, line_color="black", opacity=1, row=3, col=1)

    fig.update_layout(
        height=800, 
        showlegend=False, 
        plot_bgcolor="white",
        hovermode="x unified",
        margin=dict(t=40, b=40, l=60, r=40)
    )
    
    # Update Axes Labels
    fig.update_yaxes(title_text=f"Load", row=1, col=1, showticklabels=False)
    fig.update_yaxes(title_text=f"Shear ({force_unit})", row=2, col=1)
    fig.update_yaxes(title_text=f"Moment ({moment_unit})", row=3, col=1)
    fig.update_xaxes(title_text=f"Length ({dist_unit})", row=3, col=1)

    return fig

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
    try:
        # Save inputs for visualization later
        st.session_state['loads_input'] = loads_input
        
        # New Wrapper Function from beam_analysis.py
        vis_spans_df, vis_supports_df = run_beam_analysis(spans, supports, loads_input)
        
        # Store in session state
        st.session_state['res_df'] = vis_spans_df
        st.session_state['vis_data'] = (spans, vis_supports_df) # Store input spans and Result reactions
        st.session_state['analyzed'] = True
        
    except ValueError as ve:
        st.error(str(ve))
    except Exception as e:
        st.error(f"System Error: {e}")

# 3. Visualization
if st.session_state['analyzed'] and st.session_state.get('res_df') is not None:
    df = st.session_state['res_df']
    vis_spans, vis_supports_df = st.session_state['vis_data']
    loads = st.session_state['loads_input']
    
    st.markdown('<div class="sub-header">1️⃣ Analysis Results</div>', unsafe_allow_html=True)
    
    # --- PLOT ANALYSIS (ENGINEERING GRADE) ---
    try:
        fig_eng = create_engineering_plots(df, vis_spans, vis_supports_df, loads, unit_sys)
        st.plotly_chart(fig_eng, use_container_width=True, key="eng_plot")
    except Exception as e:
        st.error(f"Error plotting diagrams: {e}")

    # --- DESIGN SECTION ---
    st.markdown('<div class="sub-header">2️⃣ Structural Design Results</div>', unsafe_allow_html=True)
    
    fc, fy, b_cm, h_cm, cov_cm, m_bar, s_bar, man_s_cm = ui.render_design_input(unit_sys)
    
    bar_areas = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
    stir_areas = {'RB6':0.28, 'RB9':0.64, 'DB10':0.78, 'DB12':1.13}
    bd_map = {k: int(k[2:]) for k in bar_areas}
    sd_map = {k: (int(k[2:]) if 'DB' in k else int(k[2:])) for k in stir_areas}

    span_results = []
    cum_len = [0] + list(np.cumsum(vis_spans))
    
    for i in range(n_span):
        x0, x1 = cum_len[i], cum_len[i+1]
        # Filter Data for this Span
        sub_df = df[(df['x'] >= x0) & (df['x'] <= x1)]
        
        # Get Max Forces
        max_m = sub_df['moment'].abs().max()
        max_v = sub_df['shear'].abs().max()
        
        res = calculate_rc_design(
            max_m, max_v,
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
                st.info(f"**Design Forces Span {i+1}:**\n\n- M_u = {r.get('Mu', 0):.2f}\n- V_u = {r.get('Vu', 0):.2f}")
                for line in r['logs']: st.markdown(line)
