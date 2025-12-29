
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- IMPORT MODULES ---
try:
    from beam_analysis import run_beam_analysis 
    from rc_design import calculate_rc_design
    import input_handler as ui
except ImportError as e:
    st.error(f"⚠️ Error: Missing required files. {e}")
    st.stop()

# ==========================================
# SETUP & STYLES
# ==========================================
st.set_page_config(page_title="RC Beam Pro Ultimate", layout="wide", page_icon="🏗️")

if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    .header-box { background: linear-gradient(90deg, #1565C0 0%, #0D47A1 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .sub-header { border-left: 5px solid #1565C0; padding-left: 15px; font-size: 1.25rem; font-weight: 600; margin-top: 30px; margin-bottom: 15px; background: #E3F2FD; padding: 10px; border-radius: 0 8px 8px 0; color: #0D47A1; }
    .stNumberInput input { font-weight: bold; color: #1565C0; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CUSTOM LOAD INPUT
# ==========================================
def render_custom_load_input(n_span, spans, unit_sys):
    st.markdown("### 3️⃣ Applied Loads")
    force_unit = "kg" if "Metric" in unit_sys else "kN"
    dist_unit = "m"
    loads = []
    
    tabs = st.tabs([f"Span {i+1}" for i in range(n_span)])
    
    for i, tab in enumerate(tabs):
        with tab:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Uniform Load (w)**")
                w_dl = st.number_input(f"Dead Load w ({force_unit}/{dist_unit})", value=0.0, key=f"w_dl_{i}")
                w_ll = st.number_input(f"Live Load w ({force_unit}/{dist_unit})", value=0.0, key=f"w_ll_{i}")
                if w_dl + w_ll != 0:
                    loads.append({'span_idx': i, 'type': 'U', 'w_dl': w_dl, 'w_ll': w_ll, 'w': w_dl + w_ll})
            
            with c2:
                st.markdown(f"**Point Loads (P)**")
                num_p = st.number_input(f"Qty Point Load", min_value=0, max_value=5, value=0, key=f"num_p_{i}")
                for j in range(num_p):
                    cc1, cc2, cc3 = st.columns([1, 1, 1])
                    p_dl = cc1.number_input(f"P{j+1} DL", value=0.0, key=f"p_dl_{i}_{j}")
                    p_ll = cc2.number_input(f"P{j+1} LL", value=0.0, key=f"p_ll_{i}_{j}")
                    x_loc = cc3.number_input(f"x (m)", value=spans[i]/2, min_value=0.0, max_value=float(spans[i]), key=f"px_{i}_{j}")
                    
                    if p_dl + p_ll != 0:
                        loads.append({'span_idx': i, 'type': 'P', 'P_dl': p_dl, 'P_ll': p_ll, 'P': p_dl + p_ll, 'x': x_loc})
    return loads

# ==========================================
# ENGINEERING PLOTTING (Textbook Convention Fix)
# ==========================================

def draw_support_shape(fig, x, y, sup_type, size=1.0):
    """Draws STANDARD Engineering Supports"""
    s = size * 0.8
    line_col, fill_col = "#37474F", "#CFD8DC"
    
    if sup_type == "Pin":
        fig.add_shape(type="path", path=f"M {x},{y} L {x-s/2},{y-s} L {x+s/2},{y-s} Z", fillcolor=fill_col, line_color=line_col, line_width=2, row=1, col=1)
        fig.add_shape(type="line", x0=x-s, y0=y-s, x1=x+s, y1=y-s, line=dict(color=line_col, width=3), row=1, col=1)
        for hx in np.linspace(x-s, x+s, 5): 
            fig.add_shape(type="line", x0=hx, y0=y-s, x1=hx-s/4, y1=y-s*1.3, line=dict(color=line_col, width=1), row=1, col=1)

    elif sup_type == "Roller":
        fig.add_shape(type="path", path=f"M {x},{y} L {x-s/2},{y-s*0.8} L {x+s/2},{y-s*0.8} Z", fillcolor=fill_col, line_color=line_col, line_width=2, row=1, col=1)
        r_wheel = s * 0.15
        fig.add_shape(type="circle", x0=x-s/3, y0=y-s*0.8-2*r_wheel, x1=x-s/3+2*r_wheel, y1=y-s*0.8, line_color=line_col, row=1, col=1)
        fig.add_shape(type="circle", x0=x+s/3-2*r_wheel, y0=y-s*0.8-2*r_wheel, x1=x+s/3, y1=y-s*0.8, line_color=line_col, row=1, col=1)
        base_y = y - s*0.8 - 2*r_wheel
        fig.add_shape(type="line", x0=x-s, y0=base_y, x1=x+s, y1=base_y, line=dict(color=line_col, width=2), row=1, col=1)

    elif sup_type == "Fixed":
        h_wall = s * 1.5
        fig.add_shape(type="line", x0=x, y0=y-h_wall/2, x1=x, y1=y+h_wall/2, line=dict(color=line_col, width=4), row=1, col=1)
        direction = -1 if x == 0 else 1
        for hy in np.linspace(y-h_wall/2, y+h_wall/2, 6):
            fig.add_shape(type="line", x0=x, y0=hy, x1=x + (direction * s*0.4), y1=hy - s*0.4, line=dict(color=line_col, width=1), row=1, col=1)

def create_engineering_plots(df, vis_spans, vis_supports, loads, unit_sys):
    if not vis_spans or df.empty: return go.Figure()

    cum_len = [0] + list(np.cumsum(vis_spans))
    total_len = cum_len[-1]
    
    force_unit = "kg" if "Metric" in unit_sys else "kN"
    moment_unit = "kg-m" if "Metric" in unit_sys else "kN-m"
    
    # --- INTELLIGENT SCALING ---
    w_vals = [abs(l['w']) for l in loads if l.get('type') == 'U' and l['w'] != 0]
    p_vals = [abs(l['P']) for l in loads if l.get('type') == 'P' and l['P'] != 0]
    max_w = max(w_vals) if w_vals else 1.0
    max_p = max(p_vals) if p_vals else 1.0
    target_h = 1.0 
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("<b>Loading Diagram</b>", "<b>Shear Force Diagram (SFD)</b>", "<b>Bending Moment Diagram (BMD) - Tension Side</b>"),
        row_heights=[0.25, 0.375, 0.375]
    )

    # --- ROW 1: LOADING DIAGRAM ---
    fig.add_shape(type="line", x0=0, y0=0, x1=total_len, y1=0, line=dict(color="black", width=4), row=1, col=1)
    
    # Supports
    sup_size = target_h * 0.25
    for i, x in enumerate(cum_len):
        if i < len(vis_supports):
            stype = vis_supports.iloc[i]['type']
            if stype != "None":
                draw_support_shape(fig, x, 0, stype, size=sup_size)

    # Loads
    for load in loads:
        span_idx = load.get('span_idx', 0)
        if span_idx >= len(vis_spans): continue 
        x_start = cum_len[span_idx]
        x_end = cum_len[span_idx+1]
        
        if load.get('type') == 'U' and load['w'] != 0:
            w = load['w']
            ratio = abs(w) / max_w
            h = (0.3 + 0.7 * ratio) * target_h 
            fig.add_trace(go.Scatter(x=[x_start, x_end, x_end, x_start], y=[0, 0, h, h], fill='toself', fillcolor='rgba(255, 152, 0, 0.2)', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=[x_start, x_end], y=[h, h], mode='lines', line=dict(color='#EF6C00', width=2), showlegend=False, hoverinfo='text', text=f"UDL: {w}"), row=1, col=1)
            n_arrows = max(5, int((x_end - x_start) * 4)) 
            for ax in np.linspace(x_start, x_end, n_arrows):
                fig.add_annotation(x=ax, y=0, ax=ax, ay=h, xref="x1", yref="y1", axref="x1", ayref="y1", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#EF6C00", row=1, col=1)
            fig.add_annotation(x=(x_start+x_end)/2, y=h, text=f"<b>w={w:.0f}</b>", showarrow=False, yshift=15, font=dict(color="#EF6C00", size=11), row=1, col=1)

        elif load.get('type') == 'P' and load['P'] != 0:
            P = load['P']
            load_x = load['x'] + x_start
            ratio = abs(P) / max_p
            h = (0.3 + 0.7 * ratio) * target_h
            fig.add_annotation(x=load_x, y=0, ax=load_x, ay=h, xref="x1", yref="y1", axref="x1", ayref="y1", showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor="#D32F2F", row=1, col=1)
            fig.add_annotation(x=load_x, y=h, text=f"<b>P={P:.0f}</b>", showarrow=False, yshift=15, font=dict(color="#D32F2F", size=12, weight="bold"), row=1, col=1)

    fig.update_yaxes(range=[-target_h*0.4, target_h*1.5], visible=False, row=1, col=1)

    # --- ROW 2: SFD (Standard: Up is positive) ---
    plot_shear = df['shear']
    fig.add_trace(go.Scatter(x=df['x'], y=plot_shear, mode='lines', line=dict(color='#D32F2F', width=2), fill='tozeroy', fillcolor='rgba(211, 47, 47, 0.1)', name="Shear", hovertemplate="%{y:.2f}"), row=2, col=1)

    # --- ROW 3: BMD (Textbook: Tension side / Sagging down) ---
    # *** FIX: INVERT MOMENT FOR PLOTTING ***
    plot_moment = -df['moment'] 
    fig.add_trace(go.Scatter(x=df['x'], y=plot_moment, mode='lines', line=dict(color='#1976D2', width=2), fill='tozeroy', fillcolor='rgba(25, 118, 210, 0.1)', name="Moment", hovertemplate="%{y:.2f} (Tension Side)"), row=3, col=1)
    
    # --- ANNOTATIONS (Peaks based on plotted data) ---
    for i in range(len(vis_spans)):
        # Filter data for this span
        span_indices = df['span_id'] == i
        span_x = df.loc[span_indices, 'x']
        span_v_plot = plot_shear[span_indices]
        span_m_plot = plot_moment[span_indices]

        if span_x.empty: continue

        # SFD Peaks
        for val in [span_v_plot.max(), span_v_plot.min()]:
            if abs(val) > 0.01:
                # Find x corresponding to this peak value in the plotted data
                idx = span_v_plot[span_v_plot == val].index[0]
                vx = df.loc[idx, 'x']
                fig.add_annotation(x=vx, y=val, text=f"{val:.2f}", showarrow=False, yshift=15 if val>0 else -15, font=dict(color="red", size=10), row=2, col=1)

        # BMD Peaks (Annotate the inverted values)
        for val in [span_m_plot.max(), span_m_plot.min()]:
            if abs(val) > 0.01:
                # Find x corresponding to this peak value in the plotted data
                idx = span_m_plot[span_m_plot == val].index[0]
                mx = df.loc[idx, 'x']
                # The value to show is the plotted value (which is already inverted)
                fig.add_annotation(x=mx, y=val, text=f"{val:.2f}", showarrow=False, yshift=15 if val>0 else -15, font=dict(color="blue", size=10), row=3, col=1)

    # Layout
    for r in [2, 3]:
        fig.add_hline(y=0, line_width=1, line_color="black", row=r, col=1)
        for x in cum_len:
            fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="gray", opacity=0.5, row=r, col=1)

    fig.update_layout(height=900, showlegend=False, plot_bgcolor="white", hovermode="x unified", margin=dict(t=40, b=40, l=60, r=40))
    fig.update_yaxes(title_text=f"V ({force_unit})", row=2, col=1)
    fig.update_yaxes(title_text=f"M ({moment_unit})<br>(Tension Side)", row=3, col=1) # Clarify axis label
    fig.update_xaxes(title_text=f"Position (m)", row=3, col=1)

    return fig

def draw_reinforcement_profile(spans, design_results, m_bar, s_bar):
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=-0.5, x1=total_len, y1=0.5, line=dict(color="black", width=2), fillcolor="#F5F5F5", layer="below")

    for i in range(len(spans)):
        x0, x1 = cum_len[i], cum_len[i+1]
        res = design_results[i]
        fig.add_trace(go.Scatter(x=[x0 + 0.1, x1 - 0.1], y=[-0.35, -0.35], mode='lines', line=dict(color='#1565C0', width=5), name=f"Bott {i+1}", showlegend=False, hoverinfo='text', text=f"Span {i+1}: {res['nb']}-{m_bar}"))
        fig.add_trace(go.Scatter(x=[x0, x1], y=[0.35, 0.35], mode='lines', line=dict(color='#D32F2F', width=3), name=f"Top {i+1}", showlegend=False, hoverinfo='skip'))
        mid = (x0+x1)/2
        fig.add_annotation(x=mid, y=0, text=f"<b>SPAN {i+1}</b><br>{res['nb']}-{m_bar}<br>{s_bar} {res['stirrup_text']}", showarrow=False, font=dict(size=10), bgcolor="rgba(255,255,255,0.8)")

    for x in cum_len:
         fig.add_trace(go.Scatter(x=[x], y=[-0.6], mode='markers', marker=dict(symbol="triangle-up", size=15, color="#333"), showlegend=False, hoverinfo='skip'))

    fig.update_layout(title="🏗️ Reinforcement Profile (Side View)", height=250, xaxis=dict(range=[-0.5, total_len+0.5], showgrid=True), yaxis=dict(visible=False, range=[-1, 1]), margin=dict(l=20,r=20,t=40,b=20), plot_bgcolor='white')
    return fig

def draw_section_real(b_cm, h_cm, cov_cm, nb, bd_mm, stir_d_mm, main_name, stir_name, s_val_mm, title):
    fig = go.Figure()
    bd_cm, sd_cm = bd_mm/10.0, stir_d_mm/10.0
    fig.add_shape(type="rect", x0=0, y0=0, x1=b_cm, y1=h_cm, line=dict(color="black", width=3), fillcolor="#FAFAFA")
    sx0, sy0, sx1, sy1 = cov_cm, cov_cm, b_cm - cov_cm, h_cm - cov_cm
    fig.add_shape(type="rect", x0=sx0, y0=sy0, x1=sx1, y1=sy1, line=dict(color="#C62828", width=3), fillcolor="rgba(0,0,0,0)")
    
    if nb > 0:
        start_x, end_x = cov_cm + sd_cm + bd_cm/2, b_cm - (cov_cm + sd_cm + bd_cm/2)
        y_pos = cov_cm + sd_cm + bd_cm/2
        x_pos = [start_x] if nb==1 else np.linspace(start_x, end_x, nb)
        for xp in x_pos: fig.add_shape(type="circle", x0=xp-bd_cm/2, y0=y_pos-bd_cm/2, x1=xp+bd_cm/2, y1=y_pos+bd_cm/2, line_color="black", fillcolor="#1565C0")
            
    y_top = h_cm - (cov_cm + sd_cm + bd_cm/2)
    for xp in [start_x, end_x]: fig.add_shape(type="circle", x0=xp-bd_cm/2, y0=y_top-bd_cm/2, x1=xp+bd_cm/2, y1=y_top+bd_cm/2, line_color="black", fillcolor="#B0BEC5")

    fig.add_annotation(x=b_cm/2, y=h_cm/2, text=f"{nb}-{main_name}<br>{stir_name}@{s_val_mm}mm", showarrow=False, font=dict(size=14, color="#1565C0"))
    fig.update_layout(title=dict(text=title, x=0.5), width=250, height=300, xaxis=dict(visible=False, range=[-b_cm*0.5, b_cm*1.5]), yaxis=dict(visible=False, range=[-h_cm*0.2, h_cm*1.2]), margin=dict(l=10,r=10,t=40,b=10), plot_bgcolor='white')
    return fig

# ==========================================
# MAIN APP EXECUTION
# ==========================================
st.markdown('<div class="header-box"><h2>🏗️ RC Beam Pro Ultimate</h2></div>', unsafe_allow_html=True)

# 1. Inputs
design_code, method, fact_dl, fact_ll, unit_sys = ui.render_sidebar()
c_geo, c_load = st.columns([1, 1.5])
with c_geo: n_span, spans, supports = ui.render_geometry_input()
with c_load: loads_input = render_custom_load_input(n_span, spans, unit_sys)

# 2. Calculation
if st.button("🚀 Calculate Analysis & Design", type="primary"):
    try:
        st.session_state['loads_input'] = loads_input
        # Clean inputs before sending to backend (Redundant safety)
        clean_loads = [l for l in loads_input if isinstance(l, dict)]
        vis_spans_df, vis_supports_df = run_beam_analysis(spans, supports, clean_loads)
        st.session_state['res_df'] = vis_spans_df
        st.session_state['vis_data'] = (spans, vis_supports_df) 
        st.session_state['analyzed'] = True
    except Exception as e:
        st.error(f"System Error: {e}")

# 3. Visualization
if st.session_state['analyzed'] and st.session_state.get('res_df') is not None:
    df = st.session_state['res_df']
    vis_spans, vis_supports_df = st.session_state['vis_data']
    loads = st.session_state['loads_input']
    
    st.markdown('<div class="sub-header">1️⃣ Analysis Results</div>', unsafe_allow_html=True)
    st.plotly_chart(create_engineering_plots(df, vis_spans, vis_supports_df, loads, unit_sys), use_container_width=True, key="eng_plot")

    st.markdown('<div class="sub-header">2️⃣ Structural Design Results</div>', unsafe_allow_html=True)
    fc, fy, b_cm, h_cm, cov_cm, m_bar, s_bar, man_s_cm = ui.render_design_input(unit_sys)
    
    bar_areas = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
    stir_areas = {'RB6':0.28, 'RB9':0.64, 'DB10':0.78, 'DB12':1.13}
    
    span_results = []
    cum_len = [0] + list(np.cumsum(vis_spans))
    
    for i in range(n_span):
        x0, x1 = cum_len[i], cum_len[i+1]
        sub_df = df[(df['x'] >= x0) & (df['x'] <= x1)]
        # Use abs() for design forces, as RC design uses magnitude
        res = calculate_rc_design(sub_df['moment'].abs().max(), sub_df['shear'].abs().max(), fc, fy, b_cm, h_cm, cov_cm, method, unit_sys, bar_areas[m_bar], stir_areas[s_bar], man_s_cm)
        res['id'] = i+1
        span_results.append(res)

    st.plotly_chart(draw_reinforcement_profile(vis_spans, span_results, m_bar, s_bar), use_container_width=True, key="profile_plot")

    st.markdown("#### 🔍 Section Details")
    tabs = st.tabs([f"Span {r['id']}" for r in span_results])
    for i, tab in enumerate(tabs):
        r = span_results[i]
        with tab:
            c1, c2 = st.columns([1, 2])
            with c1: st.plotly_chart(draw_section_real(b_cm, h_cm, cov_cm, r['nb'], int(m_bar[2:]), (int(s_bar[2:]) if 'DB' in s_bar else int(s_bar[2:])), m_bar, s_bar, r['s_value_mm'], f"Section {r['id']}"), use_container_width=True, key=f"sec_{i}")
            with c2: 
                st.info(f"**Design Forces Span {i+1}:**\n\n- M_u = {r['Mu']:.2f}\n- V_u = {r['Vu']:.2f}")
                for l in r['logs']: st.markdown(l)
