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
def render_custom_load_input(n_span, spans, unit_sys, f_dl, f_ll):
    st.markdown("### 3️⃣ Applied Loads (Service Loads)")
    st.caption(f"Note: Factors DL={f_dl:.1f}, LL={f_ll:.1f} will be applied for analysis.")
    force_unit = "kg" if "Metric" in unit_sys else "kN"
    dist_unit = "m"
    loads = []
    
    tabs = st.tabs([f"Span {i+1}" for i in range(n_span)])
    
    for i, tab in enumerate(tabs):
        with tab:
            c1, c2 = st.columns(2)
            # --- Uniform Load ---
            with c1:
                st.markdown(f"**Uniform Load (w)**")
                w_dl = st.number_input(f"Dead Load w ({force_unit}/{dist_unit})", value=0.0, key=f"w_dl_{i}")
                w_ll = st.number_input(f"Live Load w ({force_unit}/{dist_unit})", value=0.0, key=f"w_ll_{i}")
                
                w_u = (w_dl * f_dl) + (w_ll * f_ll)
                if w_u != 0:
                    loads.append({'span_idx': i, 'type': 'U', 'w_dl': w_dl, 'w_ll': w_ll, 'w': w_u, 'val': w_u})
            
            # --- Point Load ---
            with c2:
                st.markdown(f"**Point Loads (P)**")
                num_p = st.number_input(f"Qty Point Load", min_value=0, max_value=5, value=0, key=f"num_p_{i}")
                for j in range(num_p):
                    cc1, cc2, cc3 = st.columns([1, 1, 1])
                    p_dl = cc1.number_input(f"P{j+1} DL", value=0.0, key=f"p_dl_{i}_{j}")
                    p_ll = cc2.number_input(f"P{j+1} LL", value=0.0, key=f"p_ll_{i}_{j}")
                    x_loc = cc3.number_input(f"x (m)", value=spans[i]/2, min_value=0.0, max_value=float(spans[i]), key=f"px_{i}_{j}")
                    
                    p_u = (p_dl * f_dl) + (p_ll * f_ll)
                    if p_u != 0:
                        loads.append({'span_idx': i, 'type': 'P', 'P_dl': p_dl, 'P_ll': p_ll, 'P': p_u, 'p': p_u, 'val': p_u, 'x': x_loc, 'location': x_loc})
    return loads

# ==========================================
# VISUALIZATION FUNCTIONS (ENHANCED)
# ==========================================

def draw_support_shape(fig, x, y, sup_type, size=1.0):
    """Draws ENHANCED Engineering Supports"""
    s = size * 0.9 # Slightly larger
    # Sharper colors for engineering look
    line_col, fill_col = "#263238", "#B0BEC5" 
    
    if sup_type == "Pin":
        # Triangle
        fig.add_shape(type="path", path=f"M {x},{y} L {x-s/2},{y-s} L {x+s/2},{y-s} Z", fillcolor=fill_col, line_color=line_col, line_width=2, row=1, col=1)
        # Base line with hatch
        fig.add_shape(type="line", x0=x-s*0.8, y0=y-s, x1=x+s*0.8, y1=y-s, line=dict(color=line_col, width=3), row=1, col=1)
        for hx in np.linspace(x-s*0.8, x+s*0.8, 6): 
            fig.add_shape(type="line", x0=hx, y0=y-s, x1=hx-s/3, y1=y-s*1.4, line=dict(color=line_col, width=1), row=1, col=1)

    elif sup_type == "Roller":
        # Triangle
        fig.add_shape(type="path", path=f"M {x},{y} L {x-s/2},{y-s*0.7} L {x+s/2},{y-s*0.7} Z", fillcolor=fill_col, line_color=line_col, line_width=2, row=1, col=1)
        # Wheels
        r_wheel = s * 0.15
        fig.add_shape(type="circle", x0=x-s/3, y0=y-s*0.7-2*r_wheel, x1=x-s/3+2*r_wheel, y1=y-s*0.7, line_color=line_col, fillcolor=fill_col, row=1, col=1)
        fig.add_shape(type="circle", x0=x+s/3-2*r_wheel, y0=y-s*0.7-2*r_wheel, x1=x+s/3, y1=y-s*0.7, line_color=line_col, fillcolor=fill_col, row=1, col=1)
        # Base line
        base_y = y - s*0.7 - 2*r_wheel
        fig.add_shape(type="line", x0=x-s*0.8, y0=base_y, x1=x+s*0.8, y1=base_y, line=dict(color=line_col, width=2), row=1, col=1)

    elif sup_type == "Fixed":
        h_wall = s * 1.6
        # Wall line
        fig.add_shape(type="line", x0=x, y0=y-h_wall/2, x1=x, y1=y+h_wall/2, line=dict(color=line_col, width=5), row=1, col=1)
        # Hatch lines
        direction = -1 if x <= 0.1 else 1 # Hatch direction depends on side
        for hy in np.linspace(y-h_wall/2, y+h_wall/2, 8):
            fig.add_shape(type="line", x0=x, y0=hy, x1=x + (direction * s*0.4), y1=hy - s*0.4, line=dict(color=line_col, width=1.5), row=1, col=1)

def add_peak_annotation(fig, x, y, text, color, row, anchor="bottom"):
    """Helper to add clean peak annotations"""
    fig.add_annotation(
        x=x, y=y,
        text=f"<b>{text}</b>",
        showarrow=True,
        arrowhead=2, arrowsize=1, arrowwidth=1.5,
        arrowcolor=color,
        ax=0, ay=-25 if anchor=="bottom" else 25, # Offset text
        font=dict(color=color, size=11),
        bgcolor="rgba(255,255,255,0.7)", # Background for readability
        bordercolor=color, borderwidth=1, borderpad=3,
        row=row, col=1
    )

def create_engineering_plots(df, vis_spans, vis_supports, loads, unit_sys):
    if not vis_spans or df.empty: return go.Figure()

    cum_len = [0] + list(np.cumsum(vis_spans))
    total_len = cum_len[-1]
    
    force_unit = "kg" if "Metric" in unit_sys else "kN"
    moment_unit = "kg-m" if "Metric" in unit_sys else "kN-m"
    
    # Scaling for load diagram
    w_vals = [abs(l.get('w',0)) for l in loads if l.get('type')=='U']
    p_vals = [abs(l.get('P',0)) for l in loads if l.get('type')=='P']
    max_w = max(w_vals) if w_vals else 1.0
    max_p = max(p_vals) if p_vals else 1.0
    target_h = 1.0 
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("<b>1. Loading Diagram (Factored)</b>", f"<b>2. Shear Force Diagram (SFD) [{force_unit}]</b>", f"<b>3. Bending Moment Diagram (BMD) [{moment_unit}]</b>"),
        row_heights=[0.25, 0.375, 0.375]
    )

    # --- ROW 1: LOADING ---
    fig.add_shape(type="line", x0=0, y0=0, x1=total_len, y1=0, line=dict(color="black", width=4), row=1, col=1)
    
    # Supports (Enhanced)
    sup_size = target_h * 0.3
    for i, x in enumerate(cum_len):
        if i < len(vis_supports):
            stype = vis_supports.iloc[i]['type']
            if stype != "None": draw_support_shape(fig, x, 0, stype, size=sup_size)

    # Loads
    for load in loads:
        span_idx = load.get('span_idx', 0)
        if span_idx >= len(vis_spans): continue 
        x_start = cum_len[span_idx]
        x_end = cum_len[span_idx+1]
        
        if load.get('type') == 'U' and load.get('w', 0) != 0:
            w = load['w']
            ratio = abs(w) / max_w
            h = (0.3 + 0.7 * ratio) * target_h 
            fig.add_trace(go.Scatter(x=[x_start, x_end, x_end, x_start], y=[0, 0, h, h], fill='toself', fillcolor='rgba(255, 152, 0, 0.2)', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=[x_start, x_end], y=[h, h], mode='lines', line=dict(color='#EF6C00', width=2), showlegend=False, hoverinfo='text', text=f"UDL: {w:.1f}"), row=1, col=1)
            fig.add_annotation(x=(x_start+x_end)/2, y=h, text=f"w={w:.0f}", showarrow=False, yshift=10, font=dict(color="#EF6C00", size=10), row=1, col=1)

        elif load.get('type') == 'P' and load.get('P', 0) != 0:
            P = load['P']
            load_x = load['x'] + x_start
            ratio = abs(P) / max_p
            h = (0.3 + 0.7 * ratio) * target_h
            fig.add_annotation(x=load_x, y=0, ax=load_x, ay=h, xref="x1", yref="y1", axref="x1", ayref="y1", showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor="#D32F2F", row=1, col=1)
            fig.add_annotation(x=load_x, y=h, text=f"P={P:.0f}", showarrow=False, yshift=15, font=dict(color="#D32F2F", size=11, weight="bold"), row=1, col=1)

    fig.update_yaxes(range=[-target_h*0.5, target_h*1.8], visible=False, row=1, col=1)

    # --- DATA PREP (Global X) ---
    plot_x, plot_v, plot_m = [], [], []
    current_offset = 0.0
    for i in range(len(vis_spans)):
        span_data = df[df['span_id'] == i].copy()
        if span_data.empty: continue
        if i > 0 and span_data['x'].min() < 0.1: span_x = span_data['x'] + current_offset
        else: span_x = span_data['x']
        plot_x.extend(span_x.tolist())
        plot_v.extend(span_data['shear'].tolist())
        plot_m.extend((-span_data['moment']).tolist()) 
        current_offset += vis_spans[i]
    
    # Convert to numpy for easier indexing
    np_x, np_v, np_m = np.array(plot_x), np.array(plot_v), np.array(plot_m)

    # --- ROW 2: SFD (Shear - Reddish) ---
    shear_color = '#D32F2F'
    fig.add_trace(go.Scatter(x=np_x, y=np_v, mode='lines', line=dict(color=shear_color, width=2.5), fill='tozeroy', fillcolor='rgba(211, 47, 47, 0.1)', name="Shear", hovertemplate="V: %{y:.2f}"), row=2, col=1)
    
    # Find and Annotate Max/Min Shear
    v_max, v_min = np_v.max(), np_v.min()
    if abs(v_max) > 0.1:
        idx_max = np.argmax(np_v)
        add_peak_annotation(fig, np_x[idx_max], v_max, f"Max: {v_max:.1f}", shear_color, 2, "bottom")
    if abs(v_min) > 0.1:
        idx_min = np.argmin(np_v)
        add_peak_annotation(fig, np_x[idx_min], v_min, f"Min: {v_min:.1f}", shear_color, 2, "top")


    # --- ROW 3: BMD (Moment - Bluish) ---
    moment_color = '#1976D2'
    fig.add_trace(go.Scatter(x=np_x, y=np_m, mode='lines', line=dict(color=moment_color, width=2.5), fill='tozeroy', fillcolor='rgba(25, 118, 210, 0.1)', name="Moment", hovertemplate="M: %{y:.2f} (Tension Side)"), row=3, col=1)

    # Find and Annotate Max/Min Moment
    m_max, m_min = np_m.max(), np_m.min()
    # Max Positive (Sagging)
    if m_max > 0.1:
        idx_max = np.argmax(np_m)
        add_peak_annotation(fig, np_x[idx_max], m_max, f"+M max: {m_max:.1f}", moment_color, 3, "bottom")
    # Max Negative (Hogging)
    if m_min < -0.1:
        idx_min = np.argmin(np_m)
        add_peak_annotation(fig, np_x[idx_min], m_min, f"-M max: {m_min:.1f}", moment_color, 3, "top")
    
    # --- Final Aesthetics ---
    for r in [2, 3]:
        fig.add_hline(y=0, line_width=1.5, line_color="#37474F", row=r, col=1) # Thicker zero line
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)', zeroline=False, row=r, col=1)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)', row=r, col=1)
        for x in cum_len: fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="#90A4AE", opacity=0.7, row=r, col=1)

    fig.update_layout(height=900, showlegend=False, plot_bgcolor="white", hovermode="x unified", margin=dict(t=60, b=40, l=80, r=40))
    fig.update_xaxes(title_text=f"Position ({dist_unit})", row=3, col=1)

    return fig

# ... (ส่วน draw_reinforcement_profile และ draw_section_real เหมือนเดิม) ...
def draw_reinforcement_profile(spans, design_results, m_bar, s_bar):
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=-0.5, x1=total_len, y1=0.5, line=dict(color="black", width=2), fillcolor="#F5F5F5", layer="below")

    for i in range(len(spans)):
        x0, x1 = cum_len[i], cum_len[i+1]
        res = design_results[i]
        
        if res.get('bot_nb'):
            txt = f"{res['bot_nb']}-{m_bar}"
            fig.add_trace(go.Scatter(x=[x0 + 0.1, x1 - 0.1], y=[-0.35, -0.35], mode='lines+text', line=dict(color='#1565C0', width=5), text=[txt, txt], textposition="bottom center", showlegend=False, hoverinfo='text', hovertext=f"Bottom: {txt}"))
        
        if res.get('top_nb'):
            txt = f"{res['top_nb']}-{m_bar}"
            fig.add_trace(go.Scatter(x=[x0, x1], y=[0.35, 0.35], mode='lines+text', line=dict(color='#D32F2F', width=3), text=[txt, txt], textposition="top center", showlegend=False, hoverinfo='text', hovertext=f"Top: {txt}"))

        mid = (x0+x1)/2
        stir_txt = f"{s_bar} {res.get('stir_text', '-')}"
        fig.add_annotation(x=mid, y=0, text=f"<b>SPAN {i+1}</b><br>Stir: {stir_txt}", showarrow=False, font=dict(size=10), bgcolor="rgba(255,255,255,0.8)")

    for x in cum_len: fig.add_trace(go.Scatter(x=[x], y=[-0.6], mode='markers', marker=dict(symbol="triangle-up", size=15, color="#333"), showlegend=False, hoverinfo='skip'))
    fig.update_layout(title="🏗️ Reinforcement Profile", height=300, xaxis=dict(range=[-0.5, total_len+0.5], showgrid=True), yaxis=dict(visible=False, range=[-1, 1]), margin=dict(l=20,r=20,t=40,b=20), plot_bgcolor='white')
    return fig

def draw_section_real(b_cm, h_cm, cov_cm, nb_bot, nb_top, bd_mm, stir_d_mm, main_name, stir_name, s_val_mm, title):
    fig = go.Figure()
    bd_cm, sd_cm = bd_mm/10.0, stir_d_mm/10.0
    fig.add_shape(type="rect", x0=0, y0=0, x1=b_cm, y1=h_cm, line=dict(color="black", width=3), fillcolor="#FAFAFA")
    
    # Stirrup
    sx0, sy0, sx1, sy1 = cov_cm, cov_cm, b_cm - cov_cm, h_cm - cov_cm
    fig.add_shape(type="rect", x0=sx0, y0=sy0, x1=sx1, y1=sy1, line=dict(color="#C62828", width=3), fillcolor="rgba(0,0,0,0)")
    
    # Bottom Bars
    if nb_bot > 0:
        start_x, end_x = cov_cm + sd_cm + bd_cm/2, b_cm - (cov_cm + sd_cm + bd_cm/2)
        y_pos = cov_cm + sd_cm + bd_cm/2
        x_pos = [start_x] if nb_bot==1 else np.linspace(start_x, end_x, nb_bot)
        for xp in x_pos: fig.add_shape(type="circle", x0=xp-bd_cm/2, y0=y_pos-bd_cm/2, x1=xp+bd_cm/2, y1=y_pos+bd_cm/2, line_color="black", fillcolor="#1565C0")
            
    # Top Bars
    if nb_top > 0:
        start_x, end_x = cov_cm + sd_cm + bd_cm/2, b_cm - (cov_cm + sd_cm + bd_cm/2)
        y_top = h_cm - (cov_cm + sd_cm + bd_cm/2)
        x_pos = [start_x] if nb_top==1 else np.linspace(start_x, end_x, nb_top)
        for xp in x_pos: fig.add_shape(type="circle", x0=xp-bd_cm/2, y0=y_top-bd_cm/2, x1=xp+bd_cm/2, y1=y_top+bd_cm/2, line_color="black", fillcolor="#D32F2F")

    fig.add_annotation(x=b_cm/2, y=h_cm/2, text=f"Top: {nb_top}-{main_name}<br>Bot: {nb_bot}-{main_name}<br>{stir_name}@{s_val_mm}mm", showarrow=False, font=dict(size=14, color="#333"))
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

# Pass factors to input function
with c_load: loads_input = render_custom_load_input(n_span, spans, unit_sys, fact_dl, fact_ll)

# 2. Calculation
if st.button("🚀 Calculate Analysis & Design", type="primary"):
    try:
        st.session_state['loads_input'] = loads_input
        clean_loads = [l for l in loads_input if isinstance(l, dict)]
        
        # A. Analysis
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
    
    # 3.1 PLOT ANALYSIS (Enhanced)
    st.markdown('<div class="sub-header">1️⃣ Analysis Results</div>', unsafe_allow_html=True)
    st.plotly_chart(create_engineering_plots(df, vis_spans, vis_supports_df, loads, unit_sys), use_container_width=True, key="eng_plot")

    # 3.2 RUN DESIGN (Split Top/Bottom)
    st.markdown('<div class="sub-header">2️⃣ Structural Design Results</div>', unsafe_allow_html=True)
    fc, fy, b_cm, h_cm, cov_cm, m_bar, s_bar, man_s_cm = ui.render_design_input(unit_sys)
    
    bar_areas = {'RB6':0.28, 'RB9':0.64, 'DB10':0.79, 'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
    stir_areas = {'RB6':0.28, 'RB9':0.64, 'DB10':0.79, 'DB12':1.13}
    m_area = bar_areas.get(m_bar, 1.13)
    s_area = stir_areas.get(s_bar, 0.28)
    
    span_results = []
    
    for i in range(n_span):
        span_df = df[df['span_id'] == i]
        m_max_pos = span_df['moment'].max()
        m_max_neg = span_df['moment'].min()
        v_max = span_df['shear'].abs().max()
        
        # Design Logic
        res_bot = calculate_rc_design(m_max_pos, v_max, fc, fy, b_cm, h_cm, cov_cm, method, unit_sys, m_area, s_area, man_s_cm) if m_max_pos > 0.01 else {'nb': 0, 'logs': []}
        res_top = calculate_rc_design(abs(m_max_neg), v_max, fc, fy, b_cm, h_cm, cov_cm, method, unit_sys, m_area, s_area, man_s_cm) if m_max_neg < -0.01 else {'nb': 0, 'logs': []}
        res_stir = calculate_rc_design(max(abs(m_max_pos), abs(m_max_neg)), v_max, fc, fy, b_cm, h_cm, cov_cm, method, unit_sys, m_area, s_area, man_s_cm)

        span_results.append({
            'id': i+1,
            'Mu_pos': m_max_pos, 'Mu_neg': m_max_neg, 'Vu': v_max,
            'bot_nb': res_bot.get('nb', 0),
            'top_nb': res_top.get('nb', 0),
            'stir_text': res_stir.get('stirrup_text', 'Err'),
            's_val': res_stir.get('s_value_mm', 0),
            'logs_bot': res_bot.get('logs', []),
            'logs_top': res_top.get('logs', []),
            'logs_stir': res_stir.get('logs', [])
        })

    # 3.3 PLOT PROFILE
    st.plotly_chart(draw_reinforcement_profile(vis_spans, span_results, m_bar, s_bar), use_container_width=True, key="profile_plot")

    # 3.4 DETAILED TABS
    st.markdown("#### 🔍 Section Details & Calculations")
    tabs = st.tabs([f"Span {r['id']}" for r in span_results])
    
    for i, tab in enumerate(tabs):
        r = span_results[i]
        with tab:
            c1, c2 = st.columns([1, 2])
            # Draw Section
            with c1: 
                main_d = int(m_bar[2:])
                stir_d = int(s_bar[2:]) if 'DB' in s_bar else int(s_bar[2:])
                st.plotly_chart(draw_section_real(b_cm, h_cm, cov_cm, r['bot_nb'], r['top_nb'], main_d, stir_d, m_bar, s_bar, r['s_val'], f"Section Span {r['id']}"), use_container_width=True)
                st.info(f"**Forces Span {i+1}:**\n\n- Mu(+) = {r['Mu_pos']:.2f}\n- Mu(-) = {r['Mu_neg']:.2f}\n- Vu max = {r['Vu']:.2f}")

            # Logs Detail
            with c2:
                with st.expander(f"👇 Bottom Steel Calculation (+M)", expanded=True):
                    if r['bot_nb'] > 0:
                        for l in r['logs_bot']: st.write(l)
                    else: st.write("No positive moment significant enough for design.")
                with st.expander(f"👆 Top Steel Calculation (-M)", expanded=False):
                    if r['top_nb'] > 0:
                        for l in r['logs_top']: st.write(l)
                    else: st.write("No negative moment significant enough for design.")
                with st.expander(f"⛓️ Shear Design", expanded=False):
                     for l in r['logs_stir']: st.write(l)
