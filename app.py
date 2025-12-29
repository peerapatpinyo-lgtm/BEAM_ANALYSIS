import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- IMPORT MODULES ---
try:
    from beam_analysis import run_beam_analysis 
    import input_handler as ui
    import design_view 
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
    .load-box { border: 1px solid #E0E0E0; padding: 15px; border-radius: 8px; background-color: #FAFAFA; margin-bottom: 10px; }
    .stNumberInput input { font-weight: bold; color: #1565C0; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CUSTOM LOAD INPUT
# ==========================================
def render_custom_load_input(n_span, spans, unit_sys, f_dl, f_ll):
    st.markdown("### 3️⃣ Applied Loads (Service Loads)")
    st.caption(f"Note: Factors DL={f_dl:.1f}, LL={f_ll:.1f} will be applied automatically.")
    
    force_unit = "kg" if "Metric" in unit_sys else "kN"
    dist_unit = "m"
    loads = []
    
    tabs = st.tabs([f"📍 Span {i+1} (L={spans[i]}m)" for i in range(n_span)])
    
    for i, tab in enumerate(tabs):
        with tab:
            col_main_1, col_main_2 = st.columns([1, 1.2])
            
            # --- 1. Uniform Load ---
            with col_main_1:
                st.info(f"**Uniform Load (Span {i+1})**")
                w_dl = st.number_input(f"Dead Load (w_dl)", value=0.0, step=100.0, format="%.2f", key=f"w_dl_{i}", help=f"Unit: {force_unit}/{dist_unit}")
                w_ll = st.number_input(f"Live Load (w_ll)", value=0.0, step=100.0, format="%.2f", key=f"w_ll_{i}", help=f"Unit: {force_unit}/{dist_unit}")
                
                w_u = (w_dl * f_dl) + (w_ll * f_ll)
                if w_u != 0:
                    loads.append({'span_idx': i, 'type': 'U', 'w': w_u})
                    st.success(f"Added UDL: {w_u:.2f}")

            # --- 2. Point Load ---
            with col_main_2:
                st.warning(f"**Point Loads (Span {i+1})**")
                num_p = st.number_input(f"Add Point Loads (Qty)", min_value=0, max_value=5, value=0, key=f"qty_p_{i}")

                if num_p > 0:
                    for j in range(num_p):
                        with st.container():
                            st.markdown(f"""<div class="load-box"><b>Point Load #{j+1}</b></div>""", unsafe_allow_html=True)
                            c1, c2, c3 = st.columns([1, 1, 1.2])
                            
                            p_dl = c1.number_input(f"P_DL", value=0.0, step=100.0, key=f"p_dl_{i}_{j}")
                            p_ll = c2.number_input(f"P_LL", value=0.0, step=100.0, key=f"p_ll_{i}_{j}")
                            
                            x_max = float(spans[i])
                            x_loc = c3.number_input(f"x (0-{x_max}m)", min_value=0.0, max_value=x_max, value=x_max/2, step=0.1, key=f"p_x_{i}_{j}")
                            
                            p_u = (p_dl * f_dl) + (p_ll * f_ll)
                            if p_u != 0:
                                loads.append({
                                    'span_idx': i, 
                                    'type': 'P', 
                                    'P': p_u, 
                                    'x': x_loc
                                })
                else:
                    st.caption("No point loads on this span.")
    return loads

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================
def draw_support_shape(fig, x, y, sup_type, size=1.0):
    """Draws ENHANCED Engineering Supports"""
    s = size * 0.9 
    line_col, fill_col = "#263238", "#B0BEC5" 
    
    if sup_type == "Pin":
        fig.add_shape(type="path", path=f"M {x},{y} L {x-s/2},{y-s} L {x+s/2},{y-s} Z", fillcolor=fill_col, line_color=line_col, line_width=2, row=1, col=1)
        fig.add_shape(type="line", x0=x-s*0.8, y0=y-s, x1=x+s*0.8, y1=y-s, line=dict(color=line_col, width=3), row=1, col=1)
        for hx in np.linspace(x-s*0.8, x+s*0.8, 6): 
            fig.add_shape(type="line", x0=hx, y0=y-s, x1=hx-s/3, y1=y-s*1.4, line=dict(color=line_col, width=1), row=1, col=1)

    elif sup_type == "Roller":
        fig.add_shape(type="path", path=f"M {x},{y} L {x-s/2},{y-s*0.7} L {x+s/2},{y-s*0.7} Z", fillcolor=fill_col, line_color=line_col, line_width=2, row=1, col=1)
        r_wheel = s * 0.15
        fig.add_shape(type="circle", x0=x-s/3, y0=y-s*0.7-2*r_wheel, x1=x-s/3+2*r_wheel, y1=y-s*0.7, line_color=line_col, fillcolor=fill_col, row=1, col=1)
        fig.add_shape(type="circle", x0=x+s/3-2*r_wheel, y0=y-s*0.7-2*r_wheel, x1=x+s/3, y1=y-s*0.7, line_color=line_col, fillcolor=fill_col, row=1, col=1)
        base_y = y - s*0.7 - 2*r_wheel
        fig.add_shape(type="line", x0=x-s*0.8, y0=base_y, x1=x+s*0.8, y1=base_y, line=dict(color=line_col, width=2), row=1, col=1)

    elif sup_type == "Fixed":
        h_wall = s * 1.6
        fig.add_shape(type="line", x0=x, y0=y-h_wall/2, x1=x, y1=y+h_wall/2, line=dict(color=line_col, width=5), row=1, col=1)
        direction = -1 if x <= 0.1 else 1 
        for hy in np.linspace(y-h_wall/2, y+h_wall/2, 8):
            fig.add_shape(type="line", x0=x, y0=hy, x1=x + (direction * s*0.4), y1=hy - s*0.4, line=dict(color=line_col, width=1.5), row=1, col=1)

def add_peak_annotation(fig, x, y, text, color, row, anchor="bottom"):
    """Helper to add clean peak annotations"""
    fig.add_annotation(
        x=x, y=y,
        text=f"<b>{text}</b>",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
        arrowcolor=color, ax=0, ay=-25 if anchor=="bottom" else 25, 
        font=dict(color=color, size=11), bgcolor="rgba(255,255,255,0.7)",
        bordercolor=color, borderwidth=1, borderpad=3, row=row, col=1
    )

def create_engineering_plots(df, vis_spans, vis_supports, loads, unit_sys):
    if not vis_spans or df.empty: return go.Figure()

    cum_len = [0] + list(np.cumsum(vis_spans))
    total_len = cum_len[-1]
    
    force_unit = "kg" if "Metric" in unit_sys else "kN"
    moment_unit = "kg-m" if "Metric" in unit_sys else "kN-m"
    dist_unit = "m"
    
    # --- SCALING ---
    w_vals = [abs(l.get('w',0)) for l in loads if l.get('type')=='U']
    p_vals = [abs(l.get('P',0)) for l in loads if l.get('type')=='P']
    
    max_w = max(w_vals) if w_vals and max(w_vals) > 0 else 1.0
    max_p = max(p_vals) if p_vals and max(p_vals) > 0 else 1.0
    
    target_h = 1.0 
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("<b>1. Loading Diagram (Factored)</b>", f"<b>2. Shear Force Diagram (SFD) [{force_unit}]</b>", f"<b>3. Bending Moment Diagram (BMD) [{moment_unit}]</b>"),
        row_heights=[0.25, 0.375, 0.375]
    )

    # --- 1. Loading Diagram ---
    fig.add_shape(type="line", x0=0, y0=0, x1=total_len, y1=0, line=dict(color="black", width=4), row=1, col=1)
    
    # Supports
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
        
        # --- UDL ---
        if load.get('type') == 'U' and load.get('w', 0) != 0:
            w = load['w']
            ratio = abs(w) / max_w
            h = (0.2 + 0.8 * ratio) * target_h 
            
            fig.add_trace(go.Scatter(x=[x_start, x_end, x_end, x_start], y=[0, 0, h, h], fill='toself', fillcolor='rgba(255, 152, 0, 0.25)', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=[x_start, x_end], y=[h, h], mode='lines', line=dict(color='#EF6C00', width=2), showlegend=False, hoverinfo='text', text=f"UDL: {w:.1f}"), row=1, col=1)
            fig.add_annotation(x=(x_start+x_end)/2, y=h, text=f"w={w:.0f}", showarrow=False, yshift=10, font=dict(color="#EF6C00", size=10), row=1, col=1)

        # --- POINT LOAD (FIXED) ---
        elif load.get('type') == 'P' and load.get('P', 0) != 0:
            P = load['P']
            load_x = load['x'] + x_start
            ratio = abs(P) / max_p
            h = (0.25 + 0.75 * ratio) * target_h
            
            # Correct Arrow Drawing: Head(x, 0) <- Tail(ax, ay)
            # Use 'x1' and 'y1' refs to ensure we are plotting in the correct subplot coordinates
            fig.add_annotation(
                x=load_x, y=0,              # Head position (on beam)
                ax=load_x, ay=h,            # Tail position (height h)
                xref='x1', yref='y1',       # Data coordinates for Head
                axref='x1', ayref='y1',     # Data coordinates for Tail
                text=f"P={P:.0f}",          # Label on the arrow tail
                showarrow=True,
                arrowhead=2,                
                arrowsize=1.2,
                arrowwidth=2.5,
                arrowcolor="#D32F2F",
                yshift=10,                  
                font=dict(color="#D32F2F", size=11, weight="bold"),
                row=1, col=1
            )

    fig.update_yaxes(range=[-target_h*0.5, target_h*1.8], visible=False, row=1, col=1)

    # --- 2. SFD & 3. BMD Data Prep ---
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
    
    np_x, np_v, np_m = np.array(plot_x), np.array(plot_v), np.array(plot_m)

    # Plot SFD
    shear_color = '#D32F2F'
    fig.add_trace(go.Scatter(x=np_x, y=np_v, mode='lines', line=dict(color=shear_color, width=2.5), fill='tozeroy', fillcolor='rgba(211, 47, 47, 0.1)', name="Shear", hovertemplate="V: %{y:.2f}"), row=2, col=1)
    if len(np_v) > 0:
        v_max, v_min = np_v.max(), np_v.min()
        if abs(v_max) > 0.1: add_peak_annotation(fig, np_x[np.argmax(np_v)], v_max, f"Max: {v_max:.1f}", shear_color, 2, "bottom")
        if abs(v_min) > 0.1: add_peak_annotation(fig, np_x[np.argmin(np_v)], v_min, f"Min: {v_min:.1f}", shear_color, 2, "top")

    # Plot BMD
    moment_color = '#1976D2'
    fig.add_trace(go.Scatter(x=np_x, y=np_m, mode='lines', line=dict(color=moment_color, width=2.5), fill='tozeroy', fillcolor='rgba(25, 118, 210, 0.1)', name="Moment", hovertemplate="M: %{y:.2f}"), row=3, col=1)
    if len(np_m) > 0:
        m_max, m_min = np_m.max(), np_m.min()
        if m_max > 0.1: add_peak_annotation(fig, np_x[np.argmax(np_m)], m_max, f"+M: {m_max:.1f}", moment_color, 3, "bottom")
        if m_min < -0.1: add_peak_annotation(fig, np_x[np.argmin(np_m)], m_min, f"-M: {m_min:.1f}", moment_color, 3, "top")
    
    for r in [2, 3]:
        fig.add_hline(y=0, line_width=1.5, line_color="#37474F", row=r, col=1) 
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)', zeroline=False, row=r, col=1)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)', row=r, col=1)
        for x in cum_len: fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="#90A4AE", opacity=0.7, row=r, col=1)

    fig.update_layout(height=900, showlegend=False, plot_bgcolor="white", hovermode="x unified", margin=dict(t=60, b=40, l=80, r=40))
    fig.update_xaxes(title_text=f"Position ({dist_unit})", row=3, col=1)

    return fig

# ==========================================
# MAIN APP EXECUTION
# ==========================================
st.markdown('<div class="header-box"><h2>🏗️ RC Beam Pro Ultimate</h2></div>', unsafe_allow_html=True)

# 1. Inputs
design_code, method, fact_dl, fact_ll, unit_sys = ui.render_sidebar()
c_geo, c_load = st.columns([1, 1.5])
with c_geo: n_span, spans, supports = ui.render_geometry_input()

with c_load: loads_input = render_custom_load_input(n_span, spans, unit_sys, fact_dl, fact_ll)

# 2. Calculation
if st.button("🚀 Calculate Analysis & Design", type="primary"):
    try:
        st.session_state['loads_input'] = loads_input
        # กรองข้อมูลให้เหลือแต่ dict ที่ถูกต้องก่อนส่งเข้าคำนวณ
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
    
    st.markdown('<div class="sub-header">1️⃣ Analysis Results</div>', unsafe_allow_html=True)
    st.plotly_chart(create_engineering_plots(df, vis_spans, vis_supports_df, loads, unit_sys), use_container_width=True, key="eng_plot")

    # เรียกใช้ไฟล์ design_view ที่แยกออกไป
    design_view.render_design_section(df, vis_spans, unit_sys, method)
