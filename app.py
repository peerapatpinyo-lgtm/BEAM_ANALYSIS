import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- IMPORT MODULES ---
try:
    from beam_analysis import run_beam_analysis 
    import design_view 
except ImportError as e:
    st.error(f"⚠️ Error: Missing required files (beam_analysis.py or design_view.py). {e}")
    st.stop()

# ==========================================
# 1. SETUP & STYLES
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
# 2. INPUT SECTIONS (KEPT ORIGINAL)
# ==========================================

def render_sidebar():
    """Render Sidebar Inputs for Code, Materials, and Factors"""
    with st.sidebar:
        st.header("⚙️ Design Parameters")
        
        # Unit System
        unit_sys = st.radio("Unit System", ["Metric (kg, m)", "SI (kN, m)"])
        
        st.markdown("---")
        
        # Design Code & Method
        design_code = st.selectbox("Design Code", ["ACI 318-19", "EIT 1007-34", "EIT 1008-38"])
        method = st.radio("Design Method", ["WSD", "SDM"], index=1)
        
        st.markdown("---")
        
        # Load Factors
        st.subheader("Load Factors")
        if method == "SDM":
            col_f1, col_f2 = st.columns(2)
            f_dl = col_f1.number_input("DL Factor", value=1.4, step=0.1)
            f_ll = col_f2.number_input("LL Factor", value=1.7, step=0.1)
        else:
            f_dl, f_ll = 1.0, 1.0
            st.info("WSD uses Service Loads (Factor = 1.0)")
            
        return design_code, method, f_dl, f_ll, unit_sys

def render_geometry_input():
    """Render Geometry Inputs (Spans & Supports)"""
    st.markdown("### 1️⃣ Geometry & Supports")
    
    col_n, col_dummy = st.columns([1, 2])
    n_span = col_n.number_input("Number of Spans", min_value=1, max_value=10, value=2)
    
    # Init Data
    spans = []
    
    # Create columns for spans
    cols = st.columns(n_span)
    for i, col in enumerate(cols):
        l = col.number_input(f"Span {i+1} (m)", min_value=1.0, value=5.0, step=0.5, key=f"span_{i}")
        spans.append(l)
    
    st.markdown("#### Supports Configuration")
    sup_cols = st.columns(n_span + 1)
    support_options = ["Pin", "Roller", "Fixed", "None"]
    
    current_supports = []
    for i, col in enumerate(sup_cols):
        default_idx = 0 if i == 0 else (1 if i < n_span else 1) 
        s_type = col.selectbox(f"Sup {i+1}", support_options, index=default_idx, key=f"sup_{i}")
        current_supports.append(s_type)
    
    x_coords = [0] + list(np.cumsum(spans))
    
    supports_df = pd.DataFrame({
        'x': x_coords,
        'type': current_supports
    })
    
    return n_span, spans, supports_df

# ==========================================
# 3. CUSTOM LOAD INPUT (UPDATED TO SPLIT DL/LL)
# ==========================================
def render_custom_load_input(n_span, spans, unit_sys, f_dl, f_ll):
    st.markdown("### 2️⃣ Applied Loads (Service Loads)")
    st.caption(f"Note: Factors DL={f_dl:.1f}, LL={f_ll:.1f} will be applied automatically.")
    
    loads = []
    
    tabs = st.tabs([f"📍 Span {i+1} (L={spans[i]}m)" for i in range(n_span)])
    
    for i, tab in enumerate(tabs):
        with tab:
            col_main_1, col_main_2 = st.columns([1, 1.2])
            
            # --- 1. Uniform Load (Combined for simplicity in visual, but factored) ---
            with col_main_1:
                st.info(f"**Uniform Load (Span {i+1})**")
                w_dl = st.number_input(f"Dead Load (w_dl)", value=0.0, step=100.0, format="%.2f", key=f"w_dl_{i}")
                w_ll = st.number_input(f"Live Load (w_ll)", value=0.0, step=100.0, format="%.2f", key=f"w_ll_{i}")
                
                # Combine UDL for simpler viewing, but strictly factored
                w_u = (w_dl * f_dl) + (w_ll * f_ll)
                if w_u != 0:
                    loads.append({'span_idx': i, 'type': 'U', 'w': w_u, 'source': 'Combined'})
                    st.success(f"Total Factored UDL: {w_u:.2f}")

            # --- 2. Point Load (SPLIT DL/LL as requested) ---
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
                            
                            # ** KEY CHANGE: Append separately so we can plot separately **
                            if p_dl != 0:
                                loads.append({
                                    'span_idx': i, 'type': 'P', 
                                    'P': p_dl * f_dl,   # Factored Value for Calculation
                                    'raw': p_dl,        # Raw Value for Label
                                    'x': x_loc, 
                                    'source': 'DL'      # Tag for Color/Label
                                })
                            
                            if p_ll != 0:
                                loads.append({
                                    'span_idx': i, 'type': 'P', 
                                    'P': p_ll * f_ll,   # Factored Value for Calculation
                                    'raw': p_ll,        # Raw Value for Label
                                    'x': x_loc, 
                                    'source': 'LL'      # Tag for Color/Label
                                })

                else:
                    st.caption("No point loads on this span.")
    return loads

# ==========================================
# 4. VISUALIZATION FUNCTIONS (PROFESSIONAL & REFINED)
# ==========================================
def draw_support_shape(fig, x, y, sup_type, size=1.0):
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

def add_peak_annotation(fig, x, y, val, color, row, position="top"):
    """
    Adds a clean box annotation without arrow.
    Shows Value AND Distance (x).
    """
    yshift = 15 if position == "top" else -15
    
    # Text Formatting
    label_text = f"<b>{val:.2f}</b><br><span style='font-size:10px; color:#555'>@ {x:.2f} m</span>"
    
    fig.add_annotation(
        x=x, y=y,
        text=label_text,
        showarrow=False,  # REMOVED ARROW
        font=dict(color=color, size=11),
        bgcolor="rgba(255, 255, 255, 0.85)", # Semi-transparent white box
        bordercolor=color,
        borderwidth=1,
        borderpad=4,
        yshift=yshift,
        row=row, col=1
    )

def create_engineering_plots(df, vis_spans, vis_supports, loads, unit_sys):
    if not vis_spans or df.empty: return go.Figure()

    cum_len = [0] + list(np.cumsum(vis_spans))
    total_len = cum_len[-1]
    
    force_unit = "kg" if "Metric" in unit_sys else "kN"
    moment_unit = "kg-m" if "Metric" in unit_sys else "kN-m"
    dist_unit = "m"
    
    # --- GLOBAL SCALING LOGIC ---
    all_magnitudes = []
    for l in loads:
        if l['type'] == 'U': all_magnitudes.append(abs(l.get('w', 0)))
        if l['type'] == 'P': all_magnitudes.append(abs(l.get('P', 0)))
    
    global_max = max(all_magnitudes) if all_magnitudes and max(all_magnitudes) > 0 else 1.0
    target_h = 1.2 
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(
            "<b>1. Loading Diagram (Free Body Diagram)</b>", 
            f"<b>2. Shear Force Diagram (SFD)</b>", 
            f"<b>3. Bending Moment Diagram (BMD)</b>"
        ),
        row_heights=[0.3, 0.35, 0.35]
    )

    # --- 1. Loading Diagram ---
    fig.add_shape(type="line", x0=0, y0=0, x1=total_len, y1=0, line=dict(color="black", width=4), row=1, col=1)
    
    # Supports
    sup_size = target_h * 0.25
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
            ratio = abs(w) / global_max
            h = (0.15 + 0.85 * ratio) * target_h 
            
            # UDL uses a neutral orange
            fig.add_trace(go.Scatter(x=[x_start, x_end, x_end, x_start], y=[0, 0, h, h], fill='toself', fillcolor='rgba(255, 152, 0, 0.2)', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=[x_start, x_end], y=[h, h], mode='lines', line=dict(color='#EF6C00', width=2), showlegend=False, hoverinfo='text', text=f"Total w: {w:.1f}"), row=1, col=1)
            fig.add_annotation(x=(x_start+x_end)/2, y=h, text=f"w={w:.0f}", showarrow=False, yshift=10, font=dict(color="#EF6C00", size=10), row=1, col=1)

        # --- POINT LOAD (SEPARATE DL/LL) ---
        elif load.get('type') == 'P' and load.get('P', 0) != 0:
            P = load['P']
            load_x = load['x'] + x_start
            ratio = abs(P) / global_max
            h = (0.15 + 0.85 * ratio) * target_h
            
            # Determine Color & Label based on Source (DL vs LL)
            source = load.get('source', '')
            if source == 'DL':
                p_color = "#1565C0" # Dark Blue
                label_text = f"P_DL={load.get('raw', P):.0f}"
                x_offset = -0.05 # Slight offset to prevent text overlap if stacked (though arrows overlap)
            elif source == 'LL':
                p_color = "#C62828" # Dark Red
                label_text = f"P_LL={load.get('raw', P):.0f}"
                x_offset = 0.05
            else:
                p_color = "#333333"
                label_text = f"P={P:.0f}"
                x_offset = 0

            # Draw Arrow
            fig.add_annotation(
                x=load_x, y=0,              
                ax=load_x, ay=h,            
                xref='x1', yref='y1',       
                axref='x1', ayref='y1',     
                text=f"<b>{label_text}</b>",          
                showarrow=True,
                arrowhead=2,                
                arrowsize=1.2,
                arrowwidth=2.5,
                arrowcolor=p_color,
                yshift=10,
                xshift=x_offset * 100, # Pixel shift for label
                font=dict(color=p_color, size=10, weight="bold"),
                row=1, col=1
            )

    fig.update_yaxes(range=[-target_h*0.5, target_h*1.5], visible=False, row=1, col=1)

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
        idx_max, idx_min = np.argmax(np_v), np.argmin(np_v)
        
        # Add Peak Annotations (No Arrow, Clean Box, With Distance)
        if abs(v_max) > 0.1: 
            add_peak_annotation(fig, np_x[idx_max], v_max, v_max, shear_color, 2, "top" if v_max > 0 else "bottom")
        if abs(v_min) > 0.1: 
            add_peak_annotation(fig, np_x[idx_min], v_min, v_min, shear_color, 2, "bottom" if v_min < 0 else "top")

    # Plot BMD
    moment_color = '#1976D2'
    fig.add_trace(go.Scatter(x=np_x, y=np_m, mode='lines', line=dict(color=moment_color, width=2.5), fill='tozeroy', fillcolor='rgba(25, 118, 210, 0.1)', name="Moment", hovertemplate="M: %{y:.2f}"), row=3, col=1)
    if len(np_m) > 0:
        m_max, m_min = np_m.max(), np_m.min()
        idx_max, idx_min = np.argmax(np_m), np.argmin(np_m)
        
        if abs(m_max) > 0.1: 
            add_peak_annotation(fig, np_x[idx_max], m_max, m_max, moment_color, 3, "top" if m_max > 0 else "bottom")
        if abs(m_min) > 0.1: 
            add_peak_annotation(fig, np_x[idx_min], m_min, m_min, moment_color, 3, "bottom" if m_min < 0 else "top")
    
    # --- PROFESSIONAL LAYOUT SETTINGS ---
    # Axis Titles & Grid
    fig.update_yaxes(title_text=f"Shear V ({force_unit})", showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)', zeroline=True, zerolinewidth=1.5, zerolinecolor='#455A64', row=2, col=1)
    fig.update_yaxes(title_text=f"Moment M ({moment_unit})", showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)', zeroline=True, zerolinewidth=1.5, zerolinecolor='#455A64', row=3, col=1)
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)', row=2, col=1)
    fig.update_xaxes(title_text=f"Position ({dist_unit})", showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)', row=3, col=1)
    
    # Span Separators
    for r in [2, 3]:
        for x in cum_len: fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="#90A4AE", opacity=0.7, row=r, col=1)

    fig.update_layout(height=1000, showlegend=False, plot_bgcolor="white", hovermode="x unified", margin=dict(t=60, b=40, l=80, r=40))
    
    return fig

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
st.markdown('<div class="header-box"><h2>🏗️ RC Beam Pro Ultimate</h2></div>', unsafe_allow_html=True)

# --- 1. Calls Sidebar & Geometry ---
design_code, method, fact_dl, fact_ll, unit_sys = render_sidebar()

c_geo, c_load = st.columns([1, 1.5])
with c_geo:
    n_span, spans, supports = render_geometry_input()

with c_load:
    # Use Updated Load Input (Splits DL/LL)
    loads_input = render_custom_load_input(n_span, spans, unit_sys, fact_dl, fact_ll)

# --- 2. Calculation ---
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

# --- 3. Visualization ---
if st.session_state['analyzed'] and st.session_state.get('res_df') is not None:
    df = st.session_state['res_df']
    vis_spans, vis_supports_df = st.session_state['vis_data']
    loads = st.session_state['loads_input']
    
    st.markdown('<div class="sub-header">1️⃣ Analysis Results</div>', unsafe_allow_html=True)
    st.plotly_chart(create_engineering_plots(df, vis_spans, vis_supports_df, loads, unit_sys), use_container_width=True, key="eng_plot")

    # B. Design
    design_view.render_design_section(df, vis_spans, unit_sys, method)
