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
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; font-size: 16px; }
    
    .header-box { background: linear-gradient(90deg, #1A237E 0%, #283593 100%); color: white; padding: 25px; border-radius: 8px; text-align: center; margin-bottom: 25px; box-shadow: 0 4px 10px rgba(0,0,0,0.15); }
    
    .sub-header { border-left: 6px solid #1A237E; padding-left: 15px; font-size: 1.4rem; font-weight: 700; margin-top: 30px; margin-bottom: 20px; background: #E8EAF6; padding: 12px; border-radius: 0 8px 8px 0; color: #1A237E; }
    
    .load-box { border: 1px solid #BDBDBD; padding: 15px; border-radius: 8px; background-color: #FAFAFA; margin-bottom: 15px; }
    
    .calc-box { 
        font-family: 'Courier New', monospace; 
        background-color: #F1F8E9; 
        border-left: 5px solid #33691E; 
        padding: 12px; 
        margin-top: 8px; 
        font-size: 1rem; 
        color: #1B5E20; 
        border-radius: 4px;
        font-weight: 600;
    }
    
    .error-box { background-color: #FFEBEE; border: 2px solid #D32F2F; color: #B71C1C; padding: 15px; border-radius: 8px; margin: 15px 0; font-weight: bold; font-size: 1.1rem; }
    
    .stNumberInput input { font-weight: bold; color: #0D47A1; font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ENGINEERING LOGIC (STABILITY CHECK)
# ==========================================

def check_structural_stability(supports_df):
    """
    Engineering Stability Check:
    1. Must have supports.
    2. Single support must be Fixed (Cantilever).
    3. Cannot have only Rollers (Horizontal instability).
    4. 3+ parallel rollers without horizontal restraint is technically unstable in X, but permitted in simple beam theory if vertical load only. 
       However, strictly speaking, we need at least one Pin or Fixed.
    """
    if supports_df.empty:
        return False, "❌ CRITICAL ERROR: No supports defined."

    active_sups = supports_df[supports_df['type'] != 'None']
    types = active_sups['type'].tolist()
    num_sup = len(types)
    
    # 1. No Supports
    if num_sup == 0:
        return False, "❌ UNSTABLE: Structure is a Mechanism (Free Falling)."
    
    # 2. Single Support
    if num_sup == 1:
        s_type = types[0]
        if s_type == "Fixed":
            return True, "✅ Stable (Cantilever)"
        else:
            return False, f"❌ UNSTABLE: Single '{s_type}' support cannot provide stability (Rotation/Translation)."

    # 3. All Rollers (Horizontal Instability)
    if all(t == "Roller" for t in types):
        return False, "❌ UNSTABLE: All supports are Rollers. Structure is unstable horizontally."

    return True, "✅ Stable"

# ==========================================
# 3. INPUT SECTIONS
# ==========================================

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Design Parameters")
        unit_sys = st.radio("Unit System", ["Metric (kg, m)", "SI (kN, m)"])
        st.markdown("---")
        design_code = st.selectbox("Design Code", ["ACI 318-19", "EIT 1007-34", "EIT 1008-38"])
        method = st.radio("Design Method", ["WSD", "SDM"], index=1)
        st.markdown("---")
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
    st.markdown("### 1️⃣ Geometry & Supports")
    col_n, _ = st.columns([1, 2])
    n_span = col_n.number_input("Number of Spans", min_value=1, max_value=10, value=2)
    
    spans = []
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
    supports_df = pd.DataFrame({'x': x_coords, 'type': current_supports})
    
    # Live Stability Check
    is_stable, msg = check_structural_stability(supports_df)
    if not is_stable:
        st.markdown(f'<div class="error-box">{msg}</div>', unsafe_allow_html=True)
    
    return n_span, spans, supports_df, is_stable

def render_custom_load_input(n_span, spans, unit_sys, f_dl, f_ll):
    st.markdown("### 2️⃣ Applied Loads (Combined DL + LL)")
    st.caption(f"**Engineering Note:** Loads will be combined using $U = {f_dl}DL + {f_ll}LL$")
    
    loads = []
    tabs = st.tabs([f"📍 Span {i+1} (L={spans[i]}m)" for i in range(n_span)])
    
    for i, tab in enumerate(tabs):
        with tab:
            col_main_1, col_main_2 = st.columns([1, 1.2])
            
            # --- Uniform Load ---
            with col_main_1:
                st.info(f"**Uniform Load (Span {i+1})**")
                w_dl = st.number_input(f"Dead Load (W_DL)", value=0.0, step=100.0, format="%.2f", key=f"w_dl_{i}")
                w_ll = st.number_input(f"Live Load (W_LL)", value=0.0, step=100.0, format="%.2f", key=f"w_ll_{i}")
                
                w_u = (w_dl * f_dl) + (w_ll * f_ll)
                if w_u != 0:
                    loads.append({'span_idx': i, 'type': 'U', 'w': w_u, 'source': 'Total UDL'})
                    st.markdown(f"""
                    <div class="calc-box">
                    CALCULATION (Wu):<br>
                    = ({f_dl} × {w_dl}) + ({f_ll} × {w_ll})<br>
                    = <b>{w_u:,.2f}</b>
                    </div>
                    """, unsafe_allow_html=True)

            # --- Point Load ---
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
                            
                            # COMBINE LOADS HERE (AS REQUESTED)
                            p_total_factored = (p_dl * f_dl) + (p_ll * f_ll)
                            
                            if p_total_factored != 0:
                                # Append ONE combined load
                                loads.append({
                                    'span_idx': i, 
                                    'type': 'P', 
                                    'P': p_total_factored, # Combined Value
                                    'x': x_loc, 
                                    'source': 'Total Point'
                                })
                                
                                st.markdown(f"""
                                <div class="calc-box">
                                CALCULATION (Pu):<br>
                                = ({f_dl} × {p_dl}) + ({f_ll} × {p_ll})<br>
                                = <b>{p_total_factored:,.2f}</b>
                                </div>
                                """, unsafe_allow_html=True)
    return loads

# ==========================================
# 4. VISUALIZATION ENGINE (BIG FONTS & CLEANER)
# ==========================================

def draw_support_shape(fig, x, y, sup_type, size=1.0):
    s = size * 0.9 
    line_col, fill_col = "#263238", "#CFD8DC"
    
    if sup_type == "Pin":
        fig.add_shape(type="path", path=f"M {x},{y} L {x-s/2},{y-s} L {x+s/2},{y-s} Z", fillcolor=fill_col, line_color=line_col, line_width=2, row=1, col=1)
        fig.add_shape(type="line", x0=x-s*0.8, y0=y-s, x1=x+s*0.8, y1=y-s, line=dict(color=line_col, width=3), row=1, col=1)
        # Ground hatching
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

def add_peak_box(fig, x, y, val, unit, color, row, position="top"):
    """ Box with Larger Font and clear border """
    yshift = 30 if position == "top" else -30
    
    label_text = f"<b>{val:,.2f} {unit}</b><br><span style='font-size:12px; color:#37474F'>@ {x:.2f} m</span>"
    
    fig.add_annotation(
        x=x, y=y,
        text=label_text,
        showarrow=False,
        font=dict(color=color, size=14, family="Arial Black"), # Increased Font
        align="center",
        bgcolor="rgba(255, 255, 255, 1.0)",
        bordercolor=color,
        borderwidth=2,
        borderpad=6,
        yshift=yshift,
        row=row, col=1
    )

def create_engineering_plots(df, vis_spans, vis_supports, loads, unit_sys, factors):
    if not vis_spans or df.empty: return go.Figure()

    cum_len = [0] + list(np.cumsum(vis_spans))
    total_len = cum_len[-1]
    
    force_unit = "kg" if "Metric" in unit_sys else "kN"
    moment_unit = "kg-m" if "Metric" in unit_sys else "kN-m"
    dist_unit = "m"
    f_dl, f_ll = factors
    
    # Global Scaling
    all_magnitudes = []
    for l in loads:
        val = abs(l.get('w', 0)) if l['type'] == 'U' else abs(l.get('P', 0))
        if val > 0: all_magnitudes.append(val)
    global_max = max(all_magnitudes) if all_magnitudes else 1.0
    target_h = 1.5 
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(
            f"<b>1. FACTORED LOADING DIAGRAM (Total Load = {f_dl}DL + {f_ll}LL)</b>", 
            f"<b>2. SHEAR FORCE DIAGRAM (SFD)</b>", 
            f"<b>3. BENDING MOMENT DIAGRAM (BMD)</b>"
        ),
        row_heights=[0.3, 0.35, 0.35]
    )

    # --- ROW 1: LOADING ---
    fig.add_shape(type="line", x0=0, y0=0, x1=total_len, y1=0, line=dict(color="black", width=5), row=1, col=1)
    
    # Supports
    sup_size = target_h * 0.2
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
            
            # Fill Area
            fig.add_trace(go.Scatter(x=[x_start, x_end, x_end, x_start], y=[0, 0, h, h], fill='toself', fillcolor='rgba(255, 111, 0, 0.15)', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
            # Top Line
            fig.add_trace(go.Scatter(x=[x_start, x_end], y=[h, h], mode='lines', line=dict(color='#E65100', width=3), showlegend=False, hoverinfo='text', text=f"Total w: {w:,.1f}"), row=1, col=1)
            # Label
            fig.add_annotation(x=(x_start+x_end)/2, y=h, text=f"<b>Wu = {w:,.0f}</b>", showarrow=False, yshift=15, font=dict(color="#E65100", size=14, weight="bold"), row=1, col=1)

        # --- POINT LOAD (COMBINED) ---
        elif load.get('type') == 'P' and load.get('P', 0) != 0:
            P = load['P']
            local_x = load['x']
            vis_x = local_x + x_start
            
            ratio = abs(P) / global_max
            h = (0.15 + 0.85 * ratio) * target_h
            
            # Use a Dark color for Combined Point Load
            p_color = "#3E2723" # Dark Brown/Black
            
            fig.add_annotation(
                x=vis_x, y=0,              
                ax=vis_x, ay=h,            
                xref='x1', yref='y1', axref='x1', ayref='y1',     
                text=f"<b>Pu = {P:,.0f}</b>",          
                showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor=p_color,
                yshift=15,
                font=dict(color=p_color, size=14, weight="bold"),
                row=1, col=1
            )

    fig.update_yaxes(range=[-target_h*0.5, target_h*1.5], visible=False, row=1, col=1)

    # --- SFD & BMD ---
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

    # SFD
    shear_color = '#C62828'
    fig.add_trace(go.Scatter(x=np_x, y=np_v, mode='lines', line=dict(color=shear_color, width=3), fill='tozeroy', fillcolor='rgba(198, 40, 40, 0.1)', name="Shear", hovertemplate="V: %{y:.2f}"), row=2, col=1)
    if len(np_v) > 0:
        v_max, v_min = np_v.max(), np_v.min()
        idx_max, idx_min = np.argmax(np_v), np.argmin(np_v)
        if abs(v_max) > 0.1: add_peak_box(fig, np_x[idx_max], v_max, v_max, force_unit, shear_color, 2, "top" if v_max > 0 else "bottom")
        if abs(v_min) > 0.1: add_peak_box(fig, np_x[idx_min], v_min, v_min, force_unit, shear_color, 2, "bottom" if v_min < 0 else "top")

    # BMD
    moment_color = '#1565C0'
    fig.add_trace(go.Scatter(x=np_x, y=np_m, mode='lines', line=dict(color=moment_color, width=3), fill='tozeroy', fillcolor='rgba(21, 101, 192, 0.1)', name="Moment", hovertemplate="M: %{y:.2f}"), row=3, col=1)
    if len(np_m) > 0:
        m_max, m_min = np_m.max(), np_m.min()
        idx_max, idx_min = np.argmax(np_m), np.argmin(np_m)
        if abs(m_max) > 0.1: add_peak_box(fig, np_x[idx_max], m_max, m_max, moment_unit, moment_color, 3, "top" if m_max > 0 else "bottom")
        if abs(m_min) > 0.1: add_peak_box(fig, np_x[idx_min], m_min, m_min, moment_unit, moment_color, 3, "bottom" if m_min < 0 else "top")
    
    # Layout Config (Big Fonts)
    axis_font_size = 16
    fig.update_yaxes(title_text=f"<b>Shear V ({force_unit})</b>", title_font=dict(size=axis_font_size), tickfont=dict(size=12), showgrid=True, gridwidth=1, gridcolor='#ECEFF1', zeroline=True, zerolinewidth=2, zerolinecolor='#546E7A', row=2, col=1)
    fig.update_yaxes(title_text=f"<b>Moment M ({moment_unit})</b>", title_font=dict(size=axis_font_size), tickfont=dict(size=12), showgrid=True, gridwidth=1, gridcolor='#ECEFF1', zeroline=True, zerolinewidth=2, zerolinecolor='#546E7A', row=3, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#ECEFF1', row=2, col=1)
    fig.update_xaxes(title_text=f"<b>Position ({dist_unit})</b>", title_font=dict(size=axis_font_size), tickfont=dict(size=12), showgrid=True, gridwidth=1, gridcolor='#ECEFF1', row=3, col=1)
    
    for r in [2, 3]:
        for x in cum_len: fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="#90A4AE", opacity=0.7, row=r, col=1)

    # Increase Subplot Title Font
    fig.update_annotations(font_size=16)

    fig.update_layout(height=1100, showlegend=False, plot_bgcolor="white", hovermode="x unified", margin=dict(t=80, b=50, l=100, r=40))
    return fig

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
st.markdown('<div class="header-box"><h2>🏗️ RC Beam Pro Ultimate</h2></div>', unsafe_allow_html=True)

# 1. Sidebar & Geometry
design_code, method, fact_dl, fact_ll, unit_sys = render_sidebar()
c_geo, c_load = st.columns([1, 1.5])

with c_geo:
    n_span, spans, supports, is_stable = render_geometry_input()

with c_load:
    loads_input = render_custom_load_input(n_span, spans, unit_sys, fact_dl, fact_ll)

# 2. Calculation
if st.button("🚀 Calculate Analysis & Design", type="primary"):
    if not is_stable:
        st.error("⛔ Cannot Calculate: Structure is Unstable. Please check supports.")
    else:
        try:
            st.session_state['loads_input'] = loads_input
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
    st.plotly_chart(create_engineering_plots(df, vis_spans, vis_supports_df, loads, unit_sys, (fact_dl, fact_ll)), use_container_width=True, key="eng_plot")
    
    design_view.render_design_section(df, vis_spans, unit_sys, method)
