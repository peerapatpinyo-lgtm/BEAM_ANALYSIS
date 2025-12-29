import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- IMPORT MODULES ---
# ตรวจสอบความพร้อมของ Module คำนวณ
try:
    from beam_analysis import run_beam_analysis 
    import design_view 
except ImportError as e:
    st.error(f"⚠️ Critical System Error: Missing calculation modules. Please ensure 'beam_analysis.py' and 'design_view.py' exist. ({e})")
    st.stop()

# ==========================================
# 1. SETUP & STYLES (Professional Engineer Theme)
# ==========================================
st.set_page_config(page_title="Professional RC Beam Design", layout="wide", page_icon="🏗️")

if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False

st.markdown("""
<style>
    /* Main Font: Sarabun for Thai/English consistency */
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Sarabun', sans-serif; 
        font-size: 15px; 
    }
    
    /* Headers */
    .header-box { 
        background: linear-gradient(90deg, #1565C0 0%, #0D47A1 100%); 
        color: white; 
        padding: 20px; 
        border-radius: 8px; 
        text-align: center; 
        margin-bottom: 25px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
    
    .section-header { 
        border-left: 6px solid #1565C0; 
        padding-left: 15px; 
        font-size: 1.4rem; 
        font-weight: 700; 
        margin-top: 30px; 
        margin-bottom: 20px; 
        background-color: #E3F2FD; 
        padding: 10px;
        border-radius: 0 5px 5px 0;
        color: #0D47A1; 
    }
    
    /* Calculation Display Box */
    .calc-display { 
        font-family: 'Courier New', monospace; 
        background-color: #F1F8E9; 
        border-left: 5px solid #2E7D32; 
        padding: 10px; 
        margin-top: 10px; 
        font-size: 0.95rem; 
        color: #1B5E20; 
        border-radius: 4px;
        font-weight: 600;
    }
    
    /* Error/Warning Boxes */
    .error-box { 
        background-color: #FFEBEE; 
        border: 2px solid #C62828; 
        color: #B71C1C; 
        padding: 15px; 
        border-radius: 6px; 
        font-weight: bold; 
        font-size: 1.1rem;
        margin-bottom: 10px;
    }
    
    /* Input Fields Styling */
    .stNumberInput input { 
        font-weight: bold; 
        color: #0D47A1; 
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #fff;
        border-top: 3px solid #1565C0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ENGINEERING LOGIC (STABILITY CHECK)
# ==========================================

def check_structural_stability(supports_df):
    """
    Checks if the beam is statically stable based on support types.
    """
    if supports_df.empty:
        return False, "❌ CRITICAL: No supports defined."

    active_sups = supports_df[supports_df['type'] != 'None']
    types = active_sups['type'].tolist()
    num_sup = len(types)
    
    # 1. Unstable mechanism (Free body)
    if num_sup == 0:
        return False, "❌ UNSTABLE: Structure is a Mechanism (No Supports)."
    
    # 2. Single Support Stability
    if num_sup == 1:
        s_type = types[0]
        if s_type == "Fixed":
            return True, "✅ Stable (Cantilever)"
        else:
            return False, f"❌ UNSTABLE: A single '{s_type}' cannot support a beam (Rotation/Translation Unrestrained)."

    # 3. Horizontal Instability (All Rollers)
    if all(t == "Roller" for t in types):
        return False, "❌ UNSTABLE: All supports are Rollers. (Horizontal Instability)."

    return True, "✅ Stable"

# ==========================================
# 3. INPUT SECTIONS (Refined Grid Layout)
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
            st.info("WSD: Service Loads (Factor = 1.0)")
        return design_code, method, f_dl, f_ll, unit_sys

def render_geometry_input():
    st.markdown('<div class="section-header">1️⃣ Geometry & Supports</div>', unsafe_allow_html=True)
    
    col_n, _ = st.columns([1, 2])
    n_span = col_n.number_input("Number of Spans", min_value=1, max_value=10, value=2)
    
    # --- SPANS INPUT (Responsive Grid) ---
    st.markdown("**Span Lengths (m)**")
    spans = []
    
    # Create grid for span inputs (Max 4 per row)
    cols_per_row = 4
    for i in range(0, n_span, cols_per_row):
        end_idx = min(i + cols_per_row, n_span)
        cols = st.columns(cols_per_row)
        for j in range(i, end_idx):
            col_idx = j - i
            with cols[col_idx]:
                l = st.number_input(f"L{j+1}", min_value=1.0, value=5.0, step=0.5, key=f"span_{j}")
                spans.append(l)
    
    # --- SUPPORTS INPUT (Responsive Grid) ---
    st.markdown("#### Supports Configuration")
    support_options = ["Pin", "Roller", "Fixed", "None"]
    current_supports = []
    
    total_sups = n_span + 1
    # Create grid for support inputs (Max 5 per row to avoid squeezing)
    sup_cols_limit = 5
    
    for i in range(0, total_sups, sup_cols_limit):
        end_idx = min(i + sup_cols_limit, total_sups)
        cols = st.columns(sup_cols_limit)
        for j in range(i, end_idx):
            col_idx = j - i
            with cols[col_idx]:
                # Intelligent Default Logic
                # First = Pin, Others = Roller, Last = Roller
                default_idx = 0 if j == 0 else (1 if j <= n_span else 3)
                if j > n_span: default_idx = 1
                
                s_type = st.selectbox(f"Sup {j+1}", support_options, index=default_idx, key=f"sup_{j}")
                current_supports.append(s_type)
    
    x_coords = [0] + list(np.cumsum(spans))
    supports_df = pd.DataFrame({'x': x_coords, 'type': current_supports})
    
    # Real-time Stability Check
    is_stable, msg = check_structural_stability(supports_df)
    if not is_stable:
        st.markdown(f'<div class="error-box">{msg}</div>', unsafe_allow_html=True)
    
    return n_span, spans, supports_df, is_stable

def render_custom_load_input(n_span, spans, unit_sys, f_dl, f_ll):
    st.markdown('<div class="section-header">2️⃣ Applied Loads (Total Factored)</div>', unsafe_allow_html=True)
    st.info(f"💡 **Engineer Note:** Loads will be automatically combined using $U = {f_dl}DL + {f_ll}LL$")
    
    loads = []
    tabs = st.tabs([f"📍 Span {i+1} (L={spans[i]}m)" for i in range(n_span)])
    
    for i, tab in enumerate(tabs):
        with tab:
            col_main_1, col_main_2 = st.columns([1, 1])
            
            # --- Uniform Load (UDL) ---
            with col_main_1:
                with st.container():
                    st.markdown("#### 🔹 Uniform Distributed Load")
                    w_dl = st.number_input(f"Dead Load (DL)", value=0.0, step=100.0, format="%.2f", key=f"w_dl_{i}")
                    w_ll = st.number_input(f"Live Load (LL)", value=0.0, step=100.0, format="%.2f", key=f"w_ll_{i}")
                    
                    w_u = (w_dl * f_dl) + (w_ll * f_ll)
                    
                    if w_u != 0:
                        loads.append({'span_idx': i, 'type': 'U', 'w': w_u, 'source': 'Total UDL'})
                        st.markdown(f"""
                        <div class="calc-display">
                        Wu Calculation:<br>
                        = {f_dl:.2f}({w_dl}) + {f_ll:.2f}({w_ll})<br>
                        = <b>{w_u:,.2f}</b>
                        </div>
                        """, unsafe_allow_html=True)

            # --- Point Loads ---
            with col_main_2:
                with st.container():
                    st.markdown("#### 🔻 Point Loads")
                    num_p = st.number_input(f"Add Point Loads (Qty)", min_value=0, max_value=5, value=0, key=f"qty_p_{i}")

                    if num_p > 0:
                        for j in range(num_p):
                            st.markdown(f"**Point Load #{j+1}**")
                            c1, c2, c3 = st.columns([0.8, 0.8, 1.2])
                            
                            p_dl = c1.number_input(f"P(DL)", value=0.0, step=100.0, key=f"p_dl_{i}_{j}")
                            p_ll = c2.number_input(f"P(LL)", value=0.0, step=100.0, key=f"p_ll_{i}_{j}")
                            
                            x_max = float(spans[i])
                            x_loc = c3.number_input(f"Distance x (0-{x_max})", min_value=0.0, max_value=x_max, value=x_max/2, step=0.1, key=f"p_x_{i}_{j}")
                            
                            p_total = (p_dl * f_dl) + (p_ll * f_ll)
                            
                            if p_total != 0:
                                loads.append({
                                    'span_idx': i, 
                                    'type': 'P', 
                                    'P': p_total, 
                                    'x': x_loc, 
                                    'source': 'Total P'
                                })
                                st.markdown(f"""
                                <div class="calc-display" style="padding:5px; font-size:0.85rem;">
                                Pu = {f_dl}({p_dl}) + {f_ll}({p_ll}) = <b>{p_total:,.2f}</b>
                                </div>
                                <hr style="margin:5px 0;">
                                """, unsafe_allow_html=True)

    return loads

# ==========================================
# 4. VISUALIZATION ENGINE (Polished & Aggregated)
# ==========================================

def draw_support(fig, x, y, sup_type, size=1.0):
    """ Draws professional support symbols """
    s = size * 0.8
    line_col, fill_col = "#263238", "#B0BEC5"
    
    if sup_type == "Pin":
        fig.add_shape(type="path", path=f"M {x},{y} L {x-s/2},{y-s} L {x+s/2},{y-s} Z", fillcolor=fill_col, line_color=line_col, line_width=2, row=1, col=1)
        fig.add_shape(type="line", x0=x-s, y0=y-s, x1=x+s, y1=y-s, line=dict(color=line_col, width=3), row=1, col=1)
        # Hatching
        for hx in np.linspace(x-s, x+s, 5):
            fig.add_shape(type="line", x0=hx, y0=y-s, x1=hx-s/3, y1=y-s*1.3, line=dict(color=line_col, width=1), row=1, col=1)

    elif sup_type == "Roller":
        fig.add_shape(type="circle", x0=x-s/2, y0=y-s, x1=x+s/2, y1=y, fillcolor=fill_col, line_color=line_col, line_width=2, row=1, col=1)
        fig.add_shape(type="line", x0=x-s, y0=y-s, x1=x+s, y1=y-s, line=dict(color=line_col, width=3), row=1, col=1)

    elif sup_type == "Fixed":
        h = s * 1.5
        fig.add_shape(type="line", x0=x, y0=y-h/2, x1=x, y1=y+h/2, line=dict(color=line_col, width=5), row=1, col=1)
        direction = -1 if x < 0.1 else 1
        for hy in np.linspace(y-h/2, y+h/2, 8):
            fig.add_shape(type="line", x0=x, y0=hy, x1=x+direction*s/2, y1=hy-s/2, line=dict(color=line_col, width=1), row=1, col=1)

def add_value_label(fig, x, y, val, unit, color, row, position="top"):
    """ Clean, Smaller Label for peaks to avoid overlapping """
    yshift = 20 if position == "top" else -20
    
    # Removed LaTeX code ($) to prevent display issues
    text = f"<b>{val:,.2f}</b>"
    
    fig.add_annotation(
        x=x, y=y,
        text=text,
        showarrow=False,
        font=dict(color=color, size=11, family="Arial"), # Reduced Font Size
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor=color,
        borderwidth=1,
        borderpad=3,
        yshift=yshift,
        row=row, col=1
    )

def create_engineering_plots(df, vis_spans, vis_supports, loads, unit_sys, factors):
    if not vis_spans or df.empty: return go.Figure()

    cum_len = [0] + list(np.cumsum(vis_spans))
    total_len = cum_len[-1]
    
    is_metric = "Metric" in unit_sys
    force_unit = "kg" if is_metric else "kN"
    moment_unit = "kg-m" if is_metric else "kN-m"
    dist_unit = "m"
    f_dl, f_ll = factors
    
    # 1. Determine Max Scale (for visuals)
    all_magnitudes = []
    # Temporary aggregation for scaling
    temp_p_map = {}
    for l in loads:
        if l['type'] == 'U': all_magnitudes.append(abs(l.get('w', 0)))
        elif l['type'] == 'P':
             abs_x = cum_len[l['span_idx']] + l['x']
             k = round(abs_x, 2)
             temp_p_map[k] = temp_p_map.get(k, 0) + l.get('P', 0)
    for p_val in temp_p_map.values(): all_magnitudes.append(abs(p_val))
    
    global_max = max(all_magnitudes) if all_magnitudes else 100.0
    if global_max == 0: global_max = 100
    
    # Scale Height relative to Beam Length (Proportional)
    target_h = max(1.0, total_len * 0.15) 
    
    # 2. Setup Subplots
    # Adjusted Row Heights as requested: Loading bigger, others smaller
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>1. FACTORED LOAD DIAGRAM</b>", 
            "<b>2. SHEAR FORCE DIAGRAM (SFD)</b>", 
            "<b>3. BENDING MOMENT DIAGRAM (BMD)</b>"
        ),
        row_heights=[0.35, 0.325, 0.325] 
    )
    
    # --- ROW 1: LOADING ---
    fig.add_shape(type="line", x0=0, y0=0, x1=total_len, y1=0, line=dict(color="black", width=4), row=1, col=1)
    
    # Supports
    sup_size = target_h * 0.3
    for i, x in enumerate(cum_len):
        if i < len(vis_supports):
            stype = vis_supports.iloc[i]['type']
            if stype != "None": draw_support(fig, x, 0, stype, size=sup_size)
            
    # Process Loads (UDL Draw + Point Load Aggregation)
    point_load_aggregator = {} # Key: absolute_x, Value: sum_P
    
    for load in loads:
        span_idx = load.get('span_idx', 0)
        x_start = cum_len[span_idx]
        x_end = cum_len[span_idx+1]
        
        # UDL: Draw immediately
        if load['type'] == 'U' and load['w'] != 0:
            w = load['w']
            h = (0.2 + 0.6 * (abs(w)/global_max)) * target_h
            
            fig.add_trace(go.Scatter(x=[x_start, x_end, x_end, x_start], y=[0, 0, h, h], fill='toself', fillcolor='rgba(239, 108, 0, 0.15)', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=[x_start, x_end], y=[h, h], mode='lines', line=dict(color='#E65100', width=2), showlegend=False, hoverinfo='text'), row=1, col=1)
            
            # Clean Label (Wu)
            fig.add_annotation(x=(x_start+x_end)/2, y=h, text=f"<b>Wu = {w:,.0f}</b>", showarrow=False, yshift=12, font=dict(color="#E65100", size=11, weight="bold"), row=1, col=1)

        # Point Load: Aggregate first
        elif load['type'] == 'P' and load['P'] != 0:
            local_x = load['x']
            abs_x = x_start + local_x
            key_x = round(abs_x, 3) 
            point_load_aggregator[key_x] = point_load_aggregator.get(key_x, 0) + load['P']

    # Draw Aggregated Point Loads (Fixing the multiple arrows issue)
    for px, p_val in point_load_aggregator.items():
        if p_val != 0:
            h = (0.2 + 0.6 * (abs(p_val)/global_max)) * target_h
            p_color = "#212121"
            
            # Single Arrow for Total Load
            fig.add_annotation(
                x=px, y=0,
                ax=px, ay=h,
                xref="x1", yref="y1", axref="x1", ayref="y1",
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor=p_color,
                text=f"<b>Pu={p_val:,.0f}</b>", # Clean Label (Pu)
                font=dict(size=11, color=p_color, weight="bold"),
                yshift=12,
                row=1, col=1
            )
            
    fig.update_yaxes(visible=False, range=[-0.5, target_h*1.6], row=1, col=1)

    # --- ROW 2 & 3: SFD / BMD ---
    plot_x, plot_v, plot_m = [], [], []
    current_offset = 0.0
    
    # Reconstruct Global Coordinates for Plotting
    for i in range(len(vis_spans)):
        span_data = df[df['span_id'] == i].copy()
        if not span_data.empty:
            # Shift x to global
            if i > 0 and span_data['x'].min() < 0.1: 
                 gx = span_data['x'] + current_offset
            else: 
                 gx = span_data['x']
            
            plot_x.extend(gx.tolist())
            plot_v.extend(span_data['shear'].tolist())
            plot_m.extend((-span_data['moment']).tolist()) 
        current_offset += vis_spans[i]

    np_x, np_v, np_m = np.array(plot_x), np.array(plot_v), np.array(plot_m)

    # SFD Plot
    col_shear = '#D32F2F'
    fig.add_trace(go.Scatter(x=np_x, y=np_v, mode='lines', line=dict(color=col_shear, width=2), fill='tozeroy', fillcolor='rgba(211, 47, 47, 0.1)', name="Shear"), row=2, col=1)
    
    if len(np_v) > 0:
        v_max, v_min = np_v.max(), np_v.min()
        idx_max, idx_min = np.argmax(np_v), np.argmin(np_v)
        # Add labels only for significant values
        if abs(v_max) > 1: add_value_label(fig, np_x[idx_max], v_max, v_max, force_unit, col_shear, 2, "top" if v_max > 0 else "bottom")
        if abs(v_min) > 1: add_value_label(fig, np_x[idx_min], v_min, v_min, force_unit, col_shear, 2, "bottom" if v_min < 0 else "top")
        
        # Add Padding (Reduce visual size)
        range_v = v_max - v_min
        pad_v = range_v * 0.25 if range_v > 0 else 1.0
        fig.update_yaxes(range=[v_min - pad_v, v_max + pad_v], row=2, col=1)

    # BMD Plot
    col_moment = '#1565C0'
    fig.add_trace(go.Scatter(x=np_x, y=np_m, mode='lines', line=dict(color=col_moment, width=2), fill='tozeroy', fillcolor='rgba(21, 101, 192, 0.1)', name="Moment"), row=3, col=1)
    
    if len(np_m) > 0:
        m_max, m_min = np_m.max(), np_m.min()
        idx_max, idx_min = np.argmax(np_m), np.argmin(np_m)
        if abs(m_max) > 1: add_value_label(fig, np_x[idx_max], m_max, m_max, moment_unit, col_moment, 3, "top" if m_max > 0 else "bottom")
        if abs(m_min) > 1: add_value_label(fig, np_x[idx_min], m_min, m_min, moment_unit, col_moment, 3, "bottom" if m_min < 0 else "top")
        
        # Add Padding
        range_m = m_max - m_min
        pad_m = range_m * 0.25 if range_m > 0 else 1.0
        fig.update_yaxes(range=[m_min - pad_m, m_max + pad_m], row=3, col=1)

    # --- LAYOUT POLISH ---
    font_style = dict(family="Arial, sans-serif", size=12, color="black")
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#ECEFF1', title_font=font_style, tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#ECEFF1', title_font=font_style, tickfont=dict(size=11))
    
    fig.update_yaxes(title_text=f"Shear ({force_unit})", row=2, col=1)
    fig.update_yaxes(title_text=f"Moment ({moment_unit})", row=3, col=1)
    fig.update_xaxes(title_text=f"Distance ({dist_unit})", row=3, col=1)
    
    # Dashed Grid Lines for Supports
    for r in [2, 3]:
        for sx in cum_len:
            fig.add_vline(x=sx, line_width=1, line_dash="dash", line_color="gray", opacity=0.4, row=r, col=1)

    fig.update_annotations(font_size=14) 
    fig.update_layout(height=1000, showlegend=False, plot_bgcolor="white", hovermode="x unified", margin=dict(t=60, l=60, r=30, b=40))
    
    return fig

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
st.markdown('<div class="header-box"><h2>🏗️ RC Beam Pro: Engineer Edition</h2></div>', unsafe_allow_html=True)

# 1. Inputs
design_code, method, f_dl, f_ll, unit_sys = render_sidebar()
c_geo, c_load = st.columns([1, 1.5])

with c_geo:
    n_span, spans, supports, is_stable = render_geometry_input()

with c_load:
    loads_input = render_custom_load_input(n_span, spans, unit_sys, f_dl, f_ll)

# 2. Calculation Button
st.markdown("---")
if st.button("🚀 Run Analysis & Design", type="primary", disabled=not is_stable):
    if not is_stable:
        st.error("Cannot calculate: Structure is unstable.")
    else:
        try:
            # Clean Input
            clean_loads = [l for l in loads_input if isinstance(l, dict)]
            st.session_state['loads_input'] = clean_loads
            
            # Run Engine
            vis_spans_df, vis_supports_df = run_beam_analysis(spans, supports, clean_loads)
            
            st.session_state['res_df'] = vis_spans_df
            st.session_state['vis_data'] = (spans, vis_supports_df) 
            st.session_state['analyzed'] = True
            
        except Exception as e:
            st.error(f"Calculation Error: {e}")
            st.warning("Please check if your 'beam_analysis.py' returns the expected DataFrame format.")

# 3. Results Visualization
if st.session_state['analyzed'] and st.session_state.get('res_df') is not None:
    df = st.session_state['res_df']
    vis_spans, vis_supports_df = st.session_state['vis_data']
    loads = st.session_state['loads_input']
    
    st.markdown('<div class="section-header">3️⃣ Analysis Results (SFD/BMD)</div>', unsafe_allow_html=True)
    
    fig = create_engineering_plots(df, vis_spans, vis_supports_df, loads, unit_sys, (f_dl, f_ll))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-header">4️⃣ RC Design (Reinforcement)</div>', unsafe_allow_html=True)
    design_view.render_design_section(df, vis_spans, unit_sys, method)
