import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st
from section_plotter import plot_longitudinal_section_detailed, plot_cross_section
from reporter import render_calculation_report

# ==========================================
# 1. BOQ CALCULATION
# ==========================================
def calculate_boq_summary(design_res, spans):
    total_concrete_vol = 0.0
    total_formwork_area = 0.0
    total_steel_weight = 0.0
    
    for i, res in enumerate(design_res):
        L = spans[i]
        b_mm = res.get('b') or 300
        h_mm = res.get('h') or 500

        b_m = b_mm / 1000.0
        h_m = h_mm / 1000.0
        
        # 1. Concrete (m3)
        vol = b_m * h_m * L
        total_concrete_vol += vol
        
        # 2. Formwork (m2)
        form_area = (2 * h_m + b_m) * L
        total_formwork_area += form_area
        
        # 3. Steel Weight (kg)
        w_main = 0.0
        def get_steel_weight(n, db, length):
            if n > 0:
                unit_w = (db**2 / 162)
                return n * unit_w * length
            return 0

        # Top
        if 'top' in res and isinstance(res['top'], dict) and 'all_layers' in res['top']:
             for layer in res['top']['all_layers']:
                 w_main += get_steel_weight(layer['n'], layer['db'], L * 1.1)
        else:
            n = res.get('top_n', 0)
            db = res.get('top_db', 12)
            w_main += get_steel_weight(n, db, L * 1.1)

        # Bottom
        if 'bot' in res and isinstance(res['bot'], dict) and 'all_layers' in res['bot']:
             for layer in res['bot']['all_layers']:
                 w_main += get_steel_weight(layer['n'], layer['db'], L * 1.1)
        else:
            n = res.get('bot_n', 0)
            db = res.get('bot_db', 12)
            w_main += get_steel_weight(n, db, L * 1.1)

        # Stirrups
        stir_db = res.get('shear', {}).get('db', res.get('stir_db', 6))
        stir_s_mm = res.get('shear', {}).get('s', res.get('stir_spacing', 200))
        stir_s = stir_s_mm / 1000.0
        
        if stir_s > 0:
            n_stir = int(L / stir_s) + 1
            len_stir = 2 * (b_m + h_m) 
            w_stir_unit = (stir_db**2 / 162)
            w_stir_total = n_stir * len_stir * w_stir_unit
        else:
            w_stir_total = 0
            
        span_steel = w_main + w_stir_total
        total_steel_weight += span_steel

    data = [
        {"Item": "Concrete Structure (240 ksc)", "Unit": "m3", "Quantity": float(f"{total_concrete_vol:.2f}")},
        {"Item": "Formwork (Sides & Bottom)", "Unit": "m2", "Quantity": float(f"{total_formwork_area:.2f}")},
        {"Item": "Reinforcement (DB + RB)", "Unit": "kg", "Quantity": float(f"{total_steel_weight:.2f}")}
    ]
    return pd.DataFrame(data)

# ==========================================
# 2. PLOTLY ANALYSIS GRAPH (PROFESSIONAL SCALING)
# ==========================================
def plot_analysis_results(res_df, spans, supports, loads, reactions):
    """
    Uses Pixel-Based scaling for arrows to guarantee visual consistency
    regardless of load magnitude.
    """
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>1. Free Body Diagram (FBD)</b>", 
            "<b>2. Shear Force Diagram (SFD)</b>", 
            "<b>3. Bending Moment Diagram (BMD)</b>",
            "<b>4. Deflection Diagram</b>"
        ),
        row_heights=[0.3, 0.23, 0.23, 0.24]
    )

    # --- ROW 1: FREE BODY DIAGRAM (FBD) ---
    total_L = sum(spans)
    cum_dist = [0] + list(np.cumsum(spans))
    
    # 1.1 Beam Geometry (Thick Line)
    fig.add_trace(go.Scatter(
        x=[0, total_L], y=[0, 0], 
        mode='lines', line=dict(color='black', width=6), hoverinfo='skip'
    ), row=1, col=1)
    
    # 1.2 Supports (Visual Markers)
    for idx, row in supports.iterrows():
        sym = "triangle-up"
        if row['type'] == 'Fixed': sym = "square"
        elif row['type'] == 'Roller': sym = "circle"
        
        fig.add_trace(go.Scatter(
            x=[row['x']], y=[-0.05], # Slightly below beam
            mode='markers+text',
            marker=dict(symbol=sym, size=14, color='white', line=dict(width=2, color='black')),
            text=[row['type'][0]], textposition="bottom center",
            hoverinfo='name', name=f"Support"
        ), row=1, col=1)

    # 1.3 Loads with Pixel-Based Scaling (The Fix)
    load_iter = loads if isinstance(loads, list) else []
    
    for l in load_iter:
        span_idx = int(l['span_index'])
        start_x_span = cum_dist[span_idx]
        mag_raw = l['mag']
        mag_kN = mag_raw / 1000.0
        
        # Color coding
        case_type = l.get('case', 'DL')
        color = '#c0392b' if case_type == 'LL' else '#2980b9'
        
        # --- POINT LOAD (P) ---
        if l['type'] == 'P':
            x_loc = start_x_span + float(l['d_start'])
            
            # Use ayref='pixel' -> Fixed pixel length regardless of axis scale
            fig.add_annotation(
                x=x_loc, y=0,
                ax=0, ay=-50,      # Arrow tail is 50px above y=0
                ayref='pixel',     # Absolute pixel scaling
                xref="x1", yref="y1",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=color,
                text=f"<b>{mag_kN:.2f}</b>", 
                yshift=55,         # Text shifted just above arrow tail
                font=dict(color=color, size=10),
                row=1, col=1
            )
            
        # --- UNIFORM LOAD (U) ---
        elif l['type'] == 'U':
            x_s = start_x_span + float(l.get('d_start', 0))
            dist_val = float(l['dist'])
            x_e = x_s + dist_val
            
            # Constant visual height for UDL (e.g., 0.6 units in fixed y-range)
            h_vis = 0.5 
            
            # Top Line
            fig.add_trace(go.Scatter(
                x=[x_s, x_e], y=[h_vis, h_vis],
                mode='lines', line=dict(color=color, width=1.5), hoverinfo='skip'
            ), row=1, col=1)
            
            # Fill Area
            fig.add_trace(go.Scatter(
                x=[x_s, x_e, x_e, x_s], y=[0, 0, h_vis, h_vis],
                fill='toself', fillcolor=color, opacity=0.1, line=dict(width=0),
                hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            
            # Distributed Arrows (Fixed Pixel Height)
            n_arrows = max(2, int(dist_val * 2.0))
            arrow_x = np.linspace(x_s, x_e, n_arrows + 2)[1:-1]
            
            for ax_x in arrow_x:
                fig.add_annotation(
                    x=ax_x, y=0, 
                    ax=0, ay=-30,  # Shorter arrows for UDL (30px)
                    ayref='pixel',
                    xref="x1", yref="y1",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor=color,
                    row=1, col=1
                )
            
            # Label
            label_txt = f"<b>w={mag_kN:.2f}</b>"
            if case_type == 'SW': label_txt = f"SW={mag_kN:.2f}"
            
            fig.add_annotation(
                x=(x_s+x_e)/2, y=h_vis,
                text=label_txt, showarrow=False, yshift=10,
                font=dict(color=color, size=10), row=1, col=1
            )

    # --- ROW 2: SHEAR (SFD) ---
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['shear']/1000, 
        mode='lines', name='Shear', line=dict(color='#e74c3c', width=2),
        fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'
    ), row=2, col=1)
    
    # Annotate Max/Min Shear
    v_vals = res_df['shear']/1000
    for val in [v_vals.max(), v_vals.min()]:
        if abs(val) > 0.01:
            idx = (v_vals - val).abs().idxmin()
            fig.add_annotation(
                x=res_df['x'].iloc[idx], y=val,
                text=f"<b>{val:.2f}</b>", showarrow=False, yshift=12 if val>0 else -12,
                font=dict(color='#e74c3c', size=11), row=2, col=1
            )

    # --- ROW 3: MOMENT (BMD) ---
    fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['moment']/1000, 
        mode='lines', name='Moment', line=dict(color='#27ae60', width=2),
        fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.1)'
    ), row=3, col=1)
    
    m_vals = res_df['moment']/1000
    for val in [m_vals.max(), m_vals.min()]:
        if abs(val) > 0.01:
            idx = (m_vals - val).abs().idxmin()
            fig.add_annotation(
                x=res_df['x'].iloc[idx], y=val,
                text=f"<b>{val:.2f}</b>", showarrow=True, arrowhead=1, ay=25 if val>0 else -25,
                font=dict(color='#27ae60', size=11), row=3, col=1
            )

    # --- ROW 4: DEFLECTION ---
    fig.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['deflection'], 
        mode='lines', name='Deflection', line=dict(color='#8e44ad', width=2)
    ), row=4, col=1)
    
    idx_max_def = res_df['deflection'].abs().idxmax()
    max_def = res_df['deflection'].iloc[idx_max_def]
    fig.add_annotation(
        x=res_df['x'].iloc[idx_max_def], y=max_def,
        text=f"<b>Max: {max_def:.2f} mm</b>",
        showarrow=True, arrowhead=1, ay=30 if max_def < 0 else -30,
        font=dict(color='#8e44ad', size=11), row=4, col=1
    )

    # Grid Lines
    for x_pos in cum_dist:
        fig.add_vline(x=x_pos, line_width=1, line_dash="dash", line_color="gray", opacity=0.3)

    # Layout Setup
    fig.update_layout(
        title="<b>Structural Analysis Results</b>",
        height=1000, showlegend=False, template="plotly_white", hovermode="x unified",
        margin=dict(t=50, b=40, l=60, r=20)
    )
    
    # *** CRITICAL FIX: Lock FBD Y-Axis ***
    # This prevents the beam from squashing. We set a fixed visual range (-0.5 to 1.5).
    # Since arrows use pixel-scaling, they will overlay this perfectly without resizing the axis.
    fig.update_yaxes(range=[-0.5, 1.5], showgrid=False, visible=False, row=1, col=1)
    
    fig.update_yaxes(title_text="Shear (kN)", showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="Moment (kNm)", autorange="reversed", showgrid=True, row=3, col=1)
    fig.update_yaxes(title_text="Def. (mm)", showgrid=True, row=4, col=1)
    fig.update_xaxes(title_text="Length (m)", row=4, col=1)

    return fig

# ==========================================
# 3. DESIGN CHECK DISPLAY
# ==========================================
def display_design_comparison(mu_pos, mu_neg, vu, design_res):
    st.markdown("---")
    st.subheader("🛠 RC Design Verification")
    
    fc = design_res.get('fc', 24)
    fy = design_res.get('fy', 400)
    b = design_res.get('b', 200)
    h = design_res.get('h', 400)
    d = h - 50
    as_min = max((0.25 * np.sqrt(fc) / fy) * b * d, (1.4 / fy) * b * d)

    st.markdown("#### 📏 Reinforcement Area ($A_s$)")
    as_col1, as_col2 = st.columns(2)
    
    with as_col1:
        as_req = max(design_res.get('as_req_bot', 0.0), as_min)
        as_prov = design_res.get('as_prov_bot', 0.0)
        st.write("**Bottom Steel (Mid-span)**")
        st.write(f"Req: `{as_req:.0f}` | Prov: `{as_prov:.0f}` $mm^2$")
        if as_req > 0:
            st.progress(min(as_prov / as_req, 1.0))
            if as_prov >= as_req: st.success("✅ Pass")
            else: st.error("❌ Fail")

    with as_col2:
        as_req_t = max(design_res.get('as_req_top', 0.0), as_min)
        as_prov_t = design_res.get('as_prov_top', 0.0)
        st.write("**Top Steel (Support)**")
        st.write(f"Req: `{as_req_t:.0f}` | Prov: `{as_prov_t:.0f}` $mm^2$")
        if as_req_t > 0:
            st.progress(min(as_prov_t / as_req_t, 1.0))
            if as_prov_t >= as_req_t: st.success("✅ Pass")
            else: st.error("❌ Fail")

    st.markdown("---")
    st.markdown("#### ⚡ Section Capacity")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        cap = design_res.get('phi_Mn_pos', 0.0)
        st.metric("Mu+ Capacity", f"{cap:.2f} kNm", delta=f"{cap - mu_pos:.2f}")
    with c2:
        cap = design_res.get('phi_Mn_neg', 0.0)
        st.metric("Mu- Capacity", f"{cap:.2f} kNm", delta=f"{cap - abs(mu_neg):.2f}")
    with c3:
        cap = design_res.get('phi_Vn', 0.0)
        st.metric("Shear Capacity", f"{cap:.2f} kN", delta=f"{cap - vu:.2f}")

    st.info(f"**Detail:** Top {design_res.get('top_n')}DB{design_res.get('top_db')} | Bot {design_res.get('bot_n')}DB{design_res.get('bot_db')} | Stirrup RB{design_res.get('stir_db',6)}@{design_res.get('stir_spacing',200)}")

# ==========================================
# 4. MAIN RENDER CONTROLLER
# ==========================================
def render_design_view(res_package):
    """
    Main entry point for displaying results.
    """
    if not res_package:
        st.error("No design results to display.")
        return

    # Unpack Data safely
    design_res = res_package.get('design_results', [])
    params = res_package.get('params', {})
    
    # ======================================================
    #  [FIX] 1. LOAD COMBINATION TABLE (FORCE DISPLAY FIRST)
    # ======================================================
    st.markdown("## 🏗️ Design Dashboard")
    st.markdown("### 📋 Load Combinations")
    
    # Get params with safe defaults
    dl_f = params.get('dl_factor', 1.4)
    ll_f = params.get('ll_factor', 1.7)
    inc_sw = params.get('include_sw', True)
    
    # Construct Table Data
    tbl_data = [
        {"Load Type": "Dead Load (DL)", "Factor": f"{dl_f:.2f}", "Note": "Superimposed DL"},
        {"Load Type": "Live Load (LL)", "Factor": f"{ll_f:.2f}", "Note": "Occupancy Load"}
    ]
    if inc_sw:
        tbl_data.insert(0, {"Load Type": "Self-Weight (SW)", "Factor": f"{dl_f:.2f}", "Note": "Beam Weight (2400 kg/m³)"})
    
    # Render Table
    st.dataframe(
        pd.DataFrame(tbl_data), 
        use_container_width=True, 
        hide_index=True,
        column_config={"Load Type": st.column_config.TextColumn("Type", width="medium")}
    )
    st.caption(f"*Design Load = {dl_f}DL + {ll_f}LL*")
    st.divider()

    # ======================================================
    #  2. PREPARE DATA FOR PLOTTING
    # ======================================================
    spans = res_package['spans']
    raw_loads = res_package.get('loads', [])
    
    # Deep copy loads to prevent duplication on re-runs
    display_loads = [dict(l) for l in (raw_loads or [])]

    # Add SW if needed (Calculated here for display)
    if inc_sw:
        for i, res in enumerate(design_res):
            b_m = (res.get('b') or params.get('b', 300)) / 1000.0
            h_m = (res.get('h') or params.get('h', 500)) / 1000.0
            sw_mag = 24000 * b_m * h_m 
            display_loads.append({
                'type': 'U', 'mag': sw_mag, 'span_index': i,
                'd_start': 0, 'dist': spans[i], 'case': 'SW'
            })

    # ======================================================
    #  3. TABS & VISUALIZATION
    # ======================================================
    t1, t2, t3 = st.tabs(["📊 Analysis Results", "📐 Section Details", "📝 Report"])
    
    with t1:
        st.subheader("Analysis Diagrams")
        df_plot = pd.DataFrame({
            'x': res_package['x'], 
            'moment': res_package['m'], 
            'shear': res_package['v'], 
            'deflection': res_package['d']
        })
        
        # Call the Professional Plotter
        fig = plot_analysis_results(
            df_plot, spans, res_package['supports'], display_loads, res_package['reactions']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Reactions Table
        st.subheader("Support Reactions")
        r_data = [{"Support": k, "Reaction (kN)": f"{val/1000:.2f}"} for k, val in res_package['reactions'].items()]
        st.dataframe(pd.DataFrame(r_data), use_container_width=True, hide_index=True)

    with t2:
        st.subheader("Detailed Section Design")
        if design_res:
            span_opts = [f"Span {i+1}" for i in range(len(spans))]
            sel_idx = st.selectbox("Select Span:", range(len(spans)), format_func=lambda x: span_opts[x])
            curr = design_res[sel_idx]
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(plot_cross_section(curr), use_container_width=True)
            with c2:
                display_design_comparison(curr['Mu_pos'], curr['Mu_neg'], curr['Vu_max'], curr)
            
            st.divider()
            st.subheader("Longitudinal Profile")
            svg_long, _ = plot_longitudinal_section_detailed(spans, res_package['supports'], design_res, params.get('h', 500), params.get('cover', 25))
            st.image(svg_long, use_container_width=True)

    with t3:
        st.header("📝 Summary")
        st.subheader("1. Bill of Quantities")
        boq_df = calculate_boq_summary(design_res, spans)
        st.dataframe(boq_df, use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader("2. Calculation Report")
        for i, res in enumerate(design_res):
            with st.expander(f"Span {i+1} Calculation", expanded=False):
                res['span_id'] = i
                res['L'] = spans[i]
                render_calculation_report(res)
