import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st
from section_plotter import plot_longitudinal_section_detailed, plot_cross_section
from reporter import render_calculation_report

# ==========================================
# 1. HELPER: LOAD TABLE DISPLAY (NEW)
# ==========================================
def render_load_table(params):
    """
    ฟังก์ชันแยกสำหรับแสดงตาราง Load Combination โดยเฉพาะ
    เรียกใช้ที่ไหนก็ได้ใน app.py
    """
    st.markdown("### 📋 Load Combinations")
    
    dl_f = params.get('dl_factor', 1.4)
    ll_f = params.get('ll_factor', 1.7)
    inc_sw = params.get('include_sw', True)
    
    tbl_data = [
        {"Load Type": "Dead Load (DL)", "Factor": f"{dl_f:.2f}", "Description": "Superimposed Dead Load"},
        {"Load Type": "Live Load (LL)", "Factor": f"{ll_f:.2f}", "Description": "Occupancy / Usage Load"}
    ]
    
    if inc_sw:
        tbl_data.insert(0, {
            "Load Type": "Self-Weight (SW)", 
            "Factor": f"{dl_f:.2f}", 
            "Description": "Beam Weight (approx 2400 kg/m³)"
        })
    else:
        tbl_data.insert(0, {
            "Load Type": "Self-Weight (SW)", 
            "Factor": "Excluded", 
            "Description": "Manually excluded by user"
        })
        
    df_table = pd.DataFrame(tbl_data)
    
    # Display Table
    st.dataframe(
        df_table, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Load Type": st.column_config.TextColumn("Type", width="medium"),
            "Factor": st.column_config.TextColumn("Safety Factor", width="small"),
            "Description": st.column_config.TextColumn("Detail", width="large"),
        }
    )
    st.caption(f"ℹ️ **Equation:** Ultimate Load = {dl_f}DL + {ll_f}LL")
    st.divider()

# ==========================================
# 2. BOQ CALCULATION
# ==========================================
def calculate_boq_summary(design_res, spans):
    total_concrete_vol = 0.0
    total_formwork_area = 0.0
    total_steel_weight = 0.0
    
    for i, res in enumerate(design_res):
        L = spans[i]
        b_m = (res.get('b') or 300) / 1000.0
        h_m = (res.get('h') or 500) / 1000.0
        
        # Concrete & Formwork
        total_concrete_vol += b_m * h_m * L
        total_formwork_area += (2 * h_m + b_m) * L
        
        # Steel Weight Estimation
        w_span = 0.0
        # Helper to calc weight
        def calc_w(n, db, length): return n * (db**2 / 162) * length if n > 0 else 0

        # Main Bars
        if 'top' in res and 'all_layers' in res['top']:
             for l in res['top']['all_layers']: w_span += calc_w(l['n'], l['db'], L * 1.1)
        
        if 'bot' in res and 'all_layers' in res['bot']:
             for l in res['bot']['all_layers']: w_span += calc_w(l['n'], l['db'], L * 1.1)

        # Stirrups
        stir_db = res.get('shear', {}).get('db', 6)
        stir_s = (res.get('shear', {}).get('s', 200)) / 1000.0
        if stir_s > 0:
            n_stir = int(L / stir_s) + 1
            len_stir = 2 * (b_m + h_m)
            w_span += n_stir * (stir_db**2 / 162) * len_stir
            
        total_steel_weight += w_span

    data = [
        {"Item": "Concrete (240 ksc)", "Quantity": float(f"{total_concrete_vol:.2f}"), "Unit": "m³"},
        {"Item": "Formwork", "Quantity": float(f"{total_formwork_area:.2f}"), "Unit": "m²"},
        {"Item": "Rebar (DB+RB)", "Quantity": float(f"{total_steel_weight:.2f}"), "Unit": "kg"}
    ]
    return pd.DataFrame(data)

# ==========================================
# 3. PLOTLY ANALYSIS GRAPH (PIXEL SCALING FIX)
# ==========================================
def plot_analysis_results(res_df, spans, supports, loads, reactions):
    """
    Uses Pixel-Based scaling. Arrows stay consistent size regardless of load magnitude.
    """
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=("<b>1. Free Body Diagram</b>", "<b>2. Shear Force</b>", "<b>3. Bending Moment</b>", "<b>4. Deflection</b>"),
        row_heights=[0.3, 0.23, 0.23, 0.24]
    )

    # --- ROW 1: FBD ---
    total_L = sum(spans)
    cum_dist = [0] + list(np.cumsum(spans))
    
    # Beam
    fig.add_trace(go.Scatter(x=[0, total_L], y=[0, 0], mode='lines', line=dict(color='black', width=6), hoverinfo='skip'), row=1, col=1)
    
    # Supports
    for idx, row in supports.iterrows():
        sym = "square" if row['type'] == 'Fixed' else ("circle" if row['type'] == 'Roller' else "triangle-up")
        fig.add_trace(go.Scatter(
            x=[row['x']], y=[-0.05], mode='markers+text',
            marker=dict(symbol=sym, size=14, color='white', line=dict(width=2, color='black')),
            text=[row['type'][0]], textposition="bottom center", hoverinfo='name', name="Support"
        ), row=1, col=1)

    # Loads (Pixel Scaling)
    load_iter = loads if isinstance(loads, list) else []
    for l in load_iter:
        span_idx = int(l['span_index'])
        start_x = cum_dist[span_idx]
        mag_kN = l['mag'] / 1000.0
        color = '#c0392b' if l.get('case') == 'LL' else '#2980b9'
        
        if l['type'] == 'P':
            x_loc = start_x + float(l['d_start'])
            fig.add_annotation(
                x=x_loc, y=0, ax=0, ay=-50, ayref='pixel', # Fixed 50px height
                xref="x1", yref="y1", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=color,
                text=f"<b>{mag_kN:.2f}</b>", yshift=55, font=dict(color=color, size=10), row=1, col=1
            )
        elif l['type'] == 'U':
            x_s = start_x + float(l.get('d_start', 0))
            x_e = x_s + float(l['dist'])
            h_vis = 0.5 # Fixed visual height
            
            # Draw UDL Block
            fig.add_trace(go.Scatter(x=[x_s, x_e, x_e, x_s], y=[0, 0, h_vis, h_vis], fill='toself', fillcolor=color, opacity=0.1, line=dict(width=0), hoverinfo='skip', showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=[x_s, x_e], y=[h_vis, h_vis], mode='lines', line=dict(color=color, width=1.5), hoverinfo='skip'), row=1, col=1)
            
            # Arrows
            n_arrows = max(2, int((x_e - x_s) * 2.0))
            for ax_x in np.linspace(x_s, x_e, n_arrows + 2)[1:-1]:
                fig.add_annotation(x=ax_x, y=0, ax=0, ay=-30, ayref='pixel', xref="x1", yref="y1", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor=color, row=1, col=1)
            
            label_txt = f"<b>w={mag_kN:.2f}</b>"
            if l.get('case') == 'SW': label_txt = f"SW={mag_kN:.2f}"
            fig.add_annotation(x=(x_s+x_e)/2, y=h_vis, text=label_txt, showarrow=False, yshift=10, font=dict(color=color, size=10), row=1, col=1)

    # --- ROW 2-4: GRAPHS ---
    # Shear
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['shear']/1000, mode='lines', line=dict(color='#e74c3c', width=2), fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'), row=2, col=1)
    
    # Moment
    fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['moment']/1000, mode='lines', line=dict(color='#27ae60', width=2), fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.1)'), row=3, col=1)

    # Deflection
    fig.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['deflection'], mode='lines', line=dict(color='#8e44ad', width=2)), row=4, col=1)

    # Layout
    for x_pos in cum_dist: fig.add_vline(x=x_pos, line_width=1, line_dash="dash", line_color="gray", opacity=0.3)
    fig.update_layout(height=1000, showlegend=False, template="plotly_white", hovermode="x unified", margin=dict(t=50, b=40, l=60, r=20))
    
    # CRITICAL FIX: Lock FBD Y-Axis so arrows fit perfect
    fig.update_yaxes(range=[-0.5, 1.5], showgrid=False, visible=False, row=1, col=1)
    fig.update_yaxes(title_text="Shear (kN)", showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="Moment (kNm)", autorange="reversed", showgrid=True, row=3, col=1)
    fig.update_yaxes(title_text="Def. (mm)", showgrid=True, row=4, col=1)

    return fig

# ==========================================
# 4. DESIGN CHECK DISPLAY
# ==========================================
def display_design_comparison(mu_pos, mu_neg, vu, design_res):
    st.markdown("---")
    st.subheader("🛠 RC Design Verification")
    
    fc = design_res.get('fc', 24)
    fy = design_res.get('fy', 400)
    as_prov_b = design_res.get('as_prov_bot', 0.0)
    as_req_b = max(design_res.get('as_req_bot', 0.0), 0.0)
    
    as_prov_t = design_res.get('as_prov_top', 0.0)
    as_req_t = max(design_res.get('as_req_top', 0.0), 0.0)

    # Bars
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Bottom Steel (Mid)**")
        st.write(f"Req: `{as_req_b:.0f}` | Prov: `{as_prov_b:.0f}` mm²")
        if as_req_b > 0: st.progress(min(as_prov_b/as_req_b, 1.0))
        
    with c2:
        st.write("**Top Steel (Sup)**")
        st.write(f"Req: `{as_req_t:.0f}` | Prov: `{as_prov_t:.0f}` mm²")
        if as_req_t > 0: st.progress(min(as_prov_t/as_req_t, 1.0))

    # Capacity
    st.markdown("#### Capacity Status")
    k1, k2, k3 = st.columns(3)
    k1.metric("Mu+ Cap", f"{design_res.get('phi_Mn_pos',0):.1f} kNm", delta=f"{design_res.get('phi_Mn_pos',0)-mu_pos:.1f}")
    k2.metric("Mu- Cap", f"{design_res.get('phi_Mn_neg',0):.1f} kNm", delta=f"{design_res.get('phi_Mn_neg',0)-abs(mu_neg):.1f}")
    k3.metric("Shear Cap", f"{design_res.get('phi_Vn',0):.1f} kN", delta=f"{design_res.get('phi_Vn',0)-vu:.1f}")
