import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. HELPER: LOAD TABLE
# ==========================================
def render_load_table(params):
    st.markdown("### 📋 Design Load Parameters")
    dl_f = params.get('dl_factor', 1.4)
    ll_f = params.get('ll_factor', 1.7)
    inc_sw = params.get('include_sw', True)
    
    data = [
        {"Load Case": "Dead Load (DL)", "Factor": f"{dl_f:.2f}", "Description": "Superimposed Dead Load"},
        {"Load Case": "Live Load (LL)", "Factor": f"{ll_f:.2f}", "Description": "Live Load (Occupancy)"}
    ]
    if inc_sw:
        data.insert(0, {"Load Case": "Self-Weight (SW)", "Factor": f"{dl_f:.2f}", "Description": "Beam Self-Weight"})
        
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
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
        
        total_concrete_vol += b_m * h_m * L
        total_formwork_area += (2 * h_m + b_m) * L
        
        w_span = 0.0
        def calc_w(n, db, length): return n * (db**2 / 162) * length if n > 0 else 0

        if 'top' in res and 'all_layers' in res['top']:
             for l in res['top']['all_layers']: w_span += calc_w(l['n'], l['db'], L * 1.1)
        if 'bot' in res and 'all_layers' in res['bot']:
             for l in res['bot']['all_layers']: w_span += calc_w(l['n'], l['db'], L * 1.1)

        stir_db = res.get('shear', {}).get('db', 6)
        stir_s = (res.get('shear', {}).get('s', 200)) / 1000.0
        if stir_s > 0:
            n_stir = int(L / stir_s) + 1
            len_stir = 2 * (b_m + h_m)
            w_span += n_stir * (stir_db**2 / 162) * len_stir
            
        total_steel_weight += w_span

    return pd.DataFrame([
        {"Item": "Concrete (240 ksc)", "Quantity": float(f"{total_concrete_vol:.2f}"), "Unit": "m³"},
        {"Item": "Formwork", "Quantity": float(f"{total_formwork_area:.2f}"), "Unit": "m²"},
        {"Item": "Rebar (DB+RB)", "Quantity": float(f"{total_steel_weight:.2f}"), "Unit": "kg"}
    ])

# ==========================================
# 3. PLOTLY ANALYSIS GRAPH (FINAL PERFECTED)
# ==========================================
def plot_analysis_results(res_df, spans, supports, loads, reactions):
    """
    Final Engineered FBD:
    - Arrow Tips: Offset to y=0.08 (Sits ON beam line, doesn't sink).
    - Labels: Full units (kN, kNm, mm) for professional reporting.
    - Proportions: Tuned for clarity.
    """
    
    # --- 1. DATA PREP ---
    if isinstance(loads, pd.DataFrame):
        load_list = loads.to_dict('records')
    elif isinstance(loads, list):
        load_list = loads
    else:
        load_list = []

    # Scaling Logic
    all_mags = [l['mag'] for l in load_list] if load_list else [1]
    max_load_val = max(all_mags) if all_mags else 1.0
    if max_load_val == 0: max_load_val = 1.0

    # --- 2. SETUP PLOT ---
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("<b>1. Free Body Diagram (FBD)</b>", "<b>2. Shear Force Diagram (SFD)</b>", "<b>3. Bending Moment Diagram (BMD)</b>", "<b>4. Deflection Diagram</b>"),
        row_heights=[0.30, 0.24, 0.24, 0.22]
    )

    total_L = sum(spans)
    cum_dist = [0] + list(np.cumsum(spans))
    
    # Beam Line (y=0) - Thick Black Line
    fig.add_trace(go.Scatter(x=[0, total_L], y=[0, 0], mode='lines', line=dict(color='black', width=5), hoverinfo='skip'), row=1, col=1)
    
    # Supports
    for idx, row in supports.iterrows():
        sym = "square" if row['type'] == 'Fixed' else ("circle" if row['type'] == 'Roller' else "triangle-up")
        fig.add_trace(go.Scatter(
            x=[row['x']], y=[-0.04], mode='markers+text',
            marker=dict(symbol=sym, size=10, color='white', line=dict(width=1.5, color='black')),
            text=[row['type'][0]], textposition="bottom center", hoverinfo='name', name="Support"
        ), row=1, col=1)

    # --- 3. DRAW LOADS ---
    
    # [CONFIG] Visual Constants
    UDL_MIN_H = 0.25
    UDL_MAX_H = 0.55
    P_MIN_H = 0.70
    P_MAX_H = 1.30
    ARROW_TIP_OFFSET = 0.08  # ยกหัวลูกศรขึ้นเล็กน้อยเพื่อให้วาง "บน" เส้นคานพอดี
    
    # LAYER 1: UDL
    for l in load_list:
        if l['type'] == 'U':
            span_idx = int(l['span_index'])
            start_x = cum_dist[span_idx] + float(l.get('d_start', 0))
            end_x = start_x + float(l['dist'])
            mag = l['mag']
            
            # Scaled Height
            ratio = mag / max_load_val
            h_visual = UDL_MIN_H + (ratio * (UDL_MAX_H - UDL_MIN_H))
            
            color = '#e74c3c' if l.get('case') == 'LL' else '#2980b9'
            
            # Fill Area
            fig.add_trace(go.Scatter(
                x=[start_x, end_x, end_x, start_x], y=[0, 0, h_visual, h_visual],
                fill='toself', fillcolor=color, opacity=0.12, line=dict(width=0), hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            # Top Line
            fig.add_trace(go.Scatter(
                x=[start_x, end_x], y=[h_visual, h_visual],
                mode='lines', line=dict(color=color, width=1, dash='dot'), hoverinfo='skip'
            ), row=1, col=1)
            # Label
            label_txt = f"w={mag/1000:.2f} kN/m" if l.get('case')!='SW' else f"SW={mag/1000:.2f} kN/m"
            fig.add_annotation(
                x=(start_x+end_x)/2, y=h_visual, text=label_txt, showarrow=False, yshift=8,
                font=dict(color=color, size=9), row=1, col=1
            )
            # Internal Arrows
            n_arrows = max(3, int(float(l['dist']) * 1.8))
            for ax_x in np.linspace(start_x, end_x, n_arrows + 2)[1:-1]:
                 fig.add_annotation(
                    x=ax_x, y=ARROW_TIP_OFFSET, # Tip sits ON beam
                    ax=ax_x, ay=h_visual,       # Tail at block height
                    axref='x', ayref='y', xref='x', yref='y',
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor=color,
                    row=1, col=1
                )

    # LAYER 2: POINT LOAD
    for l in load_list:
        if l['type'] == 'P':
            span_idx = int(l['span_index'])
            x_loc = cum_dist[span_idx] + float(l['d_start'])
            mag = l['mag']
            color = '#c0392b' if l.get('case') == 'LL' else '#2980b9'
            
            ratio = mag / max_load_val
            h_arrow = P_MIN_H + (ratio * (P_MAX_H - P_MIN_H))
            
            fig.add_annotation(
                x=x_loc, y=ARROW_TIP_OFFSET, # Tip sits ON beam
                ax=x_loc, ay=h_arrow,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2.0, arrowcolor=color,
                text=f"<b>P={mag/1000:.2f} kN</b>",
                xanchor='center', yanchor='bottom', yshift=5, 
                font=dict(color=color, size=11, family="Arial"),
                row=1, col=1
            )

    # --- 4. DIAGRAMS & LAYOUT (WITH UNITS) ---
    
    # SFD
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['shear']/1000, mode='lines', line=dict(color='#e74c3c', width=2), fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'), row=2, col=1)
    # SFD Labels with Units
    v_vals = res_df['shear']/1000
    for val in [v_vals.max(), v_vals.min()]:
        if abs(val) > 0.01:
            idx = (v_vals - val).abs().idxmin()
            fig.add_annotation(
                x=res_df['x'].iloc[idx], y=val, 
                text=f"<b>{val:.2f} kN</b>", # <--- Added Unit
                showarrow=False, yshift=10 if val>0 else -10, 
                font=dict(color='#e74c3c', size=11), row=2, col=1
            )

    # BMD
    fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['moment']/1000, mode='lines', line=dict(color='#27ae60', width=2), fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.1)'), row=3, col=1)
    # BMD Labels with Units
    m_vals = res_df['moment']/1000
    for val in [m_vals.max(), m_vals.min()]:
        if abs(val) > 0.01:
            idx = (m_vals - val).abs().idxmin()
            fig.add_annotation(
                x=res_df['x'].iloc[idx], y=val, 
                text=f"<b>{val:.2f} kNm</b>", # <--- Added Unit
                showarrow=True, arrowhead=1, ay=20 if val>0 else -20, 
                font=dict(color='#27ae60', size=11), row=3, col=1
            )

    # Deflection
    fig.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['deflection'], mode='lines', line=dict(color='#8e44ad', width=2)), row=4, col=1)
    # Deflection Labels with Units
    if not res_df['deflection'].empty:
        idx_max = res_df['deflection'].abs().idxmax()
        val_max = res_df['deflection'].iloc[idx_max]
        if abs(val_max) > 0.001:
             fig.add_annotation(
                 x=res_df['x'].iloc[idx_max], y=val_max, 
                 text=f"<b>Max: {val_max:.2f} mm</b>", # <--- Added Unit
                 showarrow=True, arrowhead=1, ay=30 if val_max < 0 else -30, 
                 font=dict(color='#8e44ad', size=11), row=4, col=1
             )

    # Grid & Layout
    for x_pos in cum_dist:
        fig.add_vline(x=x_pos, line_width=1, line_dash="dash", line_color="gray", opacity=0.3)

    fig.update_layout(height=1100, showlegend=False, template="plotly_white", hovermode="x unified", margin=dict(t=50, b=40, l=60, r=20))
    
    # Scale Y for FBD headroom
    fig.update_yaxes(range=[-0.4, 1.6], showgrid=False, visible=False, row=1, col=1)
    
    fig.update_yaxes(title_text="Shear (kN)", showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="Moment (kNm)", autorange="reversed", showgrid=True, row=3, col=1)
    fig.update_yaxes(title_text="Def. (mm)", showgrid=True, row=4, col=1)
    fig.update_xaxes(title_text="Length (m)", row=4, col=1)

    return fig
