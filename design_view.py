import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# ==========================================
# 1. CAPACITY CHECK VISUALIZATION
# ==========================================
def plot_capacity_vs_demand(df_span, phi_Mn_pos, phi_Mn_neg):
    """
    สร้างกราฟเปรียบเทียบ Mu vs PhiMn (Capacity Check)
    Units: kNm
    """
    # Convert Demand to kNm (Assuming solver output is N-m)
    mu_kNm = df_span['moment'] / 1000.0
    
    fig = go.Figure()
    
    # 1. Plot Mu (Demand) - Solid Blue
    fig.add_trace(go.Scatter(
        x=df_span['x'], 
        y=mu_kNm, 
        mode='lines',
        name='Applied Moment (Mu)',
        line=dict(color='#2980B9', width=3),
        fill='tozeroy',
        fillcolor='rgba(41, 128, 185, 0.1)'
    ))
    
    # 2. Plot +PhiMn (Positive Capacity) - Dashed Green
    x_min, x_max = df_span['x'].min(), df_span['x'].max()
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], 
        y=[phi_Mn_pos, phi_Mn_pos],
        mode='lines',
        name=f'Capacity +M (φMn = {phi_Mn_pos:.1f})',
        line=dict(color='#27AE60', width=2, dash='dash')
    ))
    
    # 3. Plot -PhiMn (Negative Capacity) - Dashed Red
    # Capacity is scalar, but we plot it on negative side for comparison
    neg_cap = -abs(phi_Mn_neg)
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], 
        y=[neg_cap, neg_cap],
        mode='lines',
        name=f'Capacity -M (φMn = {abs(neg_cap):.1f})',
        line=dict(color='#C0392B', width=2, dash='dash')
    ))

    # Add Max Demand Label
    max_mu = mu_kNm.max()
    min_mu = mu_kNm.min()
    
    # Annotate Max Positive
    if max_mu > 1.0:
        fig.add_annotation(x=df_span.loc[mu_kNm.idxmax(), 'x'], y=max_mu,
                           text=f"Max +: {max_mu:.2f}", showarrow=True, arrowhead=1)
    # Annotate Max Negative
    if min_mu < -1.0:
        fig.add_annotation(x=df_span.loc[mu_kNm.idxmin(), 'x'], y=min_mu,
                           text=f"Max -: {min_mu:.2f}", showarrow=True, arrowhead=1, ay=30)

    # Layout
    fig.update_layout(
        title="Moment Capacity Check (kNm)",
        xaxis_title="Position (m)",
        yaxis_title="Moment (kNm)",
        height=350,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    return fig

# ==========================================
# 2. ANALYSIS DIAGRAMS (UPDATED)
# ==========================================
def draw_interactive_diagrams(df, reac, spans, sup_df, loads, dl_factor=1.4, ll_factor=1.7):
    """
    วาดกราฟวิเคราะห์โครงสร้าง (SFD, BMD, Deflection)
    Units: kN, kNm, mm
    """
    # --- 1. Data Prep & Unit Conversion ---
    # Convert N -> kN, Nm -> kNm, m -> mm
    df_plot = df.copy()
    df_plot['shear_kn'] = df_plot['shear'] / 1000.0
    df_plot['moment_knm'] = df_plot['moment'] / 1000.0
    df_plot['deflection_mm'] = df_plot['deflection'] * 1000.0

    # Invert Moment for "Tension Side" plotting (Civil Convention: +Moment plotted Downwards)
    # Note: We plot -Moment so that Sagging (+) appears below axis (Visual preference)
    # Or strictly: Positive Y is Up. If we want Tension Side (Bottom), and Moment is +, we plot -Y.
    df_plot['moment_plot'] = -df_plot['moment_knm']

    cum_spans = [0] + list(np.cumsum(spans))
    total_len = cum_spans[-1] if cum_spans else 0
    
    # Prepare Loads
    clean_loads = []
    if loads:
        for l in loads:
            try:
                span_idx = int(l.get('span_index', 0))
                local_x = float(l.get('x', 0))
                abs_x = cum_spans[span_idx] + local_x if span_idx < len(cum_spans)-1 else local_x
                clean_loads.append({
                    'mag': float(l.get('mag', 0)), # Display raw input value
                    'global_x': abs_x,
                    'type': str(l.get('type', 'P')),
                    'case': str(l.get('case', 'DL')),
                    'dist': float(l.get('dist', 0))
                })
            except: continue

    # --- 2. Create Plotly Subplots ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.06,
        subplot_titles=(
            "Structure Model", 
            "Shear Force (kN)", 
            "Bending Moment (kNm) - [Tension Side]", 
            "Deflection (mm)"
        ),
        row_heights=[0.15, 0.25, 0.30, 0.30]
    )

    # --- ROW 1: Structure ---
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], line=dict(color='black', width=4), hoverinfo='skip'), row=1, col=1)
    # Supports
    for node_idx, val in reac.items():
        if isinstance(node_idx, int) and node_idx < len(cum_spans):
            x = cum_spans[node_idx]
            fig.add_trace(go.Scatter(
                x=[x], y=[-0.1], 
                mode='markers', 
                marker=dict(symbol='triangle-up', size=15, color='#2C3E50'), 
                name=f"R={val/1000:.2f} kN"
            ), row=1, col=1)

    # Loads
    for l in clean_loads:
        color = "#C0392B" if l['case'] == 'LL' else "#7F8C8D"
        lbl_text = f"{l['mag']}"
        if l['type'] == 'P':
            fig.add_annotation(
                x=l['global_x'], y=0, ax=0, ay=-50, arrowhead=2, 
                text=lbl_text, font=dict(color=color, size=12, family="Arial Black"), row=1, col=1
            )
        elif l['type'] == 'U':
            x_end = l['global_x'] + l['dist']
            fig.add_shape(type="rect", x0=l['global_x'], x1=x_end, y0=0.05, y1=0.2, fillcolor=color, opacity=0.3, line_width=0, row=1, col=1)
            fig.add_annotation(x=(l['global_x']+x_end)/2, y=0.25, text=lbl_text, showarrow=False, font=dict(color=color, size=10), row=1, col=1)

    # --- ROW 2: Shear (kN) ---
    fig.add_trace(go.Scatter(
        x=df_plot['x'], y=df_plot['shear_kn'], 
        fill='tozeroy', line=dict(color='#D35400', width=2), name="Shear (kN)"
    ), row=2, col=1)

    # Annotate Max/Min Shear
    v_max = df_plot['shear_kn'].max()
    v_min = df_plot['shear_kn'].min()
    fig.add_annotation(x=df_plot.loc[df_plot['shear_kn'].idxmax(), 'x'], y=v_max, text=f"{v_max:.2f}", showarrow=False, yshift=10, row=2, col=1)
    fig.add_annotation(x=df_plot.loc[df_plot['shear_kn'].idxmin(), 'x'], y=v_min, text=f"{v_min:.2f}", showarrow=False, yshift=-10, row=2, col=1)

    # --- ROW 3: Moment (kNm) ---
    # Plot using Tension Side convention (Sagging is + but plotted down)
    fig.add_trace(go.Scatter(
        x=df_plot['x'], y=df_plot['moment_plot'], 
        fill='tozeroy', line=dict(color='#2980B9', width=2), name="Moment (kNm)"
    ), row=3, col=1)

    # Annotate Moment Peaks
    # Note: Since we plotted inverted, Max Value in Data is Max Sagging (Plotted lowest)
    m_max_val = df_plot['moment_knm'].max() # Sagging (+)
    m_min_val = df_plot['moment_knm'].min() # Hogging (-)

    if abs(m_max_val) > 0.1:
        # Locate index of max positive moment
        idx_max = df_plot['moment_knm'].idxmax()
        fig.add_annotation(
            x=df_plot.loc[idx_max, 'x'], 
            y=df_plot.loc[idx_max, 'moment_plot'], 
            text=f"M+ {m_max_val:.2f}", 
            arrowhead=1, ax=0, ay=30, # Arrow points up to the bottom curve
            font=dict(color="blue"), row=3, col=1
        )
        
    if abs(m_min_val) > 0.1:
        # Locate index of max negative moment
        idx_min = df_plot['moment_knm'].idxmin()
        fig.add_annotation(
            x=df_plot.loc[idx_min, 'x'], 
            y=df_plot.loc[idx_min, 'moment_plot'], 
            text=f"M- {m_min_val:.2f}", 
            arrowhead=1, ax=0, ay=-30, 
            font=dict(color="red"), row=3, col=1
        )

    # --- ROW 4: Deflection (mm) ---
    fig.add_trace(go.Scatter(
        x=df_plot['x'], y=df_plot['deflection_mm'], 
        fill='tozeroy', line=dict(color='#27AE60', width=2), name="Deflection (mm)"
    ), row=4, col=1)
    
    # Annotate Max Deflection
    d_max_idx = df_plot['deflection_mm'].abs().idxmax()
    d_val = df_plot.loc[d_max_idx, 'deflection_mm']
    fig.add_annotation(
        x=df_plot.loc[d_max_idx, 'x'], y=d_val, 
        text=f"δ = {d_val:.2f} mm", 
        arrowhead=1, ay=30 if d_val < 0 else -30, row=4, col=1
    )

    fig.update_layout(height=900, showlegend=False, template="plotly_white", hovermode="x unified")
    return fig
