import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# ==========================================
# 1. CAPACITY CHECK VISUALIZATION
# ==========================================
def plot_capacity_vs_demand(df_span, phi_Mn_pos, phi_Mn_neg):
    """
    สร้างกราฟเปรียบเทียบ Mu vs PhiMn
    """
    mu_kNm = df_span['moment'] / 1000.0
    fig = go.Figure()
    
    x_min, x_max = df_span['x'].min(), df_span['x'].max()
    
    # +PhiMn (Green Dashed)
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], y=[phi_Mn_pos, phi_Mn_pos],
        mode='lines', name=f'+φMn',
        line=dict(color='#27AE60', width=2, dash='dash')
    ))
    
    # -PhiMn (Red Dashed)
    neg_cap = -abs(phi_Mn_neg)
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], y=[neg_cap, neg_cap],
        mode='lines', name=f'-φMn',
        line=dict(color='#C0392B', width=2, dash='dash')
    ))

    # Mu (Solid Blue)
    fig.add_trace(go.Scatter(
        x=df_span['x'], y=mu_kNm,
        mode='lines', name='Mu',
        line=dict(color='#2980B9', width=3),
        fill='tozeroy', fillcolor='rgba(41, 128, 185, 0.1)'
    ))
    
    # Annotations
    max_mu = mu_kNm.max()
    min_mu = mu_kNm.min()
    
    if max_mu > 0.1:
        fig.add_annotation(x=df_span.loc[mu_kNm.idxmax(), 'x'], y=max_mu, text=f"{max_mu:.2f}", showarrow=True, arrowhead=1, yshift=10)
    if min_mu < -0.1:
        fig.add_annotation(x=df_span.loc[mu_kNm.idxmin(), 'x'], y=min_mu, text=f"{min_mu:.2f}", showarrow=True, arrowhead=1, ay=30)

    fig.update_layout(
        title="<b>Capacity Check:</b> $M_u$ vs $\phi M_n$",
        xaxis_title="<b>Distance (m)</b>",
        yaxis_title="<b>Moment (kNm)</b>",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified"
    )
    return fig

# ==========================================
# 2. ANALYSIS DIAGRAMS (Vertical Grids + Fixed Support)
# ==========================================
def draw_interactive_diagrams(df, reac, spans, sup_df, loads, dl_factor=1.4, ll_factor=1.7):
    # --- Data Prep ---
    df_plot = df.copy()
    df_plot['shear_kn'] = df_plot['shear'] / 1000.0
    df_plot['moment_knm'] = df_plot['moment'] / 1000.0
    df_plot['deflection_mm'] = df_plot['deflection'] * 1000.0
    df_plot['moment_plot'] = -df_plot['moment_knm'] 

    cum_spans = [0] + list(np.cumsum(spans))
    total_len = cum_spans[-1] if cum_spans else 0
    
    # Load Prep
    clean_loads = []
    if loads:
        for l in loads:
            try:
                span_idx = int(l.get('span_index', 0))
                local_x = float(l.get('x', 0))
                abs_x = cum_spans[span_idx] + local_x if span_idx < len(cum_spans)-1 else local_x
                clean_loads.append({
                    'mag': float(l.get('mag', 0)),
                    'global_x': abs_x,
                    'type': str(l.get('type', 'P')),
                    'case': str(l.get('case', 'DL')),
                    'dist': float(l.get('dist', 0))
                })
            except: continue

    # --- Create Subplots ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        subplot_titles=("Structure", "Shear (kN)", "Moment (kNm)", "Deflection (mm)"),
        row_heights=[0.15, 0.25, 0.30, 0.30]
    )

    # === ROW 1: Structure ===
    # Draw Beam Line at y=0
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], line=dict(color='black', width=6), hoverinfo='skip'), row=1, col=1)
    
    # Draw Supports & Vertical Grid Lines
    for i, x_pos in enumerate(cum_spans):
        # 1. Add Vertical Dashed Line (Grid) across ALL subplots
        fig.add_vline(x=x_pos, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

        # 2. Draw Support Symbols
        # Find type from input
        sup_type = "Pin" # Default
        if not sup_df.empty:
            match = sup_df[sup_df['id'] == i]
            if not match.empty:
                sup_type = match.iloc[0]['type']

        stype = sup_type.lower()
        if 'pin' in stype:
            sym, col = 'triangle-up', '#2C3E50'
        elif 'roller' in stype:
            sym, col = 'circle', '#27AE60'
        elif 'fix' in stype:
            sym, col = 'square', '#000000'
        else:
            sym, col = 'triangle-up', 'gray'

        # Plot Support Marker exactly at y=0 (behind beam) or slightly offset
        # size=20 makes it visible.
        fig.add_trace(go.Scatter(
            x=[x_pos], y=[-0.02], # Offset slightly down visually
            mode='markers', 
            marker=dict(symbol=sym, size=20, color=col, line=dict(width=2, color='black')), 
            name=sup_type, hoverinfo="name"
        ), row=1, col=1)

    # Draw Loads
    for l in clean_loads:
        color = "#C0392B" if l['case'] == 'LL' else "#7F8C8D"
        if l['type'] == 'P':
            fig.add_annotation(
                x=l['global_x'], y=0, ax=0, ay=-40, arrowhead=2, 
                text=str(l['mag']), font=dict(color=color, size=11, family="Arial Black"), row=1, col=1
            )
        elif l['type'] == 'U':
            x_end = l['global_x'] + l['dist']
            fig.add_shape(type="rect", x0=l['global_x'], x1=x_end, y0=0.05, y1=0.2, fillcolor=color, opacity=0.3, line_width=0, row=1, col=1)
            fig.add_annotation(x=(l['global_x']+x_end)/2, y=0.25, text=str(l['mag']), showarrow=False, font=dict(color=color, size=10), row=1, col=1)

    # === ROW 2: Shear ===
    fig.add_trace(go.Scatter(x=df_plot['x'], y=df_plot['shear_kn'], fill='tozeroy', line=dict(color='#D35400'), name="Shear"), row=2, col=1)
    v_max, v_min = df_plot['shear_kn'].max(), df_plot['shear_kn'].min()
    fig.add_annotation(x=df_plot.loc[df_plot['shear_kn'].idxmax(), 'x'], y=v_max, text=f"{v_max:.2f}", showarrow=False, yshift=10, row=2, col=1)
    fig.add_annotation(x=df_plot.loc[df_plot['shear_kn'].idxmin(), 'x'], y=v_min, text=f"{v_min:.2f}", showarrow=False, yshift=-10, row=2, col=1)

    # === ROW 3: Moment ===
    fig.add_trace(go.Scatter(x=df_plot['x'], y=df_plot['moment_plot'], fill='tozeroy', line=dict(color='#2980B9'), name="Moment"), row=3, col=1)
    m_sag = df_plot['moment_knm'].max()
    m_hog = df_plot['moment_knm'].min()
    if m_sag > 0.1:
        fig.add_annotation(x=df_plot.loc[df_plot['moment_knm'].idxmax(), 'x'], y=-m_sag, text=f"{m_sag:.2f}", arrowhead=1, ay=30, row=3, col=1)
    if m_hog < -0.1:
        fig.add_annotation(x=df_plot.loc[df_plot['moment_knm'].idxmin(), 'x'], y=-m_hog, text=f"{m_hog:.2f}", arrowhead=1, ay=-30, row=3, col=1)

    # === ROW 4: Deflection ===
    fig.add_trace(go.Scatter(x=df_plot['x'], y=df_plot['deflection_mm'], fill='tozeroy', line=dict(color='#27AE60'), name="Deflection"), row=4, col=1)
    d_max_idx = df_plot['deflection_mm'].abs().idxmax()
    d_val = df_plot.loc[d_max_idx, 'deflection_mm']
    fig.add_annotation(x=df_plot.loc[d_max_idx, 'x'], y=d_val, text=f"{d_val:.2f}", arrowhead=1, ay=30 if d_val<0 else -30, row=4, col=1)

    # === Layout Settings ===
    fig.update_layout(height=850, showlegend=False, template="plotly_white", hovermode="x unified")
    
    # Adjust Y-axis for Structure to NOT look floating
    # Set range tight around 0 (-0.2 to 0.4) so beam is at bottom
    fig.update_yaxes(range=[-0.25, 0.4], showticklabels=False, row=1, col=1) 
    
    # Add Axis Labels
    fig.update_yaxes(title_text="V (kN)", row=2, col=1)
    fig.update_yaxes(title_text="M (kNm)", row=3, col=1)
    fig.update_yaxes(title_text="δ (mm)", row=4, col=1)
    fig.update_xaxes(title_text="<b>Distance (m)</b>", row=4, col=1)

    return fig
