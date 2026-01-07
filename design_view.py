
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# ==========================================
# 1. CAPACITY CHECK (Standard Engineering Style)
# ==========================================
def plot_capacity_vs_demand(df_span, phi_Mn_pos, phi_Mn_neg):
    """
    Standard Plot: 
    - Dashed Lines = Capacity Limits (+/-)
    - Solid Line = Demand (Mu)
    """
    mu_kNm = df_span['moment'] / 1000.0
    x = df_span['x']
    x_min, x_max = x.min(), x.max()
    
    fig = go.Figure()

    # 1. Limits (Capacity) - เส้นประ
    # Positive Limit (Green)
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], y=[phi_Mn_pos, phi_Mn_pos],
        mode='lines', name=f'+φMn (Cap) = {phi_Mn_pos:.2f}',
        line=dict(color='green', width=2, dash='dash')
    ))
    # Negative Limit (Red)
    neg_cap = -abs(phi_Mn_neg)
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], y=[neg_cap, neg_cap],
        mode='lines', name=f'-φMn (Cap) = {abs(neg_cap):.2f}',
        line=dict(color='red', width=2, dash='dash')
    ))

    # 2. Demand (Applied Moment) - เส้นทึบ
    fig.add_trace(go.Scatter(
        x=x, y=mu_kNm,
        mode='lines', name='Mu (Applied)',
        line=dict(color='blue', width=3),
        fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.05)'
    ))
    
    # Annotations (Max/Min)
    max_mu = mu_kNm.max()
    min_mu = mu_kNm.min()
    
    if max_mu > 0.01:
        fig.add_annotation(x=df_span.loc[mu_kNm.idxmax(), 'x'], y=max_mu, text=f"{max_mu:.2f}", showarrow=True, arrowhead=1, yshift=10)
    if min_mu < -0.01:
        fig.add_annotation(x=df_span.loc[mu_kNm.idxmin(), 'x'], y=min_mu, text=f"{min_mu:.2f}", showarrow=True, arrowhead=1, ay=30)

    fig.update_layout(
        title="<b>Capacity Check:</b> Blue Line MUST be inside Dashed Lines",
        xaxis_title="Distance (m)",
        yaxis_title="Moment (kNm)",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified"
    )
    return fig

# ==========================================
# 2. ANALYSIS DIAGRAMS (Vertical Lines Fixed)
# ==========================================
def draw_interactive_diagrams(df, reac, spans, sup_df, loads):
    # --- Data Prep ---
    df_plot = df.copy()
    df_plot['shear_kn'] = df_plot['shear'] / 1000.0
    df_plot['moment_knm'] = df_plot['moment'] / 1000.0
    df_plot['deflection_mm'] = df_plot['deflection'] * 1000.0
    df_plot['moment_plot'] = -df_plot['moment_knm'] 

    cum_spans = [0] + list(np.cumsum(spans))
    total_len = cum_spans[-1] if cum_spans else 0
    
    # Clean Loads
    clean_loads = []
    if loads:
        for l in loads:
            try:
                span_idx = int(l.get('span_index', 0))
                local_x = float(l.get('x', 0))
                abs_x = cum_spans[span_idx] + local_x if span_idx < len(cum_spans)-1 else local_x
                clean_loads.append({'mag': float(l['mag']), 'global_x': abs_x, 'type': l.get('type','P'), 'case': l.get('case','DL'), 'dist': float(l.get('dist',0))})
            except: continue

    # --- Create Subplots ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.06,
        subplot_titles=("Structure Model", "Shear Force (V)", "Bending Moment (M)", "Deflection (δ)"),
        row_heights=[0.15, 0.25, 0.30, 0.30]
    )

    # === 1. FORCED VERTICAL GRID LINES (ใช้วิธีที่เสถียรที่สุด) ===
    # ใช้คำสั่ง add_vline วนลูปใส่ทุก Row
    for x_pos in cum_spans:
        for r in [1, 2, 3, 4]:
            fig.add_vline(
                x=x_pos, 
                row=r, col=1, 
                line_width=1, 
                line_dash="dash", 
                line_color="gray", 
                opacity=0.5,
                layer="below" # ให้เส้นอยู่ข้างหลังกราฟ
            )

    # === ROW 1: Structure ===
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], line=dict(color='black', width=5), hoverinfo='skip'), row=1, col=1)
    
    # Supports & Labels
    for i, x_pos in enumerate(cum_spans):
        sup_type = "Pin"
        if not sup_df.empty:
             match = sup_df[sup_df['id'] == i]
             if not match.empty: sup_type = match.iloc[0]['type']
        
        stype = sup_type.lower()
        if 'pin' in stype: sym, col = 'triangle-up', '#2C3E50'
        elif 'roller' in stype: sym, col = 'circle', '#27AE60'
        elif 'fix' in stype: sym, col = 'square', '#000000'
        else: sym, col = 'triangle-up', 'gray'
        
        # Label Text
        label = sup_type.capitalize()
        if reac and i in reac:
            label += f"<br>R={reac[i]/1000:.2f} kN"

        # วาด Support (ขยับ y ลงมา -0.2 เพื่อให้อยู่ใต้คาน)
        fig.add_trace(go.Scatter(
            x=[x_pos], y=[-0.2], 
            mode='markers+text',
            marker=dict(symbol=sym, size=20, color=col, line=dict(width=2, color='black')),
            text=[label], textposition="bottom center",
            hoverinfo='none'
        ), row=1, col=1)

    # Loads
    for l in clean_loads:
        c = "#C0392B" if l['case'] == 'LL' else "#7F8C8D"
        if l['type'] == 'P':
            fig.add_annotation(x=l['global_x'], y=0, ax=0, ay=-40, arrowhead=2, text=str(l['mag']), font=dict(color=c), row=1, col=1)
        elif l['type'] == 'U':
            fig.add_shape(type="rect", x0=l['global_x'], x1=l['global_x']+l['dist'], y0=0.05, y1=0.25, fillcolor=c, opacity=0.3, line_width=0, row=1, col=1)
            fig.add_annotation(x=l['global_x'] + l['dist']/2, y=0.3, text=str(l['mag']), showarrow=False, font=dict(color=c), row=1, col=1)

    # === ROW 2: Shear ===
    fig.add_trace(go.Scatter(x=df_plot['x'], y=df_plot['shear_kn'], fill='tozeroy', line=dict(color='#D35400'), name="Shear"), row=2, col=1)
    vmax, vmin = df_plot['shear_kn'].max(), df_plot['shear_kn'].min()
    fig.add_annotation(x=df_plot.loc[df_plot['shear_kn'].idxmax(), 'x'], y=vmax, text=f"{vmax:.2f}", showarrow=False, yshift=10, row=2, col=1)
    fig.add_annotation(x=df_plot.loc[df_plot['shear_kn'].idxmin(), 'x'], y=vmin, text=f"{vmin:.2f}", showarrow=False, yshift=-10, row=2, col=1)

    # === ROW 3: Moment ===
    fig.add_trace(go.Scatter(x=df_plot['x'], y=df_plot['moment_plot'], fill='tozeroy', line=dict(color='#2980B9'), name="Moment"), row=3, col=1)
    msag, mhog = df_plot['moment_knm'].max(), df_plot['moment_knm'].min()
    if msag > 0.01: fig.add_annotation(x=df_plot.loc[df_plot['moment_knm'].idxmax(), 'x'], y=-msag, text=f"{msag:.2f}", arrowhead=1, ay=30, row=3, col=1)
    if mhog < -0.01: fig.add_annotation(x=df_plot.loc[df_plot['moment_knm'].idxmin(), 'x'], y=-mhog, text=f"{mhog:.2f}", arrowhead=1, ay=-30, row=3, col=1)

    # === ROW 4: Deflection ===
    fig.add_trace(go.Scatter(x=df_plot['x'], y=df_plot['deflection_mm'], fill='tozeroy', line=dict(color='#27AE60'), name="Deflection"), row=4, col=1)
    didx = df_plot['deflection_mm'].abs().idxmax()
    dval = df_plot.loc[didx, 'deflection_mm']
    if abs(dval) > 0.001: 
        fig.add_annotation(x=df_plot.loc[didx, 'x'], y=dval, text=f"{dval:.2f}", arrowhead=1, ay=30 if dval<0 else -30, row=4, col=1)

    # === Layout Finalization ===
    fig.update_layout(height=900, showlegend=False, template="plotly_white", hovermode="x unified")
    
    # Adjust Y-axis for Structure (Model)
    # Range [-0.8, 0.4] ensures y=0 is near top, giving space below for supports
    fig.update_yaxes(range=[-0.8, 0.4], showticklabels=False, row=1, col=1)
    
    # Labels
    fig.update_yaxes(title_text="V (kN)", row=2, col=1)
    fig.update_yaxes(title_text="M (kNm)", row=3, col=1)
    fig.update_yaxes(title_text="δ (mm)", row=4, col=1)
    fig.update_xaxes(title_text="Distance (m)", row=4, col=1)

    return fig
