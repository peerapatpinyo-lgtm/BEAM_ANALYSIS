import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# ==========================================
# 1. CAPACITY CHECK VISUALIZATION (Fixed Dashed Lines)
# ==========================================
def plot_capacity_vs_demand(df_span, phi_Mn_pos, phi_Mn_neg):
    """
    สร้างกราฟเปรียบเทียบ Mu vs PhiMn
    """
    # Convert Demand to kNm
    mu_kNm = df_span['moment'] / 1000.0
    
    fig = go.Figure()
    
    x_min, x_max = df_span['x'].min(), df_span['x'].max()
    
    # 1. Plot +PhiMn (Positive Capacity) - GREEN DASHED
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], 
        y=[phi_Mn_pos, phi_Mn_pos],
        mode='lines',
        name=f'+φMn = {phi_Mn_pos:.2f}',
        line=dict(color='#27AE60', width=3, dash='dash') # <--- เส้นประหนาขึ้น
    ))
    
    # 2. Plot -PhiMn (Negative Capacity) - RED DASHED
    neg_cap = -abs(phi_Mn_neg)
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], 
        y=[neg_cap, neg_cap],
        mode='lines',
        name=f'-φMn = {abs(neg_cap):.2f}',
        line=dict(color='#C0392B', width=3, dash='dash') # <--- เส้นประหนาขึ้น
    ))

    # 3. Plot Mu (Demand) - SOLID BLUE
    fig.add_trace(go.Scatter(
        x=df_span['x'], 
        y=mu_kNm, 
        mode='lines',
        name='Mu (Applied)',
        line=dict(color='#2980B9', width=3),
        fill='tozeroy',
        fillcolor='rgba(41, 128, 185, 0.1)'
    ))
    
    # Annotations for Max/Min
    max_mu = mu_kNm.max()
    min_mu = mu_kNm.min()
    
    if max_mu > 0.1:
        fig.add_annotation(x=df_span.loc[mu_kNm.idxmax(), 'x'], y=max_mu, text=f"{max_mu:.2f}", showarrow=True, arrowhead=1, yshift=10)
    if min_mu < -0.1:
        fig.add_annotation(x=df_span.loc[mu_kNm.idxmin(), 'x'], y=min_mu, text=f"{min_mu:.2f}", showarrow=True, arrowhead=1, ay=30)

    # Layout Setup
    fig.update_layout(
        title="<b>Moment Capacity Check:</b> Applied ($M_u$) vs Capacity ($\phi M_n$)",
        xaxis_title="<b>Distance (m)</b>", # <--- ระบุแกน X
        yaxis_title="<b>Moment (kNm)</b>", # <--- ระบุแกน Y
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=50, b=50),
        hovermode="x unified"
    )
    return fig

# ==========================================
# 2. ANALYSIS DIAGRAMS (Fixed Floating Supports & Labels)
# ==========================================
def draw_interactive_diagrams(df, reac, spans, sup_df, loads, dl_factor=1.4, ll_factor=1.7):
    # --- Data Prep ---
    df_plot = df.copy()
    df_plot['shear_kn'] = df_plot['shear'] / 1000.0
    df_plot['moment_knm'] = df_plot['moment'] / 1000.0
    df_plot['deflection_mm'] = df_plot['deflection'] * 1000.0
    df_plot['moment_plot'] = -df_plot['moment_knm'] # Tension Side

    cum_spans = [0] + list(np.cumsum(spans))
    total_len = cum_spans[-1] if cum_spans else 0
    
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

    # --- Plot Setup ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=("Structure Model", "Shear Force (kN)", "Bending Moment (kNm)", "Deflection (mm)"),
        row_heights=[0.15, 0.25, 0.30, 0.30]
    )

    # --- ROW 1: Structure Model ---
    # Beam Line (Black Thick)
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], line=dict(color='black', width=5), hoverinfo='skip'), row=1, col=1)
    
    # Supports (Fixed Floating Issue: Plot at y=0)
    for _, sup in sup_df.iterrows():
        sx = cum_spans[int(sup['id'])]
        stype = sup['type'].lower()
        
        # Shapes & Colors
        if 'pin' in stype:
            sym = 'triangle-up'
            col = '#2C3E50'
        elif 'roller' in stype:
            sym = 'circle'
            col = '#27AE60'
        elif 'fix' in stype:
            sym = 'square'
            col = '#000000'
        else:
            sym = 'triangle-up'
            col = 'gray'

        # Plot Marker EXACTLY at y=0
        fig.add_trace(go.Scatter(
            x=[sx], y=[0], 
            mode='markers', 
            marker=dict(symbol=sym, size=20, color=col, line=dict(width=2, color='black')), 
            name=f"{stype.capitalize()}",
            hoverinfo="name"
        ), row=1, col=1)

    # Loads
    for l in clean_loads:
        color = "#C0392B" if l['case'] == 'LL' else "#7F8C8D"
        lbl = f"{l['mag']}"
        if l['type'] == 'P':
            fig.add_annotation(
                x=l['global_x'], y=0, ax=0, ay=-40, arrowhead=2, 
                text=lbl, font=dict(color=color, size=11, family="Arial Black"), row=1, col=1
            )
        elif l['type'] == 'U':
            x_end = l['global_x'] + l['dist']
            fig.add_shape(type="rect", x0=l['global_x'], x1=x_end, y0=0.05, y1=0.2, fillcolor=color, opacity=0.3, line_width=0, row=1, col=1)
            fig.add_annotation(x=(l['global_x']+x_end)/2, y=0.25, text=lbl, showarrow=False, font=dict(color=color, size=10), row=1, col=1)

    # --- ROW 2: Shear ---
    fig.add_trace(go.Scatter(x=df_plot['x'], y=df_plot['shear_kn'], fill='tozeroy', line=dict(color='#D35400'), name="Shear"), row=2, col=1)
    # Peak Labels
    v_max, v_min = df_plot['shear_kn'].max(), df_plot['shear_kn'].min()
    fig.add_annotation(x=df_plot.loc[df_plot['shear_kn'].idxmax(), 'x'], y=v_max, text=f"{v_max:.2f}", showarrow=False, yshift=10, font=dict(size=10), row=2, col=1)
    fig.add_annotation(x=df_plot.loc[df_plot['shear_kn'].idxmin(), 'x'], y=v_min, text=f"{v_min:.2f}", showarrow=False, yshift=-10, font=dict(size=10), row=2, col=1)

    # --- ROW 3: Moment ---
    fig.add_trace(go.Scatter(x=df_plot['x'], y=df_plot['moment_plot'], fill='tozeroy', line=dict(color='#2980B9'), name="Moment"), row=3, col=1)
    # Peak Labels
    m_sag = df_plot['moment_knm'].max()
    m_hog = df_plot['moment_knm'].min()
    if m_sag > 0.1:
        fig.add_annotation(x=df_plot.loc[df_plot['moment_knm'].idxmax(), 'x'], y=-m_sag, text=f"M(+) {m_sag:.2f}", arrowhead=1, ay=30, row=3, col=1)
    if m_hog < -0.1:
        fig.add_annotation(x=df_plot.loc[df_plot['moment_knm'].idxmin(), 'x'], y=-m_hog, text=f"M(-) {m_hog:.2f}", arrowhead=1, ay=-30, row=3, col=1)

    # --- ROW 4: Deflection ---
    fig.add_trace(go.Scatter(x=df_plot['x'], y=df_plot['deflection_mm'], fill='tozeroy', line=dict(color='#27AE60'), name="Deflection"), row=4, col=1)
    # Peak Labels
    d_max_idx = df_plot['deflection_mm'].abs().idxmax()
    d_val = df_plot.loc[d_max_idx, 'deflection_mm']
    fig.add_annotation(x=df_plot.loc[d_max_idx, 'x'], y=d_val, text=f"δ={d_val:.2f}", arrowhead=1, ay=30 if d_val<0 else -30, row=4, col=1)

    # --- FINAL LAYOUT (Axes Labels & Scaling) ---
    fig.update_layout(height=900, showlegend=False, template="plotly_white", hovermode="x unified")
    
    # Fix Y-axis for Structure Row to prevent floating supports
    fig.update_yaxes(range=[-0.5, 0.5], showticklabels=False, row=1, col=1) 
    
    # Add Axis Labels
    fig.update_yaxes(title_text="V (kN)", row=2, col=1)
    fig.update_yaxes(title_text="M (kNm)", row=3, col=1)
    fig.update_yaxes(title_text="δ (mm)", row=4, col=1)
    fig.update_xaxes(title_text="<b>Distance (m)</b>", row=4, col=1) # Only on bottom graph

    return fig
