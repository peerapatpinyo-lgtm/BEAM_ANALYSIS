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
    # Convert Demand from N-m to kNm
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
    neg_cap = -abs(phi_Mn_neg)
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], 
        y=[neg_cap, neg_cap],
        mode='lines',
        name=f'Capacity -M (φMn = {abs(neg_cap):.1f})',
        line=dict(color='#C0392B', width=2, dash='dash')
    ))

    # --- Annotations (Peak Values) ---
    max_mu = mu_kNm.max()
    min_mu = mu_kNm.min()
    
    # Show Max Positive if significant
    if max_mu > 0.1:
        fig.add_annotation(
            x=df_span.loc[mu_kNm.idxmax(), 'x'], y=max_mu,
            text=f"Max+: {max_mu:.2f}", showarrow=True, arrowhead=1, yshift=10 # <--- แก้ไขตรงนี้ครับ (ys -> yshift)
        )
    # Show Max Negative if significant
    if min_mu < -0.1:
        fig.add_annotation(
            x=df_span.loc[mu_kNm.idxmin(), 'x'], y=min_mu,
            text=f"Max-: {min_mu:.2f}", showarrow=True, arrowhead=1, ay=30
        )

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
    Units Displayed: kN, kNm, mm
    """
    # --- 1. Data Prep & Unit Conversion ---
    # Convert Solver output (N, Nm, m) to Display units (kN, kNm, mm)
    df_plot = df.copy()
    df_plot['shear_kn'] = df_plot['shear'] / 1000.0
    df_plot['moment_knm'] = df_plot['moment'] / 1000.0
    df_plot['deflection_mm'] = df_plot['deflection'] * 1000.0

    # Invert Moment for "Tension Side" plotting (Civil Eng Convention)
    # Note: We plot (-Moment) so Sagging (+) appears below axis
    df_plot['moment_plot'] = -df_plot['moment_knm']

    cum_spans = [0] + list(np.cumsum(spans))
    total_len = cum_spans[-1] if cum_spans else 0
    
    # Prepare Loads for visualization
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

    # --- 2. Create Plotly Subplots ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
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
                marker=dict(symbol='triangle-up', size=14, color='#2C3E50'), 
                name=f"R={val/1000:.2f} kN",
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

    # --- ROW 2: Shear (kN) ---
    fig.add_trace(go.Scatter(
        x=df_plot['x'], y=df_plot['shear_kn'], 
        fill='tozeroy', line=dict(color='#D35400', width=2), name="Shear (kN)"
    ), row=2, col=1)

    # Annotate Max/Min Shear
    v_max = df_plot['shear_kn'].max()
    v_min = df_plot['shear_kn'].min()
    # ใช้ yshift แทน ys หรือใช้ yshift=10 ตรงๆ
    fig.add_annotation(x=df_plot.loc[df_plot['shear_kn'].idxmax(), 'x'], y=v_max, text=f"{v_max:.2f}", showarrow=False, yshift=10, font=dict(size=10), row=2, col=1)
    fig.add_annotation(x=df_plot.loc[df_plot['shear_kn'].idxmin(), 'x'], y=v_min, text=f"{v_min:.2f}", showarrow=False, yshift=-10, font=dict(size=10), row=2, col=1)

    # --- ROW 3: Moment (kNm) ---
    fig.add_trace(go.Scatter(
        x=df_plot['x'], y=df_plot['moment_plot'], 
        fill='tozeroy', line=dict(color='#2980B9', width=2), name="Moment (kNm)"
    ), row=3, col=1)

    # Annotate Moment Peaks (Using Real Values, not plotted values)
    m_sag_max = df_plot['moment_knm'].max() # Value > 0
    m_hog_max = df_plot['moment_knm'].min() # Value < 0

    if m_sag_max > 0.5:
        idx = df_plot['moment_knm'].idxmax()
        fig.add_annotation(
            x=df_plot.loc[idx, 'x'], y=df_plot.loc[idx, 'moment_plot'], 
            text=f"M(+) {m_sag_max:.2f}", arrowhead=1, ax=0, ay=30, font=dict(color="blue", size=10), row=3, col=1
        )
    
    if m_hog_max < -0.5:
        idx = df_plot['moment_knm'].idxmin()
        fig.add_annotation(
            x=df_plot.loc[idx, 'x'], y=df_plot.loc[idx, 'moment_plot'], 
            text=f"M(-) {m_hog_max:.2f}", arrowhead=1, ax=0, ay=-30, font=dict(color="red", size=10), row=3, col=1
        )

    # --- ROW 4: Deflection (mm) ---
    fig.add_trace(go.Scatter(
        x=df_plot['x'], y=df_plot['deflection_mm'], 
        fill='tozeroy', line=dict(color='#27AE60', width=2), name="Deflection (mm)"
    ), row=4, col=1)
    
    # Annotate Max Deflection
    d_max_idx = df_plot['deflection_mm'].abs().idxmax()
    d_val = df_plot.loc[d_max_idx, 'deflection_mm']
    if abs(d_val) > 0.01:
        fig.add_annotation(
            x=df_plot.loc[d_max_idx, 'x'], y=d_val, 
            text=f"δ={d_val:.2f}", arrowhead=1, ay=30 if d_val < 0 else -30, row=4, col=1
        )

    fig.update_layout(height=850, showlegend=False, template="plotly_white", hovermode="x unified")
    return fig
