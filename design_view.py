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
    Units: kNm
    """
    mu_kNm = df_span['moment'] / 1000.0
    
    fig = go.Figure()
    
    # Plot Mu
    fig.add_trace(go.Scatter(
        x=df_span['x'], y=mu_kNm, mode='lines', name='Mu',
        line=dict(color='#2980B9', width=3),
        fill='tozeroy', fillcolor='rgba(41, 128, 185, 0.1)'
    ))
    
    # Plot Capacity
    x_min, x_max = df_span['x'].min(), df_span['x'].max()
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], y=[phi_Mn_pos, phi_Mn_pos],
        mode='lines', name=f'φMn+ ({phi_Mn_pos:.1f})',
        line=dict(color='#27AE60', width=2, dash='dash')
    ))
    
    neg_cap = -abs(phi_Mn_neg)
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], y=[neg_cap, neg_cap],
        mode='lines', name=f'φMn- ({abs(neg_cap):.1f})',
        line=dict(color='#C0392B', width=2, dash='dash')
    ))

    # Annotations
    max_mu = mu_kNm.max()
    min_mu = mu_kNm.min()
    
    if max_mu > 0.1:
        fig.add_annotation(x=df_span.loc[mu_kNm.idxmax(), 'x'], y=max_mu,
                           text=f"{max_mu:.2f}", showarrow=True, arrowhead=1, yshift=10)
    if min_mu < -0.1:
        fig.add_annotation(x=df_span.loc[mu_kNm.idxmin(), 'x'], y=min_mu,
                           text=f"{min_mu:.2f}", showarrow=True, arrowhead=1, ay=30)

    fig.update_layout(
        title="Moment Capacity Check (kNm)",
        xaxis_title="Position (m)",
        yaxis_title="Moment (kNm)",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
        hovermode="x unified"
    )
    return fig

# ==========================================
# 2. ANALYSIS DIAGRAMS (FULL ENGINEERING DETAIL)
# ==========================================
def draw_interactive_diagrams(df, reac, spans, sup_df, loads, dl_factor=1.4, ll_factor=1.7):
    """
    วาดกราฟแบบ Full Engineering Detail
    - มี Node Number
    - มี Grid Line แนวดิ่ง
    - มี Axis Label ชัดเจน
    """
    # 1. Data Conversion
    df_plot = df.copy()
    df_plot['shear_kn'] = df_plot['shear'] / 1000.0
    df_plot['moment_knm'] = df_plot['moment'] / 1000.0
    df_plot['deflection_mm'] = df_plot['deflection'] * 1000.0
    # Tension Side Convention (Invert Moment)
    df_plot['moment_plot'] = -df_plot['moment_knm']

    cum_spans = [0] + list(np.cumsum(spans))
    total_len = cum_spans[-1] if cum_spans else 0
    
    # 2. Prepare Loads
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
                    'dist': float(l.get('dist', 0))
                })
            except: continue

    # 3. Create Subplots
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>Structure Model & Nodes</b>", 
            "<b>Shear Force Diagram (SFD)</b>", 
            "<b>Bending Moment Diagram (BMD)</b>", 
            "<b>Deflection</b>"
        ),
        row_heights=[0.15, 0.25, 0.30, 0.30]
    )

    # --- ROW 1: Structure ---
    # Beam Line
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], line=dict(color='black', width=5), hoverinfo='skip'), row=1, col=1)
    
    # Supports & Node Numbers
    for i, x in enumerate(cum_spans):
        # Support Marker
        fig.add_trace(go.Scatter(
            x=[x], y=[-0.1], 
            mode='markers', 
            marker=dict(symbol='triangle-up', size=15, color='#2C3E50'), 
            name=f"Node {i+1}",
            hoverinfo="name"
        ), row=1, col=1)
        
        # Node Number Annotation (①, ②...)
        fig.add_annotation(
            x=x, y=0.2, text=f"<b>Node {i+1}</b>", 
            showarrow=False, 
            font=dict(size=12, color="black"),
            bgcolor="#ECF0F1", bordercolor="#BDC3C7", borderwidth=1,
            row=1, col=1
        )
        
        # *** VERTICAL GRID LINES (The most important fix) ***
        # ลากเส้นประแนวดิ่งผ่านทุกกราฟ เพื่อให้อ่านค่าตรงกัน
        fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

    # Loads
    for l in clean_loads:
        lbl = f"{l['mag']}"
        if l['type'] == 'P':
            fig.add_annotation(
                x=l['global_x'], y=0, ax=0, ay=-40, arrowhead=2, arrowwidth=2, arrowcolor="#C0392B",
                text=f"<b>P={lbl}</b>", font=dict(color="#C0392B", size=11), row=1, col=1
            )
        elif l['type'] == 'U':
            x_end = l['global_x'] + l['dist']
            fig.add_shape(type="rect", x0=l['global_x'], x1=x_end, y0=0.05, y1=0.2, fillcolor="#C0392B", opacity=0.2, line_width=0, row=1, col=1)
            fig.add_annotation(x=(l['global_x']+x_end)/2, y=0.25, text=f"w={lbl}", showarrow=False, font=dict(color="#C0392B", size=10), row=1, col=1)

    # --- ROW 2: Shear (kN) ---
    fig.add_trace(go.Scatter(
        x=df_plot['x'], y=df_plot['shear_kn'], 
        fill='tozeroy', line=dict(color='#D35400', width=2), name="Shear (kN)"
    ), row=2, col=1)

    # Max/Min Labels (Big & Bold)
    v_max = df_plot['shear_kn'].max()
    v_min = df_plot['shear_kn'].min()
    fig.add_annotation(
        x=df_plot.loc[df_plot['shear_kn'].idxmax(), 'x'], y=v_max, 
        text=f"<b>{v_max:.2f}</b>", showarrow=False, yshift=15, 
        font=dict(size=12, color="#D35400"), bgcolor="rgba(255,255,255,0.8)", row=2, col=1
    )
    fig.add_annotation(
        x=df_plot.loc[df_plot['shear_kn'].idxmin(), 'x'], y=v_min, 
        text=f"<b>{v_min:.2f}</b>", showarrow=False, yshift=-15, 
        font=dict(size=12, color="#D35400"), bgcolor="rgba(255,255,255,0.8)", row=2, col=1
    )

    # --- ROW 3: Moment (kNm) ---
    fig.add_trace(go.Scatter(
        x=df_plot['x'], y=df_plot['moment_plot'], 
        fill='tozeroy', line=dict(color='#2980B9', width=2), name="Moment (kNm)"
    ), row=3, col=1)

    # Moment Peaks Labels
    m_sag = df_plot['moment_knm'].max() # +Val
    m_hog = df_plot['moment_knm'].min() # -Val

    if m_sag > 0.1:
        idx = df_plot['moment_knm'].idxmax()
        fig.add_annotation(
            x=df_plot.loc[idx, 'x'], y=df_plot.loc[idx, 'moment_plot'], 
            text=f"<b>M(+) {m_sag:.2f}</b>", arrowhead=1, ax=0, ay=30, 
            font=dict(color="blue", size=12), bgcolor="rgba(255,255,255,0.8)", row=3, col=1
        )
    
    if m_hog < -0.1:
        idx = df_plot['moment_knm'].idxmin()
        fig.add_annotation(
            x=df_plot.loc[idx, 'x'], y=df_plot.loc[idx, 'moment_plot'], 
            text=f"<b>M(-) {m_hog:.2f}</b>", arrowhead=1, ax=0, ay=-30, 
            font=dict(color="red", size=12), bgcolor="rgba(255,255,255,0.8)", row=3, col=1
        )

    # --- ROW 4: Deflection (mm) ---
    fig.add_trace(go.Scatter(
        x=df_plot['x'], y=df_plot['deflection_mm'], 
        fill='tozeroy', line=dict(color='#27AE60', width=2), name="Deflection (mm)"
    ), row=4, col=1)
    
    d_max_idx = df_plot['deflection_mm'].abs().idxmax()
    d_val = df_plot.loc[d_max_idx, 'deflection_mm']
    if abs(d_val) > 0.001:
        fig.add_annotation(
            x=df_plot.loc[d_max_idx, 'x'], y=d_val, 
            text=f"<b>δ = {d_val:.2f}</b>", arrowhead=1, ay=30 if d_val < 0 else -30, 
            font=dict(color="#27AE60", size=12), bgcolor="white", row=4, col=1
        )

    # --- FINAL LAYOUT & AXIS LABELS ---
    # Update Y-Axes Labels (Explicitly)
    fig.update_yaxes(title_text="Shear V (kN)", row=2, col=1)
    fig.update_yaxes(title_text="Moment M (kNm)", row=3, col=1)
    fig.update_yaxes(title_text="Deflection δ (mm)", row=4, col=1)
    
    # Update X-Axis Label (Only on bottom)
    fig.update_xaxes(title_text="Position along Beam (m)", row=4, col=1)

    fig.update_layout(
        height=900, 
        showlegend=False, 
        template="plotly_white", 
        hovermode="x unified",
        margin=dict(l=60, r=20, t=50, b=50) # เพิ่มขอบซ้าย (l) ให้มีที่วาง Label แกน Y
    )
    
    return fig
