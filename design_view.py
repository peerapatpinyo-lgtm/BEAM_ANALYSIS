import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def plot_capacity_vs_demand(df_span, phi_Mn_pos, phi_Mn_neg):
    """
    สร้างกราฟเปรียบเทียบ Mu vs PhiMn (Capacity Check)
    """
    fig = go.Figure()
    
    # 1. Plot Mu (Demand) - Solid Line
    fig.add_trace(go.Scatter(
        x=df_span['x'], 
        y=df_span['moment'] / 1000, # Convert Nm -> kNm
        mode='lines',
        name='Applied Moment (Mu)',
        line=dict(color='#2980B9', width=3),
        fill='tozeroy',
        fillcolor='rgba(41, 128, 185, 0.1)'
    ))
    
    # 2. Plot +PhiMn (Positive Capacity) - Dashed Green
    # สร้างเส้น Capacity ตลอดช่วงคาน
    x_range = [df_span['x'].min(), df_span['x'].max()]
    fig.add_trace(go.Scatter(
        x=x_range, 
        y=[phi_Mn_pos, phi_Mn_pos],
        mode='lines',
        name=f'Capacity +M (φMn = {phi_Mn_pos:.1f})',
        line=dict(color='#27AE60', width=2, dash='dash')
    ))
    
    # 3. Plot -PhiMn (Negative Capacity) - Dashed Red
    # ต้องเป็นค่าลบในกราฟ เพื่อเทียบกับ Moment ลบ
    neg_cap = -abs(phi_Mn_neg)
    fig.add_trace(go.Scatter(
        x=x_range, 
        y=[neg_cap, neg_cap],
        mode='lines',
        name=f'Capacity -M (φMn = {abs(neg_cap):.1f})',
        line=dict(color='#C0392B', width=2, dash='dash')
    ))

    # Layout
    fig.update_layout(
        title="Moment Capacity Check (Mu vs φMn)",
        xaxis_title="Position (m)",
        yaxis_title="Moment (kNm)",
        height=350,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    # Add Shading for unsafe zones (Optional Logic)
    # ถ้าเส้น Mu ทะลุเส้น Capacity (Unsafe)
    
    return fig

def draw_interactive_diagrams(df, reac, spans, sup_df, loads, dl_factor=1.4, ll_factor=1.7):
    # (โค้ดเดิมส่วนใหญ่... แต่เพื่อความชัวร์ Copy ทั้งก้อนนี้ไปแทนที่ของเดิมครับ)
    
    # Data Sanitization
    if isinstance(spans, (pd.DataFrame, pd.Series)): spans_val = spans.values.flatten().tolist()
    elif isinstance(spans, list): spans_val = spans
    else: spans_val = []

    cum_spans = [0] + list(np.cumsum(spans_val))
    total_len = cum_spans[-1] if cum_spans else 0
    
    # Create Loads for visualization
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

    # Fix Column Names
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # Auto-Scale Deflection
    max_defl = df['deflection'].abs().max() if not df['deflection'].empty else 0
    defl_unit = "m"
    defl_scale = 1.0
    if max_defl > 0 and max_defl < 0.01:
        defl_unit = "mm"
        defl_scale = 1000.0
    df['deflection_plot'] = df['deflection'] * defl_scale

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Structure Model", "Shear Force (Vu)", "Bending Moment (Mu)", f"Deflection ({defl_unit})"),
        row_heights=[0.15, 0.25, 0.25, 0.35]
    )

    # 1. Structure
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], line=dict(color='black', width=4), showlegend=False), row=1, col=1)
    # Supports
    for node_idx, val in reac.items():
        if isinstance(node_idx, int) and node_idx < len(cum_spans):
            x = cum_spans[node_idx]
            fig.add_trace(go.Scatter(x=[x], y=[-0.1], mode='markers', marker=dict(symbol='triangle-up', size=12, color='black'), showlegend=False), row=1, col=1)

    # Loads
    for l in clean_loads:
        color = "#E74C3C" if l['case'] == 'LL' else "#555555"
        if l['type'] == 'P':
            fig.add_annotation(x=l['global_x'], y=0, ax=0, ay=-40, arrowhead=2, text=f"{l['mag']}", font=dict(color=color, size=9), row=1, col=1)
        elif l['type'] == 'U':
            x_end = l['global_x'] + l['dist']
            fig.add_shape(type="rect", x0=l['global_x'], x1=x_end, y0=0.05, y1=0.15, fillcolor=color, opacity=0.3, line_width=0, row=1, col=1)

    # 2. Shear
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#E67E22'), name="Shear"), row=2, col=1)
    
    # 3. Moment (Inverted)
    fig.add_trace(go.Scatter(x=df['x'], y=-df['moment'], fill='tozeroy', line=dict(color='#2980B9'), name="Moment"), row=3, col=1)
    
    # 4. Deflection
    fig.add_trace(go.Scatter(x=df['x'], y=df['deflection_plot'], fill='tozeroy', line=dict(color='#27AE60'), name="Deflection"), row=4, col=1)

    fig.update_layout(height=800, showlegend=False, template="plotly_white")
    return fig
