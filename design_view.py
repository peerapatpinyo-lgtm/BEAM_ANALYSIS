import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_analysis_results(res_df, spans, supports_df, load_list_raw):
    if res_df.empty: return go.Figure()
    cum_dist = [0] + list(np.cumsum(spans))
    total_len = cum_dist[-1]

    # Clean loads for viz
    clean_loads, max_mag = [], 0.1
    if load_list_raw:
        for l in load_list_raw:
            try:
                mag_kn = float(l['mag']) / 1000.0
                if abs(mag_kn) > max_mag: max_mag = abs(mag_kn)
                clean_loads.append({
                    'mag': mag_kn, 'abs': abs(mag_kn),
                    'gx': cum_dist[int(l['span_index'])] + float(l['x']),
                    'type': l.get('type','P'), 'case': l.get('case','DL'),
                    'dist': float(l.get('dist',0))
                })
            except: continue
    
    scale = 0.6 / max_mag if max_mag > 0 else 1

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Structure Model", "Shear Force (SFD)", "Bending Moment (BMD)", "Deflection"),
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )

    # Row 1: Structure
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', line=dict(color='black', width=4)), row=1, col=1)
    
    # Supports
    for _, s in supports_df.iterrows():
        if s['type'] != 'None':
            sym, col = {'Pin':('triangle-up','#2C3E50'), 'Roller':('circle','#27AE60'), 'Fixed':('square','#C0392B')}.get(s['type'], ('circle','gray'))
            fig.add_trace(go.Scatter(x=[cum_dist[int(s['id'])]], y=[-0.15], mode='markers+text', marker=dict(symbol=sym, size=15, color=col), text=[s['type']], textposition="bottom center"), row=1, col=1)

    # Loads
    for l in clean_loads:
        col = "#C0392B" if l['case'] == 'LL' else "#5D6D7E"
        h = l['abs'] * scale
        if l['type'] == 'P':
            fig.add_trace(go.Scatter(x=[l['gx'], l['gx']], y=[h, 0], mode='lines+markers', marker=dict(symbol='arrow-down', size=10), line=dict(color=col, width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=[l['gx']], y=[h+0.1], mode='text', text=[f"{l['mag']:.1f}"], textfont=dict(color=col, size=10)), row=1, col=1)
        elif l['type'] == 'U':
            fig.add_trace(go.Scatter(x=[l['gx'], l['gx'], l['gx']+l['dist'], l['gx']+l['dist']], y=[0, h, h, 0], fill='toself', line=dict(width=0), fillcolor=col, opacity=0.4), row=1, col=1)
            fig.add_trace(go.Scatter(x=[l['gx'] + l['dist']/2], y=[h+0.1], mode='text', text=[f"{l['mag']:.1f}"], textfont=dict(color=col, size=10)), row=1, col=1)

    # Row 2,3,4: Diagrams
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['shear']/1000, fill='tozeroy', line=dict(color='#27AE60'), name="V"), row=2, col=1)
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['moment']/1000, fill='tozeroy', line=dict(color='#C0392B'), name="M"), row=3, col=1)
    fig.update_yaxes(autorange="reversed", row=3, col=1)
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['deflection']*1000, line=dict(color='#2980B9'), name="Def"), row=4, col=1)

    # Layout
    for x in cum_dist: 
        for r in range(1,5): fig.add_vline(x=x, line_dash="dash", line_color="gray", opacity=0.3, row=r, col=1)
    
    fig.update_layout(height=800, showlegend=False, hovermode="x unified", margin=dict(l=50, r=20, t=40, b=40), plot_bgcolor="white")
    fig.update_yaxes(showgrid=False, showticklabels=False, row=1, col=1, range=[-0.5, 1.0])
    return fig
