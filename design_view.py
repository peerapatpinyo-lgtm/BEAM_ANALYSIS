import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def draw_interactive_diagrams(df, reac, spans, sup_df, raw_loads, unit_force="kg", unit_len="m", dl_factor=1.4, ll_factor=1.7):
    
    # Setup Figure
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=("Structure & Loads", "Shear Force (V)", "Bending Moment (M)", "Deflection (delta)"),
        row_heights=[0.2, 0.26, 0.26, 0.28]
    )
    
    total_len = df['x'].max()
    nodes = [0] + list(np.cumsum(spans))
    
    # 1. Structure (Row 1)
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], line=dict(color='black', width=4), showlegend=False), row=1, col=1)
    
    # Supports
    if not sup_df.empty:
        for _, s in sup_df.iterrows():
            x_pos = nodes[int(s['id'])]
            symbol = "triangle-up" if s['type'] != "Fixed" else "square"
            fig.add_trace(go.Scatter(x=[x_pos], y=[-0.1], mode='markers', marker=dict(size=15, symbol=symbol, color='black'), showlegend=False), row=1, col=1)
            
    # Loads (Visualization of INPUTS)
    if raw_loads:
        for l in raw_loads:
            x_abs = nodes[l['span_idx']] + l['x']
            color = "red" if l['case'] == 'LL' else "gray"
            if l['type'] == 'P':
                fig.add_annotation(x=x_abs, y=0, ax=0, ay=-40, arrowhead=2, arrowcolor=color, text=f"P={l['mag']}", row=1, col=1)
            elif l['type'] == 'U':
                xs = nodes[l['span_idx']]
                xe = nodes[l['span_idx']+1]
                fig.add_shape(type="rect", x0=xs, x1=xe, y0=0.1, y1=0.3, fillcolor=color, opacity=0.3, line_width=0, row=1, col=1)
                fig.add_annotation(x=(xs+xe)/2, y=0.3, text=f"w={l['mag']}", showarrow=False, row=1, col=1)

    # 2. Shear (Row 2)
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='orange'), name="Shear"), row=2, col=1)
    
    # 3. Moment (Row 3)
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='blue'), name="Moment"), row=3, col=1)
    
    # 4. Deflection (Row 4)
    fig.add_trace(go.Scatter(x=df['x'], y=df['deflection'], line=dict(color='green'), name="Deflection"), row=4, col=1)
    
    # Formatting
    fig.update_layout(height=900, hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans, u_force, u_len):
    st.subheader("Reaction Forces")
    if reac is not None:
        nodes = len(reac) // 2
        r_data = []
        for i in range(nodes):
            r_data.append({
                "Node": i,
                f"Ry ({u_force})": f"{reac[2*i]:.2f}",
                f"M ({u_force}-{u_len})": f"{reac[2*i+1]:.2f}"
            })
        st.table(pd.DataFrame(r_data))
