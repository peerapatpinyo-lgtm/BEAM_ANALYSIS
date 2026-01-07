import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_analysis_results(res, spans, sup_input, loads, reactions):
    """
    Generate Professional Beam Analysis Diagrams
    Input:
        res: DataFrame containing ['x', 'shear', 'moment', 'deflection']
        spans: List of span lengths
        sup_input: List of supports [{'id':0, 'type':'Pin'}, ...]
        loads: List of loads
        reactions: Dict or List of reactions {node_id: value}
    """
    
    # 1. Prepare Data Structure
    cum_dist = [0] + list(np.cumsum(spans))
    total_len = cum_dist[-1]
    
    # Handle Reactions Input format
    reac_map = {}
    if isinstance(reactions, list):
        for r in reactions:
            reac_map[int(r['node_id'])] = r['fy']
    elif isinstance(reactions, dict):
        reac_map = {int(k): v for k, v in reactions.items()}

    # 2. Setup Plot Layout
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1,
        subplot_titles=(
            "1. Free Body Diagram & Loads", 
            "2. Shear Force Diagram (SFD)", 
            "3. Bending Moment Diagram (BMD)"
        ),
        row_heights=[0.25, 0.35, 0.40]
    )

    # ==========================================
    # ROW 1: Free Body Diagram (FBD)
    # ==========================================
    
    # Beam Line
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=5), hoverinfo='skip'), row=1, col=1)

    # Draw Supports & Reactions
    for s in sup_input:
        node_idx = int(s.get('id', 0))
        if node_idx < len(cum_dist):
            x_s = cum_dist[node_idx]
            
            # Support Symbol (Triangle)
            fig.add_trace(go.Scatter(
                x=[x_s], y=[-0.2], 
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=15, color='#2c3e50'),
                text=[s.get('type', 'Sup')], textposition="bottom center",
                showlegend=False, hoverinfo='text'
            ), row=1, col=1)

            # Reaction Arrow & Label
            r_val = reac_map.get(node_idx, 0)
            if abs(r_val) > 1: # Show only significant reactions
                r_kNm = r_val / 1000.0
                fig.add_annotation(
                    x=x_s, y=-0.5,
                    text=f"R={r_kNm:.2f} kN",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                    ax=0, ay=40, arrowcolor='green',
                    font=dict(color='green', size=12, weight='bold'),
                    row=1, col=1
                )

    # Draw Loads
    for l in loads:
        x_pos = l['x']
        mag = l['mag'] / 1000.0
        
        if l['type'] == 'P':
            fig.add_annotation(
                x=x_pos, y=0.5,
                text=f"{mag:.2f} kN",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                ax=0, ay=-40, arrowcolor='red',
                row=1, col=1
            )
        elif l['type'] == 'U':
            dist = l.get('dist', 0)
            x_end = x_pos + dist
            # Draw UDL Line
            fig.add_trace(go.Scatter(
                x=[x_pos, x_end], y=[0.5, 0.5], 
                mode='lines', line=dict(color='orange', width=2), showlegend=False
            ), row=1, col=1)
            fig.add_annotation(
                x=(x_pos + x_end)/2, y=0.8,
                text=f"w = {mag:.2f} kN/m",
                showarrow=False, font=dict(color='orange'),
                row=1, col=1
            )
            # Add small arrows for UDL
            for xa in np.linspace(x_pos, x_end, num=max(3, int(dist)+1)):
                fig.add_annotation(x=xa, y=0.1, ax=0, ay=-30, 
                                   arrowhead=1, arrowcolor='orange', row=1, col=1)

    # ==========================================
    # ROW 2: Shear Force Diagram (SFD)
    # ==========================================
    y_shear = res['shear'] / 1000.0
    fig.add_trace(go.Scatter(
        x=res['x'], y=y_shear, 
        mode='lines', fill='tozeroy', 
        line=dict(color='#e74c3c', width=2), 
        name='Shear (kN)', fillcolor='rgba(231, 76, 60, 0.3)'
    ), row=2, col=1)

    # Add Annotations for Max/Min Shear
    v_max = y_shear.max()
    v_min = y_shear.min()
    # Find x positions for labels
    idx_max = y_shear.idxmax()
    idx_min = y_shear.idxmin()
    
    fig.add_annotation(x=res['x'][idx_max], y=v_max, text=f"{v_max:.2f}", showarrow=False, yshift=10, row=2, col=1)
    fig.add_annotation(x=res['x'][idx_min], y=v_min, text=f"{v_min:.2f}", showarrow=False, yshift=-10, row=2, col=1)

    # ==========================================
    # ROW 3: Bending Moment Diagram (BMD)
    # ==========================================
    y_moment = res['moment'] / 1000.0
    fig.add_trace(go.Scatter(
        x=res['x'], y=y_moment, 
        mode='lines', fill='tozeroy', 
        line=dict(color='#3498db', width=2), 
        name='Moment (kNm)', fillcolor='rgba(52, 152, 219, 0.3)'
    ), row=3, col=1)
    
    # Invert Y axis for Moment (Thai/Civil Standard)
    fig.update_yaxes(autorange="reversed", row=3, col=1)

    # Add Annotations for Max Moment
    # Filter local peaks for better labelling
    m_max = y_moment.max()
    m_min = y_moment.min() # Negative moment (Top steel)
    
    if abs(m_max) > 0.1:
        idx_m_max = y_moment.idxmax()
        fig.add_annotation(x=res['x'][idx_m_max], y=m_max, text=f"{m_max:.2f}", 
                           showarrow=True, arrowhead=1, ay=30, row=3, col=1)
    
    if abs(m_min) > 0.1:
        idx_m_min = y_moment.idxmin()
        fig.add_annotation(x=res['x'][idx_m_min], y=m_min, text=f"{m_min:.2f}", 
                           showarrow=True, arrowhead=1, ay=-30, row=3, col=1)

    # ==========================================
    # FINAL FORMATTING
    # ==========================================
    # Zero Lines
    fig.add_hline(y=0, line_width=2, line_color="black", opacity=0.5, row=2, col=1)
    fig.add_hline(y=0, line_width=2, line_color="black", opacity=0.5, row=3, col=1)

    fig.update_layout(
        height=900, 
        showlegend=False, 
        plot_bgcolor='white',
        margin=dict(l=50, r=20, t=60, b=50),
        hovermode="x unified"
    )
    
    # Hide y-axis grid for FBD, Show for others
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1, range=[-1, 1.5])
    fig.update_yaxes(showgrid=True, gridcolor='#eee', title_text="Shear (kN)", row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='#eee', title_text="Moment (kNm)", row=3, col=1)

    return fig
