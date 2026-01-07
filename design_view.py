import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_analysis_results(df, spans):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        subplot_titles=("Shear Force Diagram (SFD)", "Bending Moment Diagram (BMD)", "Deflection"),
                        vertical_spacing=0.1)
    
    # Shear (kN)
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear']/1000, fill='tozeroy', 
                             name="Shear (kN)", line=dict(color='#E74C3C')), row=1, col=1)
    
    # Moment (kNm) - Invert Y for Engineering Convention
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment']/1000, fill='tozeroy', 
                             name="Moment (kNm)", line=dict(color='#2980B9')), row=2, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)

    # Deflection (mm)
    fig.add_trace(go.Scatter(x=df['x'], y=df['deflection']*1000, 
                             name="Deflection (mm)", line=dict(color='#27AE60')), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=False, hovermode="x unified")
    
    # Add vertical lines for supports
    cum = 0
    for s in spans:
        cum += s
        for i in range(3):
            fig.add_vline(x=cum, line_dash="dot", line_color="gray", row=i+1, col=1)
            
    return fig
