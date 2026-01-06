import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_analysis_results(df, spans):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        subplot_titles=("Shear Force (N)", "Bending Moment (N-m)", "Deflection (m)"))
    
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', name="Shear"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', name="Moment"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['deflection'], name="Deflection"), row=3, col=1)
    
    fig.update_layout(height=700, showlegend=False)
    return fig
