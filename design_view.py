import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def draw_interactive_diagrams(df, reac, spans, sup_df, raw_loads):
    """
    Professional Structural Engineering Visualization
    """
    # Nodes calculation
    nodes = [0] + list(np.cumsum(spans))
    total_len = nodes[-1]
    
    # Setup Subplots
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.06,
        row_heights=[0.20, 0.25, 0.25, 0.30],
        subplot_titles=(
            "🏗️ Structural Model & Loading", 
            "⚡ Shear Force Diagram (SFD)", 
            "🔄 Bending Moment Diagram (BMD)", 
            "📉 Elastic Deflection"
        )
    )

    # ==========================================
    # 1. STRUCTURAL MODEL (The "Real" Beam)
    # ==========================================
    # Beam Line
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0], 
        mode='lines', line=dict(color='black', width=6), 
        hoverinfo='skip'
    ), row=1, col=1)

    # Draw Supports (Shapes)
    if not sup_df.empty:
        for _, s in sup_df.iterrows():
            x_pos = nodes[int(s['id'])]
            stype = s['type']
            
            # Draw Triangle for Pin/Roller
            if stype in ["Pin", "Roller"]:
                fig.add_trace(go.Scatter(
                    x=[x_pos], y=[-0.05],
                    mode='markers+text',
                    marker=dict(symbol="triangle-up", size=18, color="black"),
                    text=stype, textposition="bottom center",
                    showlegend=False
                ), row=1, col=1)
                # If Roller, add wheels
                if stype == "Roller":
                    fig.add_trace(go.Scatter(
                        x=[x_pos-0.1, x_pos+0.1], y=[-0.12, -0.12],
                        mode='markers', marker=dict(size=5, color='black'),
                        showlegend=False
                    ), row=1, col=1)
            elif stype == "Fixed":
                fig.add_shape(
                    type="rect", x0=x_pos-0.1, x1=x_pos+0.1, y0=-0.15, y1=0.15,
                    fillcolor="black", line=dict(width=0),
                    row=1, col=1
                )
                fig.add_annotation(x=x_pos, y=-0.2, text="Fixed", showarrow=False, row=1, col=1)

    # Draw Loads (Professional Arrows)
    max_load = 100 # Default scale
    if raw_loads:
        max_load = max([l['mag'] for l in raw_loads]) if raw_loads else 100
        
    for l in raw_loads:
        x_start = nodes[int(l['span_idx'])] + l['x']
        color = "#d32f2f" if l['case'] == 'LL' else "#1976d2" # Red for LL, Blue for DL
        
        if l['type'] == 'P':
            # Arrow pointing down
            fig.add_annotation(
                x=x_start, y=0, ax=0, ay=-50,
                xref=f"x1", yref=f"y1",
                arrowhead=2, arrowwidth=2, arrowcolor=color,
                text=f"P={l['mag']}", font=dict(color=color),
                row=1, col=1
            )
        elif l['type'] == 'U':
            x_end = nodes[int(l['span_idx']) + 1]
            # Draw Distributed Block
            fig.add_shape(
                type="rect", x0=x_start, x1=x_end, y0=0, y1=0.25,
                line=dict(width=0), fillcolor=color, opacity=0.2,
                row=1, col=1
            )
            # Add Arrows inside block
            for xa in np.linspace(x_start, x_end, 5):
                fig.add_annotation(
                    x=xa, y=0, ax=0, ay=-20,
                    arrowhead=3, arrowwidth=1, arrowcolor=color,
                    row=1, col=1
                )
            fig.add_annotation(
                x=(x_start+x_end)/2, y=0.3, 
                text=f"w={l['mag']}", showarrow=False, font=dict(color=color),
                row=1, col=1
            )

    # ==========================================
    # 2. SHEAR FORCE DIAGRAM (SFD)
    # ==========================================
    # Use 'hv' line shape for Step function (Correct engineering viz)
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'], 
        mode='lines', line_shape='hv', 
        fill='tozeroy', line=dict(color='#FF9800', width=2),
        name="Shear"
    ), row=2, col=1)
    
    # Annotate Max/Min
    v_max = df['shear'].max()
    v_min = df['shear'].min()
    fig.add_annotation(x=df.loc[df['shear'].idxmax(), 'x'], y=v_max, text=f"{v_max:.0f}", showarrow=True, row=2, col=1)
    fig.add_annotation(x=df.loc[df['shear'].idxmin(), 'x'], y=v_min, text=f"{v_min:.0f}", showarrow=True, row=2, col=1)

    # ==========================================
    # 3. BENDING MOMENT DIAGRAM (BMD)
    # ==========================================
    # Separate Positive and Negative for coloring
    df['m_pos'] = df['moment'].apply(lambda x: x if x >= 0 else 0)
    df['m_neg'] = df['moment'].apply(lambda x: x if x < 0 else 0)

    # Positive Moment (Blue)
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['m_pos'], 
        mode='lines', fill='tozeroy', 
        line=dict(color='#2196F3', width=0), fillcolor='rgba(33, 150, 243, 0.3)',
        name="+ Moment"
    ), row=3, col=1)
    
    # Negative Moment (Red)
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['m_neg'], 
        mode='lines', fill='tozeroy', 
        line=dict(color='#E91E63', width=0), fillcolor='rgba(233, 30, 99, 0.3)',
        name="- Moment"
    ), row=3, col=1)
    
    # Main Line
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['moment'], 
        mode='lines', line=dict(color='#333333', width=2),
        showlegend=False
    ), row=3, col=1)

    # Annotations for BMD
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    if abs(m_max) > 1:
        fig.add_annotation(x=df.loc[df['moment'].idxmax(), 'x'], y=m_max, text=f"Max (+): {m_max:.1f}", showarrow=True, row=3, col=1)
    if abs(m_min) > 1:
        fig.add_annotation(x=df.loc[df['moment'].idxmin(), 'x'], y=m_min, text=f"Max (-): {m_min:.1f}", showarrow=True, row=3, col=1)

    # ==========================================
    # 4. DEFLECTION
    # ==========================================
    # Convert to mm
    defl_mm = df['deflection'] * 1000
    fig.add_trace(go.Scatter(
        x=df['x'], y=defl_mm, 
        mode='lines', line=dict(color='#4CAF50', width=3, dash='dash'),
        name="Deflection (mm)"
    ), row=4, col=1)
    
    # Max Deflection
    d_max_idx = defl_mm.abs().idxmax()
    d_max = defl_mm[d_max_idx]
    fig.add_annotation(
        x=df.loc[d_max_idx, 'x'], y=d_max, 
        text=f"δ_max: {d_max:.2f} mm", 
        showarrow=True, row=4, col=1
    )

    # Global Layout Styling
    fig.update_layout(
        height=1100,
        title_text="<b>Analysis Results</b>",
        hovermode="x unified",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Axis Titles
    fig.update_yaxes(title_text="Load", showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="Shear (kg)", row=2, col=1)
    fig.update_yaxes(title_text="Moment (kg-m)", row=3, col=1)
    fig.update_yaxes(title_text="Deflection (mm)", row=4, col=1)
    fig.update_xaxes(title_text="Distance (m)", row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans, u_force, u_len):
    st.markdown("### 📊 Engineering Summary Report")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📍 Support Reactions")
        if reac is not None:
            r_data = []
            for i in range(len(reac)//2):
                fy = reac[2*i]
                mz = reac[2*i+1]
                if abs(fy) > 1e-3 or abs(mz) > 1e-3:
                    r_data.append({
                        "Node": f"Node {i}",
                        "Ry (kg)": f"{fy:,.2f}",
                        "M (kg-m)": f"{mz:,.2f}"
                    })
            st.dataframe(pd.DataFrame(r_data), hide_index=True, use_container_width=True)
            
    with col2:
        st.markdown("#### 📐 Span Design Forces")
        design_data = []
        cum_dist = [0] + list(np.cumsum(spans))
        
        for i, L in enumerate(spans):
            start, end = cum_dist[i], cum_dist[i+1]
            sub = df[(df['x'] >= start) & (df['x'] <= end)]
            
            design_data.append({
                "Span": f"Span {i+1} ({L}m)",
                "V_max (kg)": f"{sub['shear'].abs().max():,.2f}",
                "+M_max (kg-m)": f"{sub['moment'].max():,.2f}",
                "-M_max (kg-m)": f"{sub['moment'].min():,.2f}",
                "Defl (mm)": f"{sub['deflection'].abs().max()*1000:.2f}"
            })
        st.dataframe(pd.DataFrame(design_data), hide_index=True, use_container_width=True)
