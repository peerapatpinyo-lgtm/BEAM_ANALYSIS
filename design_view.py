import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def draw_interactive_diagrams(df, reac, spans, sup_df, raw_loads, unit_force="kg", unit_len="m"):
    """
    Generate professional engineering diagrams for Beam Analysis.
    """
    # 1. Prepare Data
    total_len = df['x'].max()
    nodes = [0] + list(np.cumsum(spans))
    
    # Create Subplots: 4 Rows (Model, Shear, Moment, Deflection)
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.2, 0.25, 0.25, 0.3],
        subplot_titles=(
            "🏗️ Structural Model & Loads", 
            f"⚡ Shear Force Diagram (V) [{unit_force}]", 
            f"🔄 Bending Moment Diagram (M) [{unit_force}-{unit_len}]", 
            f"📉 Deflection (δ) [mm]" # Assuming deflection converted to mm for display
        )
    )

    # --- ROW 1: BEAM GEOMETRY & LOADS ---
    # Draw Beam Line
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', line=dict(color='black', width=5), hoverinfo='skip'), row=1, col=1)
    
    # Draw Supports
    if not sup_df.empty:
        for _, s in sup_df.iterrows():
            x_pos = nodes[int(s['id'])]
            stype = s['type']
            symbol = "triangle-up" if stype == "Pin" else ("circle" if stype == "Roller" else "square")
            color = "green" if stype == "Fixed" else "black"
            # Support Marker
            fig.add_trace(go.Scatter(
                x=[x_pos], y=[-0.05], mode='markers+text', 
                marker=dict(symbol=symbol, size=15, color=color),
                text=[stype], textposition="bottom center",
                name="Support"
            ), row=1, col=1)

    # Draw Loads (Visualization)
    max_load_mag = 1.0 # Scale factor reference
    if raw_loads:
        max_load_mag = max([l['mag'] for l in raw_loads]) if raw_loads else 1.0
        
        for l in raw_loads:
            # Determine Color (DL=Blue, LL=Red)
            color = "red" if l['case'] == 'LL' else "blue"
            
            if l['type'] == 'P':
                x_abs = nodes[int(l['span_idx'])] + l['x']
                # Draw Arrow using Annotation
                fig.add_annotation(
                    x=x_abs, y=0, ax=0, ay=-40, 
                    xref='x1', yref='y1',
                    arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor=color,
                    text=f"<b>P={l['mag']}</b>", row=1, col=1
                )
            elif l['type'] == 'U':
                start = nodes[int(l['span_idx'])]
                end = nodes[int(l['span_idx'])+1]
                # Draw Rectangle area
                fig.add_shape(
                    type="rect", x0=start, x1=end, y0=0, y1=0.2,
                    line=dict(width=0), fillcolor=color, opacity=0.2,
                    xref='x1', yref='y1', row=1, col=1
                )
                fig.add_annotation(
                    x=(start+end)/2, y=0.1, text=f"<b>w={l['mag']}</b>", 
                    showarrow=False, font=dict(color=color), row=1, col=1
                )

    # --- ROW 2: SHEAR FORCE (SFD) ---
    # Fill area
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'], mode='lines', 
        fill='tozeroy', line=dict(color='#FF9800', width=2), name="Shear (V)"
    ), row=2, col=1)
    
    # Annotate Max/Min Shear
    v_max = df['shear'].max()
    v_min = df['shear'].min()
    # Find x for max/min to place label
    x_v_max = df.loc[df['shear'].idxmax(), 'x']
    x_v_min = df.loc[df['shear'].idxmin(), 'x']
    
    fig.add_annotation(x=x_v_max, y=v_max, text=f"{v_max:.2f}", showarrow=True, arrowhead=1, row=2, col=1)
    fig.add_annotation(x=x_v_min, y=v_min, text=f"{v_min:.2f}", showarrow=True, arrowhead=1, row=2, col=1)

    # --- ROW 3: BENDING MOMENT (BMD) ---
    # Civil Engineering Convention: Tension side? 
    # Here we plot standard mechanics: Sagging (+), Hogging (-)
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['moment'], mode='lines', 
        fill='tozeroy', line=dict(color='#2196F3', width=2), name="Moment (M)"
    ), row=3, col=1)
    
    # Annotate Max Moments
    m_max = df['moment'].max() # Sagging
    m_min = df['moment'].min() # Hogging
    x_m_max = df.loc[df['moment'].idxmax(), 'x']
    x_m_min = df.loc[df['moment'].idxmin(), 'x']
    
    if abs(m_max) > 1e-3:
        fig.add_annotation(x=x_m_max, y=m_max, text=f"Max(+): {m_max:.2f}", showarrow=True, row=3, col=1, font=dict(color='blue'))
    if abs(m_min) > 1e-3:
        fig.add_annotation(x=x_m_min, y=m_min, text=f"Max(-): {m_min:.2f}", showarrow=True, row=3, col=1, font=dict(color='red'))

    # --- ROW 4: DEFLECTION ---
    # Convert m to mm for display (optional but standard)
    defl_mm = df['deflection'] * 1000 
    fig.add_trace(go.Scatter(
        x=df['x'], y=defl_mm, mode='lines', 
        line=dict(color='#4CAF50', width=2, dash='dot'), name="Deflection (mm)"
    ), row=4, col=1)
    
    # Annotate Max Deflection
    d_max = defl_mm.abs().max()
    if d_max > 0:
        idx_d = defl_mm.abs().idxmax()
        fig.add_annotation(
            x=df.loc[idx_d, 'x'], y=defl_mm[idx_d], 
            text=f"Max: {defl_mm[idx_d]:.2f} mm", 
            showarrow=True, row=4, col=1
        )

    # Layout Polish
    fig.update_layout(
        height=1000, 
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    # Fix Y-Axis Ranges (Optional: Add padding)
    fig.update_yaxes(title_text="Load", showticklabels=False, row=1, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans, u_force, u_len):
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📌 Support Reactions")
        if reac is not None:
            nodes = len(reac) // 2
            r_data = []
            for i in range(nodes):
                fy = reac[2*i]
                mz = reac[2*i+1]
                # Filter small numbers
                if abs(fy) < 1e-5: fy = 0
                if abs(mz) < 1e-5: mz = 0
                
                r_data.append({
                    "Node ID": i,
                    f"Reaction Y ({u_force})": f"{fy:,.2f}",
                    f"Reaction M ({u_force}-{u_len})": f"{mz:,.2f}"
                })
            
            st.dataframe(pd.DataFrame(r_data), use_container_width=True, hide_index=True)

    with c2:
        st.subheader("📊 Critical Design Values")
        # Extract per span
        design_vals = []
        cum_dist = [0] + list(np.cumsum(spans))
        
        for i, span_len in enumerate(spans):
            start = cum_dist[i]
            end = cum_dist[i+1]
            # Filter span
            sub_df = df[(df['x'] >= start) & (df['x'] <= end)]
            
            m_pos = sub_df['moment'].max()
            m_neg = sub_df['moment'].min() # This might be at support
            v_max = sub_df['shear'].abs().max()
            
            design_vals.append({
                "Span": f"Span {i+1}",
                "Length": f"{span_len} m",
                f"+M_max": f"{m_pos:.2f}",
                f"-M_max": f"{m_neg:.2f}",
                f"V_max": f"{v_max:.2f}"
            })
        
        st.dataframe(pd.DataFrame(design_vals), use_container_width=True, hide_index=True)
