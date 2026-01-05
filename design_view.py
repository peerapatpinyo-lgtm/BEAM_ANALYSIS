import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def draw_interactive_diagrams(df, reac, spans, sup_df, raw_loads):
    
    nodes = [0] + list(np.cumsum(spans))
    total_len = nodes[-1]
    
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.25, 0.25, 0.25, 0.25],
        subplot_titles=("🏗️ Model & Loads", "⚡ Shear Force (V)", "🔄 Bending Moment (M)", "📉 Deflection (δ)")
    )

    # ==========================
    # 1. VISUALIZATION (Real Beam)
    # ==========================
    # Draw Beam as a Thick Rectangle
    beam_h = 0.05 * total_len # Visual thickness relative to length
    beam_h = max(0.2, min(beam_h, 0.5)) # Clamp visual size
    
    fig.add_shape(type="rect", 
        x0=0, x1=total_len, y0=-beam_h/2, y1=beam_h/2,
        fillcolor="#e0e0e0", line=dict(color="black", width=2),
        row=1, col=1
    )

    # Draw Supports (Improved Shapes)
    if not sup_df.empty:
        for _, s in sup_df.iterrows():
            x = nodes[int(s['id'])]
            stype = s['type']
            
            if stype == "Pin":
                # Triangle
                fig.add_trace(go.Scatter(
                    x=[x], y=[-beam_h/2], mode="markers",
                    marker=dict(symbol="triangle-up", size=15, color="black"),
                    hoverinfo="text", text="Pin", showlegend=False
                ), row=1, col=1)
            elif stype == "Roller":
                # Triangle + Circles
                fig.add_trace(go.Scatter(
                    x=[x], y=[-beam_h/2], mode="markers",
                    marker=dict(symbol="triangle-up", size=15, color="black"),
                    showlegend=False
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=[x-0.1, x+0.1], y=[-beam_h/2 - 0.15], mode="markers",
                    marker=dict(symbol="circle", size=6, color="black"),
                    showlegend=False
                ), row=1, col=1)
            elif stype == "Fixed":
                # Vertical Line/Rect
                fig.add_shape(type="rect", 
                    x0=x-0.05, x1=x+0.05, y0=-beam_h, y1=beam_h,
                    fillcolor="black", line_width=0, row=1, col=1
                )
                # Hatching (Simulated with lines)
                for h in np.linspace(-beam_h, beam_h, 5):
                    fig.add_shape(type="line", x0=x, y0=h, x1=x-0.2, y1=h-0.1, line=dict(width=1), row=1, col=1)

    # Draw Loads (The "Pro" Logic)
    max_mag = max([l['mag'] for l in raw_loads]) if raw_loads else 100
    
    for l in raw_loads:
        color = "#FF5252" if l['case'] == 'LL' else "#448AFF" # Red/Blue
        
        if l['type'] == 'P':
            # Point Load: Single Arrow
            x_loc = nodes[int(l['span_idx'])] + l['x']
            arrow_len = 1.0 + (l['mag']/max_mag)*1.0 # Scale visual length
            
            fig.add_annotation(
                x=x_loc, y=beam_h/2, ax=0, ay=-50, # Fixed pixel length for arrow is better for UX
                xref="x1", yref="y1",
                arrowhead=2, arrowwidth=2, arrowcolor=color,
                text=f"<b>P={l['mag']:.0f}</b>", font=dict(color=color),
                bgcolor="rgba(255,255,255,0.7)",
                row=1, col=1
            )
            
        elif l['type'] == 'U':
            # Uniform Load: "Comb" Style (Many arrows connected by a bar)
            start_x = nodes[int(l['span_idx'])] + l['x']
            # Handle user end input, default to full span if not in dict
            span_len = spans[int(l['span_idx'])]
            end_x = nodes[int(l['span_idx'])] + l.get('end', span_len) 
            
            # The "Lid" line (Horizontal bar)
            load_h = 1.5 # Visual height of load
            fig.add_shape(type="line", 
                x0=start_x, x1=end_x, y0=beam_h/2 + load_h, y1=beam_h/2 + load_h,
                line=dict(color=color, width=2), row=1, col=1
            )
            
            # Draw multiple vertical arrows
            num_arrows = max(3, int((end_x - start_x) * 2)) # Approx 2 arrows per meter
            for ax in np.linspace(start_x, end_x, num_arrows):
                fig.add_annotation(
                    x=ax, y=beam_h/2, # Tip touches beam
                    ax=0, ay=-40, # Tail is up (pixels)
                    xref="x1", yref="y1",
                    arrowhead=2, arrowwidth=1.5, arrowcolor=color,
                    showarrow=True, row=1, col=1
                )
                
            # Label
            fig.add_annotation(
                x=(start_x+end_x)/2, y=beam_h/2 + load_h,
                text=f"<b>w={l['mag']:.0f}</b>",
                yshift=15, showarrow=False, font=dict(color=color),
                row=1, col=1
            )

    # ==========================
    # 2. SHEAR (Step Line)
    # ==========================
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'], mode='lines', line_shape='hv', # hv = Horizontal-Vertical step
        fill='tozeroy', line=dict(color='#FFA726', width=2), name="Shear"
    ), row=2, col=1)
    
    # ==========================
    # 3. MOMENT (Curved)
    # ==========================
    # Invert Moment for Civil Convention? (Positive Down) -> Let's stick to standard mechanics (Positive Up)
    # But highlight Tension zone
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['moment'], mode='lines',
        fill='tozeroy', line=dict(color='#29B6F6', width=2), name="Moment"
    ), row=3, col=1)
    
    # ==========================
    # 4. DEFLECTION
    # ==========================
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['deflection']*1000, mode='lines',
        line=dict(color='#66BB6A', width=3, dash='dot'), name="Defl"
    ), row=4, col=1)

    # Formatting
    fig.update_layout(height=1000, template="plotly_white", showlegend=False, hovermode="x unified")
    fig.update_yaxes(showticklabels=False, title="Load", row=1, col=1)
    fig.update_yaxes(title="V (kg)", row=2, col=1)
    fig.update_yaxes(title="M (kg-m)", row=3, col=1)
    fig.update_yaxes(title="δ (mm)", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans, u_force, u_len):
    st.markdown("#### 📊 Analysis Results")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.write("**Reactions:**")
        if reac is not None:
            r_data = []
            for i in range(len(reac)//2):
                r_data.append({"Node": i, "Ry": f"{reac[2*i]:.2f}", "M": f"{reac[2*i+1]:.2f}"})
            st.dataframe(pd.DataFrame(r_data), hide_index=True)
            
    with c2:
        st.write("**Max Forces per Span:**")
        s_data = []
        nodes = [0] + list(np.cumsum(spans))
        for i in range(len(spans)):
            sub = df[(df['x'] >= nodes[i]) & (df['x'] <= nodes[i+1])]
            s_data.append({
                "Span": i+1,
                "V_max": f"{sub['shear'].abs().max():.2f}",
                "+M_max": f"{sub['moment'].max():.2f}",
                "-M_max": f"{sub['moment'].min():.2f}",
                "δ_max (mm)": f"{sub['deflection'].abs().max()*1000:.2f}"
            })
        st.dataframe(pd.DataFrame(s_data), hide_index=True)
