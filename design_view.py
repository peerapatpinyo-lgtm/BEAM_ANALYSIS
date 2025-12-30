import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- THEME CONSTANTS ---
C_BEAM = 'black'
C_PIN_ROLLER = 'white' # ไส้ในสีขาว
C_OUTLINE = 'black'    # ขอบสีดำ
C_SHEAR = '#D97706'    # Amber
C_MOMENT = '#2563EB'   # Blue

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    n_nodes = len(cum_spans)
    
    # Create Subplots
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.15, 
        row_heights=[0.3, 0.35, 0.35],
        subplot_titles=("", "", "") 
    )

    # ==========================================
    # ROW 1: STRUCTURAL MODEL (FBD)
    # ==========================================
    
    # 1.1 Beam Line
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0],
        mode='lines', line=dict(color=C_BEAM, width=5),
        hoverinfo='skip'
    ), row=1, col=1)

    # 1.2 Nodes & Supports Logic
    sup_map = {int(r['id']): r['type'] for _, r in sup_df.iterrows()}

    for i in range(n_nodes):
        x = cum_spans[i]
        
        if i in sup_map:
            # === HAS SUPPORT ===
            sType = sup_map[i]
            
            if sType == 'Pin':
                # Triangle
                fig.add_trace(go.Scatter(
                    x=[x], y=[-0.12], 
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=20, color=C_PIN_ROLLER, line=dict(width=2, color=C_OUTLINE)),
                    hoverinfo='name', name=f"Pin @ Node {i+1}"
                ), row=1, col=1)
                # Ground Line & Hatch
                fig.add_shape(type="line", x0=x-0.2, y0=-0.22, x1=x+0.2, y1=-0.22, line=dict(color='black', width=2), row=1, col=1)
                for hx in np.linspace(x-0.2, x+0.2, 5):
                    fig.add_shape(type="line", x0=hx, y0=-0.22, x1=hx-0.05, y1=-0.28, line=dict(color='black', width=1), row=1, col=1)

            elif sType == 'Roller':
                # Circle
                fig.add_trace(go.Scatter(
                    x=[x], y=[-0.12],
                    mode='markers',
                    marker=dict(symbol='circle', size=18, color=C_PIN_ROLLER, line=dict(width=2, color=C_OUTLINE)),
                    hoverinfo='name', name=f"Roller @ Node {i+1}"
                ), row=1, col=1)
                # Ground Line & Hatch
                fig.add_shape(type="line", x0=x-0.2, y0=-0.22, x1=x+0.2, y1=-0.22, line=dict(color='black', width=2), row=1, col=1)
                for hx in np.linspace(x-0.2, x+0.2, 5):
                    fig.add_shape(type="line", x0=hx, y0=-0.22, x1=hx-0.05, y1=-0.28, line=dict(color='black', width=1), row=1, col=1)

            elif sType == 'Fixed':
                # Vertical Line & Wall Hatch
                fig.add_shape(type="line", x0=x, y0=-0.3, x1=x, y1=0.3, line=dict(color='black', width=4), row=1, col=1)
                h_dir = -0.15 if x == 0 else 0.15 
                for hy in np.linspace(-0.3, 0.3, 7):
                    fig.add_shape(type="line", x0=x, y0=hy, x1=x+h_dir, y1=hy-0.05, line=dict(color='black', width=1), row=1, col=1)

        else:
            # === NO SUPPORT (Internal Node) ===
            fig.add_trace(go.Scatter(
                x=[x], y=[0],
                mode='markers',
                marker=dict(symbol='circle', size=10, color='white', line=dict(width=2, color='black')),
                hoverinfo='name', name=f"Node {i+1}"
            ), row=1, col=1)

    # 1.3 Loads
    for l in loads:
        x_s = cum_spans[int(l['span_idx'])]
        if l['type'] == 'P':
            fig.add_annotation(
                x=x_s + l['x'], y=0, ax=0, ay=-60,
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#DC2626',
                text=f"<b>P={l['P']}</b>", font=dict(color='#DC2626', size=11),
                bgcolor="white", row=1, col=1
            )
        elif l['type'] == 'U':
            x_e = cum_spans[int(l['span_idx'])+1]
            fig.add_shape(type="rect", x0=x_s, y0=0.15, x1=x_e, y1=0.3, line_width=0, fillcolor='#DC2626', opacity=0.15, row=1, col=1)
            fig.add_annotation(x=(x_s+x_e)/2, y=0.35, showarrow=False, text=f"<b>w={l['w']}</b>", font=dict(color='#DC2626'), row=1, col=1)

    # ==========================================
    # ROW 2: SHEAR FORCE (SFD)
    # ==========================================
    fig.add_shape(type="line", x0=0, x1=total_len, y0=0, y1=0, line=dict(color='black', width=1.5), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'],
        mode='lines', line=dict(color=C_SHEAR, width=2),
        fill='tozeroy', fillcolor='rgba(217, 119, 6, 0.1)',
        name='Shear'
    ), row=2, col=1)

    # Labels (Added @ Distance Back!)
    v_max = df['shear'].abs().max()
    if v_max > 0:
        idx = df['shear'].abs().idxmax()
        row_v = df.loc[idx]
        val = row_v['shear']
        x_loc = row_v['x']
        
        # Format: Value @ Distance
        label_text = f"<b>{val:.2f} @ {x_loc:.2f}m</b>"
        
        fig.add_annotation(
            x=x_loc, y=val, text=label_text,
            showarrow=False, yshift=15 if val>0 else -15,
            font=dict(color=C_SHEAR, size=11), bgcolor="rgba(255,255,255,0.8)", row=2, col=1
        )

    # ==========================================
    # ROW 3: BENDING MOMENT (BMD)
    # ==========================================
    fig.add_shape(type="line", x0=0, x1=total_len, y0=0, y1=0, line=dict(color='black', width=1.5), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df['x'], y=df['moment'],
        mode='lines', line=dict(color=C_MOMENT, width=2),
        fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.1)',
        name='Moment'
    ), row=3, col=1)

    # Labels (Added @ Distance Back!)
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    
    if abs(m_max) > 0.01:
        xm = df.loc[df['moment'] == m_max, 'x'].iloc[0]
        label_text = f"<b>{m_max:.2f} @ {xm:.2f}m</b>"
        fig.add_annotation(x=xm, y=m_max, text=label_text, showarrow=False, yshift=15, font=dict(color=C_MOMENT), row=3, col=1)
        
    if abs(m_min) > 0.01:
        xm = df.loc[df['moment'] == m_min, 'x'].iloc[0]
        label_text = f"<b>{m_min:.2f} @ {xm:.2f}m</b>"
        fig.add_annotation(x=xm, y=m_min, text=label_text, showarrow=False, yshift=-15, font=dict(color=C_MOMENT), row=3, col=1)

    # ==========================================
    # LAYOUT & AXIS
    # ==========================================
    fig.update_layout(
        height=850,
        margin=dict(l=80, r=40, t=40, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        hovermode="x unified",
        font=dict(family="Sarabun, sans-serif", size=14, color='black')
    )

    ax_style = dict(
        showline=True, linewidth=1.5, linecolor='black',
        showgrid=True, gridcolor='#EEEEEE',
        ticks="outside", tickwidth=1.5, ticklen=5,
        mirror=True
    )

    fig.update_xaxes(**ax_style)
    fig.update_yaxes(**ax_style)

    # Force Titles
    fig.update_yaxes(visible=False, row=1, col=1, range=[-0.6, 0.6])
    fig.update_xaxes(visible=True, showticklabels=True, title_text="", row=1, col=1)

    fig.update_yaxes(title_text=f"<b>Shear Force<br>(V) [{unit_force}]</b>", title_standoff=10, row=2, col=1)
    
    fig.update_yaxes(title_text=f"<b>Bending Moment<br>(M) [{unit_force}-{unit_len}]</b>", title_standoff=10, row=3, col=1)
    fig.update_xaxes(title_text=f"<b>Distance (x) [{unit_len}]</b>", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans):
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Shear", f"{df['shear'].abs().max():.2f}")
    c2.metric("Max Moment (+)", f"{df['moment'].max():.2f}")
    c3.metric("Max Moment (-)", f"{df['moment'].min():.2f}")

    st.write("#### 📍 Support Reactions")
    data = []
    n_nodes = len(spans) + 1
    for i in range(n_nodes):
        ry = reac[2*i]
        mz = reac[2*i+1]
        data.append({"Node": i+1, "Ry": f"{ry:.2f}", "Mz": f"{mz:.2f}" if abs(mz)>0.001 else "-"})
    st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
