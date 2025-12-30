import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- ENGINEERING PALETTE ---
C_SHEAR_FILL = 'rgba(245, 158, 11, 0.2)'  # Amber with opacity
C_SHEAR_LINE = '#D97706'                  # Dark Amber
C_MOMENT_FILL = 'rgba(37, 99, 235, 0.2)'  # Blue with opacity
C_MOMENT_LINE = '#2563EB'                 # Royal Blue
C_BEAM = '#1F2937'                        # Dark Grey
C_LOAD = '#DC2626'                        # Red
C_GRID = '#E5E7EB'

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    """
    Creates a synchronized interactive subplot system (Model + SFD + BMD).
    Fits within screen height. No scrolling needed.
    """
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # 1. Setup Subplots (3 Rows, Shared X-Axis)
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        row_heights=[0.2, 0.4, 0.4],
        subplot_titles=("<b>STRUCTURAL MODEL</b>", f"<b>SHEAR FORCE ({unit_force})</b>", f"<b>BENDING MOMENT ({unit_force}-{unit_len})</b>")
    )

    # === ROW 1: BEAM MODEL ===
    # Draw Beam
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0],
        mode='lines', line=dict(color=C_BEAM, width=5),
        hoverinfo='skip', showlegend=False
    ), row=1, col=1)

    # Draw Supports
    sup_x = [cum_spans[int(row['id'])] for _, row in sup_df.iterrows()]
    sup_text = [row['type'] for _, row in sup_df.iterrows()]
    fig.add_trace(go.Scatter(
        x=sup_x, y=[0]*len(sup_x),
        mode='markers', marker=dict(symbol='triangle-up', size=15, color='#374151'),
        text=sup_text, hoverinfo='text', name='Support'
    ), row=1, col=1)

    # Draw Loads (Interactive Arrows)
    for l in loads:
        x_s = cum_spans[int(l['span_idx'])]
        if l['type'] == 'P':
            fig.add_annotation(
                x=x_s + l['x'], y=0, ax=0, ay=-40, xref='x1', yref='y1',
                arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor=C_LOAD,
                text=f"<b>P={l['P']}</b>", font=dict(color=C_LOAD)
            )
        elif l['type'] == 'U':
            x_e = cum_spans[int(l['span_idx'])+1]
            fig.add_shape(
                type="rect", x0=x_s, y0=0.05, x1=x_e, y1=0.25,
                line=dict(width=0), fillcolor=C_LOAD, opacity=0.15,
                row=1, col=1
            )
            fig.add_annotation(
                x=(x_s+x_e)/2, y=0.35, showarrow=False,
                text=f"<b>w={l['w']}</b>", font=dict(color=C_LOAD)
            )

    # === ROW 2: SHEAR (SFD) ===
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'],
        mode='lines', line=dict(color=C_SHEAR_LINE, width=2),
        fill='tozeroy', fillcolor=C_SHEAR_FILL,
        name='Shear', customdata=df['shear']
    ), row=2, col=1)

    # Annotate Max Shear
    if not df['shear'].empty:
        max_v = df['shear'].abs().max()
        max_v_row = df.loc[df['shear'].abs() == max_v].iloc[0]
        fig.add_annotation(
            x=max_v_row['x'], y=max_v_row['shear'],
            text=f"<b>Vmax: {max_v:.2f}</b>", showarrow=True, arrowhead=1,
            row=2, col=1, font=dict(color=C_SHEAR_LINE)
        )

    # === ROW 3: MOMENT (BMD) ===
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['moment'],
        mode='lines', line=dict(color=C_MOMENT_LINE, width=2),
        fill='tozeroy', fillcolor=C_MOMENT_FILL,
        name='Moment'
    ), row=3, col=1)

    # Annotate Max/Min Moment
    if not df['moment'].empty:
        max_m = df['moment'].max()
        min_m = df['moment'].min()
        
        if abs(max_m) > 1e-3:
            xm = df.loc[df['moment'] == max_m, 'x'].iloc[0]
            fig.add_annotation(x=xm, y=max_m, text=f"<b>M(+): {max_m:.2f}</b>", showarrow=True, arrowhead=1, row=3, col=1, font=dict(color=C_MOMENT_LINE))
        
        if abs(min_m) > 1e-3:
            xm = df.loc[df['moment'] == min_m, 'x'].iloc[0]
            fig.add_annotation(x=xm, y=min_m, text=f"<b>M(-): {min_m:.2f}</b>", showarrow=True, arrowhead=1, row=3, col=1, ay=30, font=dict(color=C_MOMENT_LINE))

    # === GLOBAL LAYOUT ===
    fig.update_layout(
        height=650,  # พอดีหน้าจอ Laptop ทั่วไป
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        hovermode="x unified", # Highlight: Hover แล้วขึ้นเส้นตัดทุกกราฟ
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="monospace"),
    )
    
    fig.update_xaxes(showgrid=True, gridcolor=C_GRID, zeroline=True, zerolinecolor='black')
    fig.update_yaxes(showgrid=True, gridcolor=C_GRID, zeroline=True, zerolinecolor='black')
    fig.update_yaxes(visible=False, showticklabels=False, row=1, col=1)

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_engineering_table(df, reac, spans):
    """Renders a compact, high-density engineering summary table."""
    v_max = df['shear'].abs().max()
    m_max_pos = df['moment'].max()
    m_max_neg = df['moment'].min()
    
    st.markdown("""
    <style>
        .eng-table { width: 100%; border-collapse: collapse; font-family: 'monospace'; font-size: 0.85rem; }
        .eng-table td, .eng-table th { border: 1px solid #ddd; padding: 4px 8px; text-align: right; }
        .eng-table th { background-color: #f8f9fa; font-weight: bold; text-align: center; color: #333; }
        .val-highlight { color: #d63031; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.caption("📍 Support Reactions")
        r_data = [{"Node": f"Sup {i+1}", "Ry": f"{reac[2*i]:.2f}", "Mz": f"{reac[2*i+1]:.2f}" if abs(reac[2*i+1])>0.01 else "-"} 
                  for i in range(len(spans)+1)]
        st.table(pd.DataFrame(r_data).set_index("Node"))
        
    with c2:
        st.caption("🔥 Governing Design Forces")
        html = f"""
        <table class="eng-table">
            <tr><th>Force Type</th><th>Value</th></tr>
            <tr><td>Max Shear (Vu)</td><td class="val-highlight">{v_max:.2f}</td></tr>
            <tr><td>Max Moment (+)</td><td class="val-highlight">{m_max_pos:.2f}</td></tr>
            <tr><td>Max Moment (-)</td><td class="val-highlight">{m_max_neg:.2f}</td></tr>
        </table>
        """
        st.markdown(html, unsafe_allow_html=True)
