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
    # Row 1: Model (20%), Row 2: SFD (40%), Row 3: BMD (40%)
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        row_heights=[0.2, 0.4, 0.4],
        subplot_titles=("<b>STRUCTURAL MODEL</b>", f"<b>SHEAR FORCE DIAGRAM ({unit_force})</b>", f"<b>BENDING MOMENT DIAGRAM ({unit_force}-{unit_len})</b>")
    )

    # === ROW 1: BEAM MODEL ===
    # Draw Beam
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0],
        mode='lines', line=dict(color=C_BEAM, width=6),
        hoverinfo='skip', showlegend=False
    ), row=1, col=1)

    # Draw Supports (Markers)
    sup_x = [cum_spans[int(row['id'])] for _, row in sup_df.iterrows()]
    sup_text = [row['type'] for _, row in sup_df.iterrows()]
    fig.add_trace(go.Scatter(
        x=sup_x, y=[0]*len(sup_x),
        mode='markers', marker=dict(symbol='triangle-up', size=15, color='#374151'),
        text=sup_text, hoverinfo='text', name='Support'
    ), row=1, col=1)

    # Draw Loads (Arrows logic visualized as markers/annotations for Plotly)
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
            # Draw a shaded rectangle for UDL
            fig.add_shape(
                type="rect", x0=x_s, y0=0.1, x1=x_e, y1=0.5,
                line=dict(color=C_LOAD, width=0), fillcolor=C_LOAD, opacity=0.1,
                row=1, col=1
            )
            fig.add_annotation(
                x=(x_s+x_e)/2, y=0.6, showarrow=False,
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
    max_v = df['shear'].abs().max()
    max_v_x = df.loc[df['shear'].abs() == max_v, 'x'].iloc[0]
    fig.add_annotation(
        x=max_v_x, y=df.loc[df['x']==max_v_x, 'shear'].values[0],
        text=f"<b>V_max: {max_v:.2f}</b>", showarrow=True, arrowhead=1,
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
    max_m = df['moment'].max()
    min_m = df['moment'].min()
    
    if abs(max_m) > 0.1:
        xm = df.loc[df['moment'] == max_m, 'x'].iloc[0]
        fig.add_annotation(x=xm, y=max_m, text=f"<b>M(+): {max_m:.2f}</b>", showarrow=True, arrowhead=1, row=3, col=1, font=dict(color=C_MOMENT_LINE))
    
    if abs(min_m) > 0.1:
        xm = df.loc[df['moment'] == min_m, 'x'].iloc[0]
        fig.add_annotation(x=xm, y=min_m, text=f"<b>M(-): {min_m:.2f}</b>", showarrow=True, arrowhead=1, row=3, col=1, ay=30, font=dict(color=C_MOMENT_LINE))

    # === GLOBAL LAYOUT & STYLING (The "Educated" Look) ===
    fig.update_layout(
        height=700,  # Fixed height to fit standard laptop screen
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        hovermode="x unified", # <--- THE KEY FEATURE (Spike Line across all)
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Roboto Mono"),
    )
    
    # Configure Axes
    fig.update_xaxes(showgrid=True, gridcolor=C_GRID, zeroline=True, zerolinecolor='black')
    fig.update_yaxes(showgrid=True, gridcolor=C_GRID, zeroline=True, zerolinecolor='black')
    fig.update_yaxes(visible=False, showticklabels=False, row=1, col=1) # Hide Y axis for model

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- ENGINEERING RESULT TABLE ---
def render_engineering_table(df, reac, spans):
    """
    Renders a compact, high-density data table often seen in pro software.
    """
    # 1. Reaction Table
    r_data = []
    for i, r in enumerate(reac[::2]):
        m = reac[2*i+1]
        r_data.append({"Node": f"Sup {i+1}", "Ry (kg)": f"{r:.2f}", "Mz (kg-m)": f"{m:.2f}" if abs(m)>0.01 else "-"})
    
    # 2. Critical Values
    v_max = df['shear'].abs().max()
    m_max_pos = df['moment'].max()
    m_max_neg = df['moment'].min()
    
    # CSS for compact table
    st.markdown("""
    <style>
        .eng-table { width: 100%; border-collapse: collapse; font-family: 'Roboto Mono', monospace; font-size: 0.9rem; }
        .eng-table td, .eng-table th { border: 1px solid #ddd; padding: 4px 8px; text-align: right; }
        .eng-table th { background-color: #f8f9fa; font-weight: bold; text-align: center; color: #333; }
        .highlight { color: #d63031; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.caption("📍 Support Reactions")
        st.table(pd.DataFrame(r_data).set_index("Node"))
        
    with c2:
        st.caption("🔥 Critical Design Forces (Governing)")
        html = f"""
        <table class="eng-table">
            <tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
            <tr><td>V<sub>u,max</sub></td><td class="highlight">{v_max:.2f}</td><td>kg</td></tr>
            <tr><td>M<sub>u,pos</sub></td><td class="highlight">{m_max_pos:.2f}</td><td>kg-m</td></tr>
            <tr><td>M<sub>u,neg</sub></td><td class="highlight">{m_max_neg:.2f}</td><td>kg-m</td></tr>
        </table>
        """
        st.markdown(html, unsafe_allow_html=True)
