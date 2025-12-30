import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import rc_design 

def add_peak_annotations(fig, x_data, y_data, row, col, unit_label=""):
    idx_max = np.argmax(y_data); val_max = y_data[idx_max]; x_max = x_data[idx_max]
    idx_min = np.argmin(y_data); val_min = y_data[idx_min]; x_min = x_data[idx_min]
    
    if abs(val_max) > 0.01:
        fig.add_annotation(x=x_max, y=val_max, text=f"Max: {val_max:,.2f}", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor="rgba(255,255,255,0.8)", row=row, col=col)
        fig.add_trace(go.Scatter(x=[x_max], y=[val_max], mode='markers', marker=dict(color='black', size=8), showlegend=False, hoverinfo='skip'), row=row, col=col)

    if abs(val_min) > 0.01 and idx_min != idx_max:
        fig.add_annotation(x=x_min, y=val_min, text=f"Min: {val_min:,.2f}", showarrow=True, arrowhead=2, ax=0, ay=40, bgcolor="rgba(255,255,255,0.8)", row=row, col=col)
        fig.add_trace(go.Scatter(x=[x_min], y=[val_min], mode='markers', marker=dict(color='black', size=8), showlegend=False, hoverinfo='skip'), row=row, col=col)

def draw_diagrams(df, spans, supports, loads, u_force, u_len):
    cum_len = [0] + list(np.cumsum(spans))
    L_total = cum_len[-1]
    
    val_list = [abs(l['w']) for l in loads if l['type']=='U'] + [abs(l['P']) for l in loads if l['type']=='P']
    max_load = max(val_list) if val_list else 100
    viz_h = max_load * 1.5 
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("<b>Load Model</b>", f"<b>Shear Force ({u_force})</b>", f"<b>Bending Moment ({u_force}-{u_len})</b>"),
                        row_heights=[0.25, 0.375, 0.375])
    
    # Beam Line
    fig.add_trace(go.Scatter(x=[0, L_total], y=[0, 0], mode='lines', line=dict(color='black', width=4), hoverinfo='none'), row=1, col=1)

    # Supports Visualization
    sup_h = viz_h * 0.2
    for i, x in enumerate(cum_len):
        stype = supports.iloc[i]['type']
        if stype == "Pin":
            fig.add_trace(go.Scatter(x=[x], y=[-sup_h/2], mode='markers', marker=dict(symbol='triangle-up', size=15, color='#B0BEC5', line=dict(width=2, color='black')), showlegend=False), row=1, col=1)
        elif stype == "Roller":
            fig.add_trace(go.Scatter(x=[x], y=[-sup_h/2], mode='markers', marker=dict(symbol='circle', size=15, color='white', line=dict(width=2, color='black')), showlegend=False), row=1, col=1)
        elif stype == "Fixed":
            # IMPROVED FIXED SUPPORT: Tall vertical line with hashing (Wall-like)
            wall_h = sup_h * 1.2
            fig.add_shape(type="line", x0=x, y0=-wall_h, x1=x, y1=wall_h, line=dict(color="black", width=4), row=1, col=1)
            # Add hatching lines to indicate fixity
            for h_line in np.linspace(-wall_h, wall_h, 8):
                fig.add_shape(type="line", x0=x, y0=h_line, x1=x-0.4, y1=h_line-0.15, line=dict(color="black", width=1), row=1, col=1)

    # Loads
    for l in loads:
        if l['type'] == 'U':
            x1, x2 = cum_len[l['span_idx']], cum_len[l['span_idx']+1]
            h = (l['w']/max_load) * (viz_h * 0.7)
            fig.add_trace(go.Scatter(x=[x1, x2, x2, x1], y=[0, 0, h, h], fill="toself", fillcolor="rgba(33, 150, 243, 0.3)", line_width=0, hoverinfo='skip'), row=1, col=1)
            fig.add_annotation(x=(x1+x2)/2, y=h, text=f"<b>Wu={l['w']:.0f}</b>", showarrow=False, yshift=10, font=dict(color="#0D47A1", size=10), row=1, col=1)
        elif l['type'] == 'P':
            px = cum_len[l['span_idx']] + l['x']
            h = (l['P']/max_load) * (viz_h * 0.7)
            fig.add_annotation(x=px, y=0, ax=0, ay=-40, text=f"<b>P={l['P']:.0f}</b>", arrowhead=2, arrowcolor="#C62828", font=dict(color="#C62828", size=10), row=1, col=1)

    fig.update_yaxes(visible=False, range=[-sup_h*2.5, viz_h*1.2], row=1, col=1)

    # SFD & BMD with Peak Annotations
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D32F2F', width=2), name="Shear", hovertemplate='V: %{y:,.2f}'), row=2, col=1)
    add_peak_annotations(fig, df['x'].values, df['shear'].values, 2, 1)

    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1565C0', width=2), name="Moment", hovertemplate='M: %{y:,.2f}'), row=3, col=1)
    add_peak_annotations(fig, df['x'].values, df['moment'].values, 3, 1)

    # Layout with Axes Titles
    fig.update_layout(height=800, template="plotly_white", showlegend=False, hovermode="x unified")
    fig.update_xaxes(showspikes=True, spikemode='across', spikethickness=1, spikedash='dash', spikecolor='#546E7A')
    
    # Adding Axis Titles (Missing previously)
    fig.update_xaxes(title_text=f"Distance ({u_len})", row=3, col=1)
    fig.update_yaxes(title_text=f"Force ({u_force})", row=2, col=1)
    fig.update_yaxes(title_text=f"Moment ({u_force}-{u_len})", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def render_combined_section(b, h, cover, top_bars, bot_bars):
    """Draws a single section with both Top and Bottom bars"""
    fig = go.Figure()
    # Concrete Section
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=2), fillcolor="#ECEFF1")
    # Stirrup
    fig.add_shape(type="rect", x0=cover, y0=cover, x1=b-cover, y1=h-cover, line=dict(color="#43A047", width=2, dash="dash"))
    
    # Bot Bars
    parsed_bot = rc_design.parse_bars(bot_bars)
    if parsed_bot:
        num, db = parsed_bot
        xs = np.linspace(cover, b-cover, num) if num > 1 else [b/2]
        y_pos = cover + 1.5
        for x in xs:
            r = (db/10)/2 if db>0 else 0.5
            fig.add_shape(type="circle", x0=x-r, y0=y_pos-r, x1=x+r, y1=y_pos+r, fillcolor="#1565C0", line_color="black")
            
    # Top Bars
    parsed_top = rc_design.parse_bars(top_bars)
    if parsed_top:
        num, db = parsed_top
        xs = np.linspace(cover, b-cover, num) if num > 1 else [b/2]
        y_pos = h - cover - 1.5
        for x in xs:
            r = (db/10)/2 if db>0 else 0.5
            fig.add_shape(type="circle", x0=x-r, y0=y_pos-r, x1=x+r, y1=y_pos+r, fillcolor="#C62828", line_color="black")

    fig.update_xaxes(visible=False, range=[-2, b+2])
    fig.update_yaxes(visible=False, range=[-2, h+2], scaleanchor="x", scaleratio=1)
    fig.update_layout(width=250, height=300, margin=dict(l=10, r=10, t=10, b=10), 
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def render_longitudinal_view(spans, supports, design_data):
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()
    
    fig.add_shape(type="rect", x0=0, y0=0, x1=total_len, y1=10, line=dict(color="black", width=2), fillcolor="#FAFAFA", layer="below")
    
    for i, x in enumerate(cum_len):
        if supports.iloc[i]['type'] != "None":
            fig.add_shape(type="path", path=f"M {x} 0 L {x-0.25} -1.5 L {x+0.25} -1.5 Z", fillcolor="#B0BEC5", line_color="black")
    
    for i, span_len in enumerate(spans):
        x_start = cum_len[i]
        x_end = cum_len[i+1]
        data = design_data[i]
        
        if data['bot_bars']:
            fig.add_trace(go.Scatter(x=[x_start+0.1, x_end-0.1], y=[1.5, 1.5], mode="lines", 
                                     line=dict(color="#1565C0", width=4), hoverinfo='text', text=f"Bot: {data['bot_bars']}"))
            fig.add_annotation(x=x_start+span_len/2, y=1.5, text=data['bot_bars'], yshift=-15, showarrow=False, font=dict(color="#1565C0", size=10))

        if data['top_bars']:
            cut = span_len/3.0
            fig.add_trace(go.Scatter(x=[x_start, x_start+cut], y=[8.5, 8.5], mode="lines", 
                                     line=dict(color="#C62828", width=4), hoverinfo='text', text=f"Top: {data['top_bars']}"))
            fig.add_trace(go.Scatter(x=[x_end-cut, x_end], y=[8.5, 8.5], mode="lines", 
                                     line=dict(color="#C62828", width=4), showlegend=False, hoverinfo='text', text=f"Top: {data['top_bars']}"))

        if "None" not in data['stirrups']:
            fig.add_annotation(x=x_start+span_len/2, y=5.0, text=f"<b>{data['stirrups']}</b>", 
                               showarrow=False, bgcolor="white", bordercolor="#43A047", font=dict(color="#2E7D32", size=10))

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, range=[-2, 12])
    fig.update_layout(height=250, title="<b>Longitudinal Reinforcement Profile</b>", margin=dict(l=20, r=20, t=40, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def render_design_results(df, params, spans, supports):
    st.markdown("### 4️⃣ Structural Design Results")
    cum_len = [0] + list(np.cumsum(spans))
    span_data = []
    
    tabs = st.tabs([f"Span {i+1}" for i in range(len(spans))])

    for i, tab in enumerate(tabs):
        mask = (df['x'] >= cum_len[i]) & (df['x'] <= cum_len[i+1])
        sub_df = df[mask]
        
        if sub_df.empty: 
            span_data.append({'bot_bars': '', 'top_bars': '', 'stirrups': ''})
            continue
        
        m_pos = max(0, sub_df['moment'].max())
        m_neg = min(0, sub_df['moment'].min())
        v_u = sub_df['shear'].abs().max()
        
        # Calculate
        des_pos = rc_design.calculate_flexure_sdm(m_pos, "Midspan (+)", params)
        des_neg = rc_design.calculate_flexure_sdm(m_neg, "Support (-)", params)
        v_act, v_cap, stir_txt, v_log = rc_design.calculate_shear_capacity(v_u, params)
        
        span_data.append({'bot_bars': des_pos['Bars'], 'top_bars': des_neg['Bars'], 'stirrups': stir_txt})

        with tab:
            c_main, c_det = st.columns([1, 1.5])
            
            with c_main:
                st.subheader("Cross Section")
                st.plotly_chart(render_combined_section(params['b'], params['h'], params['cv'], des_neg['Bars'], des_pos['Bars']), use_container_width=True)
                st.info(f"**Stirrups:** {stir_txt}")
                if v_act > v_cap: st.error("Shear Failed")

            with c_det:
                st.subheader("Detailed Calculations")
                with st.expander("Top Reinforcement (-M)", expanded=False):
                    for line in des_neg['Log']: st.markdown(line)
                with st.expander("Bottom Reinforcement (+M)", expanded=False):
                    for line in des_pos['Log']: st.markdown(line)
                with st.expander("Shear Design", expanded=False):
                    for line in v_log: st.markdown(line)

    st.markdown("---")
    render_longitudinal_view(spans, supports, span_data)
