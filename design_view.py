import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import rc_design 

def draw_diagrams(df, spans, supports, loads, u_force, u_len):
    cum_len = [0] + list(np.cumsum(spans))
    L_total = cum_len[-1]
    
    # Scale Helpers
    val_list = [abs(l['w']) for l in loads if l['type']=='U'] + [abs(l['P']) for l in loads if l['type']=='P']
    max_load = max(val_list) if val_list else 100
    viz_h = max_load * 1.5 
    
    # Create Subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("<b>Load Model</b>", "<b>Shear Force (SFD)</b>", "<b>Bending Moment (BMD)</b>"),
                        row_heights=[0.25, 0.375, 0.375])
    
    # -- Row 1: Beam --
    fig.add_trace(go.Scatter(x=[0, L_total], y=[0, 0], mode='lines', line=dict(color='black', width=4), hoverinfo='none'), row=1, col=1)

    # Feature: Correct Support Visuals (ใช้ Markers เพื่อไม่ให้เบี้ยว)
    sup_h = viz_h * 0.15 
    for i, x in enumerate(cum_len):
        stype = supports.iloc[i]['type']
        if stype == "Pin":
            fig.add_trace(go.Scatter(x=[x], y=[-sup_h/2], mode='markers', 
                                     marker=dict(symbol='triangle-up', size=15, color='#B0BEC5', line=dict(width=2, color='black')), 
                                     showlegend=False, hoverinfo='text', text="Pin"), row=1, col=1)
        elif stype == "Roller":
            fig.add_trace(go.Scatter(x=[x], y=[-sup_h/2], mode='markers', 
                                     marker=dict(symbol='circle', size=15, color='white', line=dict(width=2, color='black')), 
                                     showlegend=False, hoverinfo='text', text="Roller"), row=1, col=1)
        elif stype == "Fixed":
            # Fixed ใช้ shape rectangle
            fig.add_shape(type="rect", x0=x-0.1, y0=-sup_h, x1=x+0.1, y1=sup_h, fillcolor="black", line_width=0, row=1, col=1)
            # Hatch
            for h_line in np.linspace(-sup_h, sup_h, 5):
                fig.add_shape(type="line", x0=x, y0=h_line, x1=x-0.3, y1=h_line-0.1, line=dict(color="black", width=1), row=1, col=1)

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

    fig.update_yaxes(visible=False, range=[-sup_h*2, viz_h*1.2], row=1, col=1)

    # -- Row 2 & 3: SFD & BMD --
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D32F2F', width=2), name="Shear", hovertemplate='V: %{y:,.2f}'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1565C0', width=2), name="Moment", hovertemplate='M: %{y:,.2f}'), row=3, col=1)

    # Feature: Spike Lines & Unified Hover (สำคัญมาก)
    fig.update_layout(height=800, template="plotly_white", showlegend=False, hovermode="x unified")
    fig.update_xaxes(showspikes=True, spikemode='across', spikethickness=1, spikedash='dash', spikecolor='#546E7A')
    fig.update_yaxes(showspikes=True, spikemode='across', spikethickness=1, spikedash='dash', spikecolor='#546E7A')
    
    st.plotly_chart(fig, use_container_width=True)

def render_cross_section(b, h, cover, bars, label=""):
    fig = go.Figure()
    # Concrete
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=2), fillcolor="#ECEFF1")
    # Stirrup
    fig.add_shape(type="rect", x0=cover, y0=cover, x1=b-cover, y1=h-cover, line=dict(color="#43A047", width=2, dash="dash"))
    
    parsed = rc_design.parse_bars(bars)
    if parsed:
        num, db = parsed
        xs = np.linspace(cover, b-cover, num) if num > 1 else [b/2]
        # Top or Bottom logic
        y_pos = cover + 1.5 if "Bot" in label else h - cover - 1.5
        color = "#1565C0" if "Bot" in label else "#D32F2F"
        
        for x in xs:
            r = (db/10)/2 if db>0 else 0.5
            fig.add_shape(type="circle", x0=x-r, y0=y_pos-r, x1=x+r, y1=y_pos+r, fillcolor=color, line_color="black")

    fig.update_xaxes(visible=False, range=[-2, b+2])
    fig.update_yaxes(visible=False, range=[-2, h+2], scaleanchor="x", scaleratio=1) # Aspect Ratio Fixed
    fig.update_layout(width=160, height=220, margin=dict(l=5, r=5, t=30, b=5), 
                      title=dict(text=label, y=0.95, x=0.5, font=dict(size=12)),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def render_longitudinal_view(spans, supports, design_data):
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()
    
    # Beam
    fig.add_shape(type="rect", x0=0, y0=0, x1=total_len, y1=10, line=dict(color="black", width=2), fillcolor="#FAFAFA", layer="below")
    
    # Supports
    for i, x in enumerate(cum_len):
        if supports.iloc[i]['type'] != "None":
            fig.add_shape(type="path", path=f"M {x} 0 L {x-0.25} -1.5 L {x+0.25} -1.5 Z", fillcolor="#B0BEC5", line_color="black")
    
    for i, span_len in enumerate(spans):
        x_start = cum_len[i]
        x_end = cum_len[i+1]
        data = design_data[i]
        
        # Feature: Longitudinal Bars Drawing
        # Bottom Bar (Blue) - Full Span
        if data['bot_bars']:
            fig.add_trace(go.Scatter(x=[x_start+0.1, x_end-0.1], y=[1.5, 1.5], mode="lines", 
                                     line=dict(color="#1565C0", width=4), hoverinfo='text', text=f"Bot: {data['bot_bars']}"))
            fig.add_annotation(x=x_start+span_len/2, y=1.5, text=data['bot_bars'], yshift=-15, showarrow=False, font=dict(color="#1565C0", size=10))

        # Top Bar (Red) - At Supports (Approx 1/3)
        if data['top_bars']:
            cut = span_len/3.0
            fig.add_trace(go.Scatter(x=[x_start, x_start+cut], y=[8.5, 8.5], mode="lines", 
                                     line=dict(color="#C62828", width=4), hoverinfo='text', text=f"Top: {data['top_bars']}"))
            fig.add_trace(go.Scatter(x=[x_end-cut, x_end], y=[8.5, 8.5], mode="lines", 
                                     line=dict(color="#C62828", width=4), showlegend=False, hoverinfo='text', text=f"Top: {data['top_bars']}"))

        # Stirrups Text
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
    
    # Feature: Separate Tabs per Span
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
        
        # Calculate using rc_design
        des_pos = rc_design.calculate_flexure_sdm(m_pos, "Midspan (+)", params)
        des_neg = rc_design.calculate_flexure_sdm(m_neg, "Support (-)", params)
        v_act, v_cap, stir_txt = rc_design.calculate_shear_capacity(v_u, params)
        
        span_data.append({'bot_bars': des_pos['Bars'], 'top_bars': des_neg['Bars'], 'stirrups': stir_txt})

        with tab:
            c1, c2, c3 = st.columns(3)
            # Section Views
            with c1:
                st.markdown("**Top Support (-M)**")
                st.caption(f"Mu = {abs(des_neg['Mu']):.2f}")
                st.info(f"**{des_neg['Bars']}**")
                if des_neg['Bars'] and "Over" not in des_neg['Status']:
                    st.plotly_chart(render_cross_section(params['b'], params['h'], params['cv'], des_neg['Bars'], "Top Section"), use_container_width=True, key=f"s{i}_top")
                else: st.warning(des_neg['Status'])
                    
            with c2:
                st.markdown("**Bottom Midspan (+M)**")
                st.caption(f"Mu = {des_pos['Mu']:.2f}")
                st.info(f"**{des_pos['Bars']}**")
                if des_pos['Bars'] and "Over" not in des_pos['Status']:
                    st.plotly_chart(render_cross_section(params['b'], params['h'], params['cv'], des_pos['Bars'], "Bot Section"), use_container_width=True, key=f"s{i}_bot")
                else: st.warning(des_pos['Status'])
                    
            with c3:
                st.markdown("**Shear (Stirrup)**")
                st.caption(f"Vu = {v_act:.2f} | Cap = {v_cap:.2f}")
                st.success(f"**{stir_txt}**")
                if v_act > v_cap: st.error("Shear Fail!")

    st.markdown("---")
    # Feature: Longitudinal Profile View
    render_longitudinal_view(spans, supports, span_data)
