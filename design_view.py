import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import rc_design 

def add_peak_annotations(fig, x_data, y_data, row, col):
    """Annotates Max/Min points without arrows, with explicit x location."""
    idx_max, idx_min = np.argmax(y_data), np.argmin(y_data)
    val_max, val_min = y_data[idx_max], y_data[idx_min]
    x_max, x_min = x_data[idx_max], x_data[idx_min]
    
    # Style for the annotation box
    note_style = dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#333", borderwidth=1, font=dict(size=10, color="black"))
    
    # Max
    if abs(val_max) > 1.0:
        fig.add_annotation(
            x=x_max, y=val_max, text=f"<b>Max: {val_max:,.0f}</b><br>@ x={x_max:.2f}m", 
            showarrow=False, yshift=20, **note_style, row=row, col=col
        )
        fig.add_trace(go.Scatter(x=[x_max], y=[val_max], mode='markers', marker=dict(color='red', size=6), showlegend=False, hoverinfo='skip'), row=row, col=col)
    
    # Min
    if abs(val_min) > 1.0 and idx_min != idx_max:
        fig.add_annotation(
            x=x_min, y=val_min, text=f"<b>Min: {val_min:,.0f}</b><br>@ x={x_min:.2f}m", 
            showarrow=False, yshift=-20, **note_style, row=row, col=col
        )
        fig.add_trace(go.Scatter(x=[x_min], y=[val_min], mode='markers', marker=dict(color='red', size=6), showlegend=False, hoverinfo='skip'), row=row, col=col)

def draw_diagrams(df, spans, supports, loads, u_force, u_len):
    cum_len = [0] + list(np.cumsum(spans))
    L_total = cum_len[-1]
    
    val_list = [abs(l['w']) for l in loads if l['type']=='U'] + [abs(l['P']) for l in loads if l['type']=='P']
    max_val = max(val_list) if val_list else 100
    viz_h = max_val * 1.5 
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("<b>Structure & Loads</b>", f"<b>Shear Force ({u_force})</b>", f"<b>Bending Moment ({u_force}-{u_len})</b>"),
                        row_heights=[0.3, 0.35, 0.35])
    
    # REQ 2: Vertical Grid Lines across ALL subplots at Support locations
    for x_s in cum_len:
        fig.add_vline(x=x_s, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

    # -- Row 1: Structure --
    # Beam Line
    fig.add_trace(go.Scatter(x=[0, L_total], y=[0, 0], mode='lines', line=dict(color='black', width=5), hoverinfo='none'), row=1, col=1)

    sup_sz = viz_h * 0.25
    for i, x in enumerate(cum_len):
        stype = supports.iloc[i]['type']
        # REQ 3: Node Labels (N1, N2...)
        fig.add_annotation(x=x, y=-sup_sz*1.3, text=f"<b>N{i+1}</b>", showarrow=False, font=dict(size=11, color="#333"), row=1, col=1)
        
        if stype == "Pin":
            fig.add_trace(go.Scatter(x=[x], y=[-sup_sz/3], mode='markers', marker=dict(symbol='triangle-up', size=18, color='#90A4AE', line=dict(width=2, color='black')), showlegend=False, hovertext="Pin"), row=1, col=1)
        elif stype == "Roller":
            fig.add_trace(go.Scatter(x=[x], y=[-sup_sz/3], mode='markers', marker=dict(symbol='circle', size=18, color='white', line=dict(width=2, color='black')), showlegend=False, hovertext="Roller"), row=1, col=1)
            # Ground line for roller
            fig.add_shape(type="line", x0=x-sup_sz/2, y0=-sup_sz/1.5, x1=x+sup_sz/2, y1=-sup_sz/1.5, line=dict(color="black", width=2), row=1, col=1)
        elif stype == "Fixed":
            # REQ 1: Engineered Fixed Support Visual (Hatched Wall)
            # Main block
            wall_w = L_total * 0.015
            fig.add_shape(type="rect", x0=x-wall_w, y0=-sup_sz, x1=x+wall_w, y1=sup_sz, line=dict(color="black", width=2), fillcolor="white", row=1, col=1)
            # Hatching
            cnt = 5
            for k in range(cnt):
                dy = (2*sup_sz) * k / (cnt-1) - sup_sz
                fig.add_shape(type="line", x0=x-wall_w, y0=dy, x1=x+wall_w, y1=dy+sup_sz*0.2, line=dict(color="black", width=1), row=1, col=1)

    # Loads
    for l in loads:
        if l['type'] == 'U':
            x1, x2 = cum_len[l['span_idx']], cum_len[l['span_idx']+1]
            h = (l['w']/max_val) * (viz_h * 0.6)
            fig.add_trace(go.Scatter(x=[x1, x2, x2, x1], y=[0, 0, h, h], fill="toself", fillcolor="rgba(30, 136, 229, 0.4)", line_width=0, hoverinfo='skip'), row=1, col=1)
            fig.add_annotation(x=(x1+x2)/2, y=h, text=f"Wu={l['w']:.0f}", showarrow=False, yshift=15, font=dict(color="#0D47A1", size=10), row=1, col=1)
        elif l['type'] == 'P':
            px = cum_len[l['span_idx']] + l['x']
            fig.add_annotation(x=px, y=0, ax=0, ay=-50, text=f"Pu={l['P']:.0f}", arrowhead=2, arrowwidth=2, arrowcolor="#D32F2F", font=dict(color="#D32F2F", size=10), row=1, col=1)

    fig.update_yaxes(visible=False, range=[-sup_sz*2.0, viz_h*1.5], row=1, col=1)

    # -- Row 2 & 3: Diagrams --
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D32F2F', width=2), name="Shear", hovertemplate='V: %{y:,.0f}'), row=2, col=1)
    add_peak_annotations(fig, df['x'].values, df['shear'].values, 2, 1)
    
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1976D2', width=2), name="Moment", hovertemplate='M: %{y:,.0f}'), row=3, col=1)
    add_peak_annotations(fig, df['x'].values, df['moment'].values, 3, 1)

    fig.update_layout(height=900, template="plotly_white", showlegend=False, hovermode="x unified", margin=dict(t=40,b=40))
    fig.update_xaxes(showspikes=True, spikethickness=1, spikedash='solid', spikecolor='#CFD8DC')
    fig.update_xaxes(title_text=f"Distance ({u_len})", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True, key="main_diagram")

def render_combined_section(b, h, cover, top_bars, bot_bars):
    """Draws section with Engineering Standard Dimensions (B, H, d) and Rebar callouts."""
    fig = go.Figure()
    
    # Concrete Outline
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=3), fillcolor="#ECEFF1")
    # Stirrup Outline
    fig.add_shape(type="rect", x0=cover, y0=cover, x1=b-cover, y1=h-cover, line=dict(color="#388E3C", width=2, dash="longdash"))
    
    # Helper to draw rebar dots
    def draw_rebars(bars, y_center, color, label_pos_offset):
        parsed = rc_design.parse_bars(bars)
        if parsed:
            num, db = parsed
            r = db/20 # approx radius in cm for visualization
            xs = np.linspace(cover+r, b-cover-r, num) if num > 1 else [b/2]
            for x in xs:
                fig.add_shape(type="circle", x0=x-r, y0=y_center-r, x1=x+r, y1=y_center+r, fillcolor=color, line_color="black")
            
            # Rebar Callout Label
            fig.add_annotation(x=b/2, y=y_center+label_pos_offset, text=f"<b>{bars}</b>", showarrow=False, font=dict(color=color, size=12))

    draw_rebars(bot_bars, cover + 1.5, "#1565C0", 5)  # Bottom Bars
    draw_rebars(top_bars, h - cover - 1.5, "#C62828", -5) # Top Bars

    # REQ 5: Dimensions Lines (Engineering Style)
    # H Dimension
    fig.add_shape(type="line", x0=-5, y0=0, x1=-5, y1=h, line=dict(color="black", width=1))
    fig.add_annotation(x=-5, y=h/2, text=f"H={h:.1f}", showarrow=False, xshift=-20)
    # B Dimension
    fig.add_shape(type="line", x0=0, y0=-5, x1=b, y1=-5, line=dict(color="black", width=1))
    fig.add_annotation(x=b/2, y=-5, text=f"B={b:.1f}", showarrow=False, yshift=-15)
    # d Dimension
    d_val = h - cover
    fig.add_shape(type="line", x0=b+5, y0=cover, x1=b+5, y1=h, line=dict(color="black", width=1))
    fig.add_annotation(x=b+5, y=(h+cover)/2, text=f"d={d_val:.1f}", showarrow=False, xshift=20)

    fig.update_xaxes(visible=False, range=[-15, b+15])
    fig.update_yaxes(visible=False, range=[-15, h+15], scaleanchor="x", scaleratio=1)
    fig.update_layout(width=300, height=350, margin=dict(l=30, r=30, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def render_longitudinal_view(spans, supports, design_data):
    """REQ 6: Complete longitudinal profile with Supports and Rebar."""
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()
    
    # Beam Body
    fig.add_shape(type="rect", x0=0, y0=0, x1=total_len, y1=12, line=dict(color="black", width=2), fillcolor="#FAFAFA", layer="below")
    
    # Supports & Grid Lines
    for i, x in enumerate(cum_len):
        fig.add_vline(x=x, line_width=1, line_dash="dot", color="gray")
        stype = supports.iloc[i]['type']
        
        # Draw Support Icons
        if stype == "Pin":
             fig.add_trace(go.Scatter(x=[x], y=[-1], mode='markers', marker=dict(symbol='triangle-up', size=15, color='#B0BEC5', line_color='black'), showlegend=False))
        elif stype == "Roller":
             fig.add_trace(go.Scatter(x=[x], y=[-1], mode='markers', marker=dict(symbol='circle', size=15, color='white', line_color='black'), showlegend=False))
             fig.add_shape(type="line", x0=x-0.2, y0=-2, x1=x+0.2, y1=-2, line_width=2)
        elif stype == "Fixed":
             fig.add_shape(type="rect", x0=x-0.15, y0=-2.5, x1=x+0.15, y1=14.5, fillcolor="#546E7A", line_color="black")

    # Rebars
    for i, L in enumerate(spans):
        xs, xe = cum_len[i], cum_len[i+1]
        data = design_data[i]
        
        # Bottom Rebar
        if data['bot_bars']:
            fig.add_trace(go.Scatter(x=[xs+0.2, xe-0.2], y=[2, 2], mode="lines+text", line=dict(color="#1565C0", width=4), 
                                     text=[f"Bot: {data['bot_bars']}", ""], textposition="top right", showlegend=False))
        # Top Rebar
        if data['top_bars']:
            cut = L/3.5 # Simplified curtailment
            # Left Support Top
            fig.add_trace(go.Scatter(x=[xs, xs+cut], y=[10, 10], mode="lines+text", line=dict(color="#C62828", width=4),
                                     text=[f"Top: {data['top_bars']}", ""], textposition="bottom right", showlegend=False))
            # Right Support Top
            fig.add_trace(go.Scatter(x=[xe-cut, xe], y=[10, 10], mode="lines", line=dict(color="#C62828", width=4), showlegend=False))
        
        # Stirrup Text
        if "None" not in data['stirrups']:
            fig.add_annotation(x=xs+L/2, y=6, text=f"Stirrups: {data['stirrups']}", showarrow=False, bordercolor="green", bgcolor="white", font=dict(size=10, color="green"))

    fig.update_xaxes(visible=False, range=[-0.5, total_len+0.5])
    fig.update_yaxes(visible=False, range=[-3, 15])
    fig.update_layout(height=300, title="<b>Longitudinal Reinforcement Profile</b>", margin=dict(t=40,b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="long_profile")

def render_design_results(df, params, spans, span_props_list, supports):
    st.markdown("### 4️⃣ Structural Design Results & Detailing")
    cum_len = [0] + list(np.cumsum(spans))
    span_data = []
    
    tabs = st.tabs([f"Span {i+1}" for i in range(len(spans))])

    for i, tab in enumerate(tabs):
        # Retrieve properties specific to this span
        sp = span_props_list[i]
        b_s, h_s, cv_s = sp['b'], sp['h'], sp['cv']

        mask = (df['x'] >= cum_len[i]) & (df['x'] <= cum_len[i+1])
        sub_df = df[mask]
        
        if sub_df.empty: 
            span_data.append({'bot_bars': '', 'top_bars': '', 'stirrups': ''})
            continue
        
        m_pos = max(0, sub_df['moment'].max())
        m_neg = min(0, sub_df['moment'].min())
        v_u = sub_df['shear'].abs().max()
        
        # Pass span specific dimensions
        des_pos = rc_design.calculate_flexure_sdm(m_pos, "Midspan (+M)", b_s, h_s, cv_s, params)
        des_neg = rc_design.calculate_flexure_sdm(m_neg, "Support (-M)", b_s, h_s, cv_s, params)
        v_act, v_cap, stir_txt, v_log = rc_design.calculate_shear_capacity(v_u, b_s, h_s, cv_s, params)
        
        span_data.append({'bot_bars': des_pos['Bars'], 'top_bars': des_neg['Bars'], 'stirrups': stir_txt})

        with tab:
            st.markdown(f"**Section Properties:** {b_s:.0f} x {h_s:.0f} cm (Cov: {cv_s} cm)")
            c_draw, c_calc = st.columns([1, 1.2])
            with c_draw:
                st.plotly_chart(render_combined_section(b_s, h_s, cv_s, des_neg['Bars'], des_pos['Bars']), use_container_width=True, key=f"sec_{i}")
                
                check = "✅ OK" if (v_act <= v_cap and "OK" in des_pos['Status']) else "❌ Verify"
                st.markdown(f"**Design Status:** {check}")
                st.info(f"Stirrups: {stir_txt}")

            with c_calc:
                with st.expander("📝 Calculation Report", expanded=True):
                    st.markdown("#### Flexure (+M Midspan)")
                    for l in des_pos['Log']: st.markdown(l)
                    st.divider()
                    st.markdown("#### Flexure (-M Support)")
                    for l in des_neg['Log']: st.markdown(l)
                    st.divider()
                    st.markdown("#### Shear")
                    for l in v_log: st.markdown(l)
                    
    st.markdown("---")
    render_longitudinal_view(spans, supports, span_data)
