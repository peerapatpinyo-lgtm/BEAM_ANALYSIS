import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import rc_design 

# --- 1. DRAW DIAGRAMS ---
def draw_diagrams(df, spans, supports, loads, u_force, u_len):
    cum_len = [0] + list(np.cumsum(spans))
    L_total = cum_len[-1]
    
    # Prepare Display Loads
    display_loads = []
    for l in loads:
        if l['type'] == 'U': display_loads.append(l)
    
    # Aggregate Point Loads
    p_map = {}
    for l in loads:
        if l['type'] == 'P':
            key = (l['span_idx'], l['x'])
            p_map[key] = p_map.get(key, 0) + l['P']
            
    for (s_idx, x_val), total_p in p_map.items():
        if total_p != 0:
            display_loads.append({'span_idx': s_idx, 'type': 'P', 'P': total_p, 'x': x_val})

    # Autoscaling
    val_list = [abs(l['w']) for l in display_loads if l['type']=='U'] + [abs(l['P']) for l in display_loads if l['type']=='P']
    max_load = max(val_list) if val_list else 100
    viz_h = max_load * 1.5 
    sup_sz = max(L_total * 0.02, 0.3) 

    # Create Subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Loading Diagram", "Shear Force Diagram (SFD)", "Bending Moment Diagram (BMD)"),
                        row_heights=[0.25, 0.375, 0.375])
    
    # -- Row 1: Beam & Supports --
    fig.add_trace(go.Scatter(x=[0, L_total], y=[0, 0], mode='lines', line=dict(color='black', width=3), hoverinfo='none'), row=1, col=1)

    for i, x in enumerate(cum_len):
        stype = supports.iloc[i]['type']
        if stype == "Pin":
            path = f"M {x} 0 L {x-sup_sz} {-sup_sz} L {x+sup_sz} {-sup_sz} Z"
            fig.add_shape(type="path", path=path, fillcolor="#B0BEC5", line=dict(color="black"), row=1, col=1)
        elif stype == "Roller":
            fig.add_shape(type="circle", x0=x-sup_sz, y0=-sup_sz, x1=x+sup_sz, y1=0, fillcolor="white", line=dict(color="black"), row=1, col=1)
        elif stype == "Fixed":
            fig.add_shape(type="line", x0=x, y0=-sup_sz*1.5, x1=x, y1=sup_sz*1.5, line=dict(color="black", width=4), row=1, col=1)

    # Draw Loads
    for l in display_loads:
        if l['type'] == 'U':
            x1 = cum_len[l['span_idx']]
            x2 = cum_len[l['span_idx']+1]
            h = (abs(l['w'])/max_load) * (viz_h * 0.6)
            fig.add_trace(go.Scatter(x=[x1, x2, x2, x1], y=[0, 0, h, h], fill="toself", fillcolor="rgba(33, 150, 243, 0.3)", line_width=0, hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=[x1, x2], y=[h, h], mode='lines', line=dict(color="#1976D2", width=2), hoverinfo='skip'), row=1, col=1)
            fig.add_annotation(x=(x1+x2)/2, y=h, text=f"Wu={l['w']:.0f}", showarrow=False, yshift=10, row=1, col=1)
        elif l['type'] == 'P':
            px = cum_len[l['span_idx']] + l['x']
            h = (abs(l['P'])/max_load) * (viz_h * 0.6)
            fig.add_annotation(x=px, y=0, ax=0, ay=-40, text=f"Pu={l['P']:.0f}", arrowhead=2, arrowcolor="red", row=1, col=1)
            
    fig.update_yaxes(visible=False, range=[-sup_sz*2.5, viz_h*1.2], row=1, col=1)

    # -- Row 2 & 3: SFD & BMD --
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D32F2F', width=2), name="Shear"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1565C0', width=2), name="Moment"), row=3, col=1)

    # Layout
    fig.update_layout(height=800, template="plotly_white", showlegend=False, hovermode="x unified")
    
    # Spike Lines
    for r in [2, 3]:
        fig.update_xaxes(showspikes=True, spikemode='across', spikedash='dash', spikecolor='gray', row=r, col=1)
    
    st.plotly_chart(fig, use_container_width=True, key="main_diagram")

# --- 2. HELPER: SECTION ---
def render_cross_section(b, h, cover, bars, label=""):
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=2), fillcolor="#ECEFF1")
    fig.add_shape(type="rect", x0=cover, y0=cover, x1=b-cover, y1=h-cover, line=dict(color="green", width=2, dash="dash"))
    
    parsed = rc_design.parse_bars(bars)
    if parsed:
        num, db = parsed
        xs = np.linspace(cover, b-cover, num) if num > 1 else [b/2]
        y_pos = cover + 1.5 if "Bot" in label else h - cover - 1.5
        
        for x in xs:
            r = (db/10)/2 if db>0 else 0.5
            fig.add_shape(type="circle", x0=x-r, y0=y_pos-r, x1=x+r, y1=y_pos+r, fillcolor="red", line_color="black")

    fig.update_xaxes(visible=False, range=[-2, b+2])
    fig.update_yaxes(visible=False, range=[-2, h+2], scaleanchor="x", scaleratio=1)
    fig.update_layout(width=150, height=200, margin=dict(l=0, r=0, t=30, b=0), 
                      title=dict(text=label, y=0.95, x=0.5, font=dict(size=12)),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    return fig

# --- 3. HELPER: LONG PROFILE ---
def render_longitudinal_view(spans, supports, design_data):
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()
    
    fig.add_shape(type="rect", x0=0, y0=0, x1=total_len, y1=10, line=dict(color="black", width=2), fillcolor="#FAFAFA", layer="below")
    
    for i, x in enumerate(cum_len):
        if supports.iloc[i]['type'] != "None":
            fig.add_shape(type="path", path=f"M {x} 0 L {x-0.2} -1.5 L {x+0.2} -1.5 Z", fillcolor="gray", line_color="black")
    
    for i, span_len in enumerate(spans):
        x_start = cum_len[i]
        x_end = cum_len[i+1]
        data = design_data[i]
        
        # Bot
        if data['bot_bars']:
            fig.add_trace(go.Scatter(x=[x_start+0.2, x_end-0.2], y=[1.5, 1.5], mode="lines", 
                                     line=dict(color="blue", width=4), showlegend=False))
            fig.add_annotation(x=x_start+span_len/2, y=1.5, text=data['bot_bars'], yshift=-10, showarrow=False, font=dict(color="blue", size=10))

        # Top
        if data['top_bars']:
            cut = span_len/3.0
            fig.add_trace(go.Scatter(x=[x_start, x_start+cut], y=[8.5, 8.5], mode="lines", 
                                     line=dict(color="red", width=4), showlegend=False))
            fig.add_trace(go.Scatter(x=[x_end-cut, x_end], y=[8.5, 8.5], mode="lines", 
                                     line=dict(color="red", width=4), showlegend=False))
            fig.add_annotation(x=x_start, y=9.5, text=data['top_bars'], showarrow=False, font=dict(color="red", size=10))

        # Stirrups
        if "None" not in data['stirrups']:
            fig.add_annotation(x=x_start+span_len/2, y=5.0, text=data['stirrups'], showarrow=False, 
                               bgcolor="white", bordercolor="green", font=dict(color="green", size=10))

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, range=[-2, 12])
    fig.update_layout(height=250, title="Longitudinal Profile", margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="long_profile_chart")

# --- 4. MAIN RENDERER ---
def render_design_results(df, params, spans, supports):
    st.markdown("### 4️⃣ Design Results")
    
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
        
        des_pos = rc_design.calculate_flexure_sdm(m_pos, "Midspan (+)", params)
        des_neg = rc_design.calculate_flexure_sdm(m_neg, "Support (-)", params)
        v_act, v_cap, stir_txt = rc_design.calculate_shear_capacity(v_u, params)
        
        span_data.append({'bot_bars': des_pos['Bars'], 'top_bars': des_neg['Bars'], 'stirrups': stir_txt})

        with tab:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Top Steel (-M)**")
                st.write(f"Mu: {abs(des_neg['Mu']):.2f}")
                st.info(f"{des_neg['Bars']}")
                if des_neg['Bars'] and "Over" not in des_neg['Status']:
                    st.plotly_chart(render_cross_section(params['b'], params['h'], params['cv'], des_neg['Bars'], "Top"), 
                                    use_container_width=True, key=f"s{i}_top")
            
            with c2:
                st.markdown("**Bottom Steel (+M)**")
                st.write(f"Mu: {des_pos['Mu']:.2f}")
                st.info(f"{des_pos['Bars']}")
                if des_pos['Bars'] and "Over" not in des_pos['Status']:
                    st.plotly_chart(render_cross_section(params['b'], params['h'], params['cv'], des_pos['Bars'], "Bottom"), 
                                    use_container_width=True, key=f"s{i}_bot")

            with c3:
                st.markdown("**Shear (Stirrup)**")
                st.write(f"Vu: {v_act:.2f}")
                st.success(f"{stir_txt}")
                st.caption(f"Capacity: {v_cap:.2f}")

    st.markdown("---")
    render_longitudinal_view(spans, supports, span_data)
