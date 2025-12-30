import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import rc_design 

def add_peak_annotations(fig, x_data, y_data, row, col, unit):
    """
    บังคับโชว์ Max/Min เสมอ และป้องกัน Error การแปลงค่า
    """
    try:
        # Convert to numpy array of floats immediately
        y_vals = np.array([float(y) for y in y_data])
        x_vals = np.array([float(x) for x in x_data])
        
        # Find indices
        idx_max = np.argmax(y_vals)
        idx_min = np.argmin(y_vals)
        
        val_max = y_vals[idx_max]
        val_min = y_vals[idx_min]
        
        # Style for annotations
        style = dict(
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
            bgcolor="rgba(255,255,255,0.9)", bordercolor="black", borderwidth=1,
            font=dict(size=11, color="black", family="Arial")
        )

        # Plot Max
        fig.add_annotation(
            x=x_vals[idx_max], y=val_max,
            text=f"<b>Max: {val_max:,.0f}</b>",
            ax=0, ay=-40, # Shift up
            row=row, col=col, **style
        )
        # Add a dot for Max
        fig.add_trace(go.Scatter(x=[x_vals[idx_max]], y=[val_max], mode='markers', 
                                 marker=dict(color='red', size=8, line=dict(width=1, color='white')), 
                                 showlegend=False, hoverinfo='skip'), row=row, col=col)

        # Plot Min (Only if value is significantly different to avoid overlap)
        if abs(idx_max - idx_min) > 0 or abs(val_max - val_min) > 1.0:
            fig.add_annotation(
                x=x_vals[idx_min], y=val_min,
                text=f"<b>Min: {val_min:,.0f}</b>",
                ax=0, ay=40, # Shift down
                row=row, col=col, **style
            )
            # Add a dot for Min
            fig.add_trace(go.Scatter(x=[x_vals[idx_min]], y=[val_min], mode='markers', 
                                     marker=dict(color='red', size=8, line=dict(width=1, color='white')), 
                                     showlegend=False, hoverinfo='skip'), row=row, col=col)
            
    except Exception as e:
        # If anything fails, don't crash, just print error to console
        print(f"Annotation Error: {e}")

def draw_diagrams(df, spans, supports, loads, u_force, u_len):
    cum_len = [0] + list(np.cumsum(spans))
    L_total = cum_len[-1]
    
    # Scale calculation for drawing visuals
    viz_scale = L_total / 15.0 
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("<b>Structure & Loads</b>", f"<b>Shear Force ({u_force})</b>", f"<b>Bending Moment ({u_force}-{u_len})</b>"),
        row_heights=[0.3, 0.35, 0.35]
    )
    
    # Grid Lines (Strictly enforced)
    for x_s in cum_len:
        fig.add_vline(x=x_s, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

    # --- ROW 1: Structure ---
    fig.add_trace(go.Scatter(x=[0, L_total], y=[0, 0], mode='lines', line=dict(color='black', width=4), hoverinfo='none'), row=1, col=1)

    for i, x in enumerate(cum_len):
        stype = supports.iloc[i]['type']
        fig.add_annotation(x=x, y=-viz_scale*0.8, text=f"<b>N{i+1}</b>", showarrow=False, font=dict(size=12), row=1, col=1)
        
        if stype == "Pin":
            fig.add_trace(go.Scatter(x=[x], y=[-viz_scale/3], mode='markers', marker=dict(symbol='triangle-up', size=15, color='#B0BEC5', line=dict(width=2, color='black')), showlegend=False, hoverinfo='text', hovertext="Pin"), row=1, col=1)
        elif stype == "Roller":
            fig.add_trace(go.Scatter(x=[x], y=[-viz_scale/3], mode='markers', marker=dict(symbol='circle', size=15, color='white', line=dict(width=2, color='black')), showlegend=False, hoverinfo='text', hovertext="Roller"), row=1, col=1)
            fig.add_shape(type="line", x0=x-viz_scale/2, y0=-viz_scale/1.5, x1=x+viz_scale/2, y1=-viz_scale/1.5, line=dict(color="black", width=2), row=1, col=1)
        elif stype == "Fixed":
            fig.add_shape(type="rect", x0=x-viz_scale/5, y0=-viz_scale/1.5, x1=x+viz_scale/5, y1=viz_scale/1.5, line=dict(color="black", width=2), fillcolor="gray", row=1, col=1)

    # Loads
    # Calculate Max Load for scaling
    try:
        max_load = max([float(l['w']) for l in loads if l['type']=='U'] + [float(l['P']) for l in loads if l['type']=='P'] + [1.0])
    except: max_load = 100

    arrow_h = viz_scale * 1.2
    
    for l in loads:
        try:
            # Force float conversion to prevent 'str' format error
            val_w = float(l.get('w', 0))
            val_p = float(l.get('P', 0))
            
            if l['type'] == 'U':
                x1, x2 = cum_len[l['span_idx']], cum_len[l['span_idx']+1]
                h_draw = arrow_h * (val_w / max_load) if max_load > 0 else arrow_h*0.5
                h_draw = max(h_draw, arrow_h*0.3) # Minimum visual height
                
                fig.add_trace(go.Scatter(x=[x1, x2, x2, x1], y=[0, 0, h_draw, h_draw], fill="toself", fillcolor="rgba(30, 136, 229, 0.3)", line_width=0, hoverinfo='skip'), row=1, col=1)
                fig.add_annotation(x=(x1+x2)/2, y=h_draw, text=f"w={val_w:,.0f}", showarrow=False, yshift=10, font=dict(color="#1565C0", size=10), row=1, col=1)
                
            elif l['type'] == 'P':
                px = cum_len[l['span_idx']] + float(l['x'])
                fig.add_annotation(x=px, y=0, ax=0, ay=-60, text=f"P={val_p:,.0f}", arrowhead=2, arrowwidth=2, arrowcolor="#D32F2F", font=dict(color="#D32F2F", size=10), row=1, col=1)
        except Exception as e:
            continue # Skip bad load data

    # Fix Row 1 Y-Axis
    fig.update_yaxes(visible=False, range=[-viz_scale*1.5, arrow_h*2.0], row=1, col=1)

    # --- ROW 2: Shear ---
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D32F2F', width=2), name="Shear", hovertemplate='V: %{y:,.0f}'), row=2, col=1)
    # Call annotation
    add_peak_annotations(fig, df['x'].values, df['shear'].values, 2, 1, u_force)
    # Enable Y-Axis ticks for Shear
    fig.update_yaxes(title_text=f"Shear ({u_force})", showticklabels=True, showgrid=True, zeroline=True, row=2, col=1)

    # --- ROW 3: Moment ---
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1976D2', width=2), name="Moment", hovertemplate='M: %{y:,.0f}'), row=3, col=1)
    # Call annotation
    add_peak_annotations(fig, df['x'].values, df['moment'].values, 3, 1, f"{u_force}-{u_len}")
    # Enable Y-Axis ticks for Moment
    fig.update_yaxes(title_text=f"Moment ({u_force}-{u_len})", showticklabels=True, showgrid=True, zeroline=True, row=3, col=1)

    # Global Layout
    fig.update_layout(height=850, template="plotly_white", showlegend=False, hovermode="x unified", margin=dict(t=40, b=40, l=60, r=20))
    fig.update_xaxes(showspikes=True, spikethickness=1, spikedash='solid', spikecolor='#CFD8DC')
    fig.update_xaxes(title_text=f"Distance ({u_len})", showticklabels=True, row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True, key="main_diagram")

def render_combined_section(b, h, cover, top_bars, bot_bars):
    try:
        b, h, cover = float(b), float(h), float(cover)
    except: return go.Figure()

    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=3), fillcolor="#ECEFF1")
    fig.add_shape(type="rect", x0=cover, y0=cover, x1=b-cover, y1=h-cover, line=dict(color="#388E3C", width=2, dash="longdash"))
    
    def draw_rebars(bars, y_center, color, label_pos_offset):
        parsed = rc_design.parse_bars(bars)
        if parsed:
            num, db = parsed
            r = db/20 
            if num == 1: xs = [b/2]
            else: xs = np.linspace(cover+r, b-cover-r, num)
            for x in xs:
                fig.add_shape(type="circle", x0=x-r, y0=y_center-r, x1=x+r, y1=y_center+r, fillcolor=color, line_color="black")
            fig.add_annotation(x=b/2, y=y_center+label_pos_offset, text=f"<b>{bars}</b>", showarrow=False, font=dict(color=color, size=14, weight="bold"))

    draw_rebars(bot_bars, cover + 2.0, "#1565C0", 6) 
    draw_rebars(top_bars, h - cover - 2.0, "#C62828", -6)

    fig.update_xaxes(visible=False, range=[-5, b+5])
    fig.update_yaxes(visible=False, range=[-5, h+5], scaleanchor="x", scaleratio=1)
    fig.update_layout(width=250, height=250, margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def render_design_results(df, params, spans, span_props_list, supports):
    st.markdown("### 4️⃣ Design Summary & Detailing")
    cum_len = [0] + list(np.cumsum(spans))
    summary_data = []
    
    for i in range(len(spans)):
        sp = span_props_list[i]
        b_s, h_s, cv_s = sp['b'], sp['h'], sp['cv']
        mask = (df['x'] >= cum_len[i]) & (df['x'] <= cum_len[i+1])
        sub_df = df[mask]
        
        if sub_df.empty: continue
        
        # Safe Max/Min extraction
        try:
            m_pos = float(max(0, sub_df['moment'].max()))
            m_neg = float(min(0, sub_df['moment'].min()))
            v_u = float(sub_df['shear'].abs().max())
        except:
            m_pos, m_neg, v_u = 0, 0, 0
        
        des_pos = rc_design.calculate_flexure_sdm(m_pos, "Midspan", b_s, h_s, cv_s, params)
        des_neg = rc_design.calculate_flexure_sdm(m_neg, "Support", b_s, h_s, cv_s, params)
        v_act, v_cap, stir_txt, v_log = rc_design.calculate_shear_capacity(v_u, b_s, h_s, cv_s, params)
        
        # Determine status
        s_pos = "OK" in des_pos['Status']
        s_neg = "OK" in des_neg['Status']
        s_shr = v_act <= v_cap
        status_icon = "✅" if (s_pos and s_neg and s_shr) else "⚠️"
        
        summary_data.append({
            "Span": f"{i+1}",
            "Size": f"{b_s:.0f}x{h_s:.0f}",
            "Top Bar": des_neg['Bars'],
            "Bot Bar": des_pos['Bars'],
            "Stirrups": stir_txt,
            "Check": status_icon,
            "_p": des_pos, "_n": des_neg, "_v": v_log, "_s": stir_txt
        })

    # Summary Table
    df_sum = pd.DataFrame(summary_data).drop(columns=["_p", "_n", "_v", "_s"])
    st.table(df_sum)

    # Details
    st.markdown("---")
    tabs = st.tabs([f"Span {d['Span']}" for d in summary_data])
    for i, tab in enumerate(tabs):
        d = summary_data[i]
        sp = span_props_list[i]
        with tab:
            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.plotly_chart(render_combined_section(sp['b'], sp['h'], sp['cv'], d['_n']['Bars'], d['_p']['Bars']), use_container_width=True)
            with c2:
                with st.expander("📝 Detailed Calculation", expanded=True):
                    st.markdown(f"**Flexure +M:** {d['Bot Bar']}")
                    for l in d['_p']['Log']: st.markdown(l)
                    st.divider()
                    st.markdown(f"**Flexure -M:** {d['Top Bar']}")
                    for l in d['_n']['Log']: st.markdown(l)
                    st.divider()
                    st.markdown(f"**Shear:** {d['Stirrups']}")
                    for l in d['_v']: st.markdown(l)
                    
    # Longitudinal
    long_data = [{'bot_bars': d['Bot Bar'], 'top_bars': d['Top Bar'], 'stirrups': d['Stirrups']} for d in summary_data]
    render_longitudinal_view(spans, supports, long_data)

def render_longitudinal_view(spans, supports, design_data):
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()
    
    fig.add_shape(type="rect", x0=0, y0=0, x1=total_len, y1=10, line=dict(color="black", width=2), fillcolor="#FAFAFA", layer="below")
    
    for i, x in enumerate(cum_len):
        fig.add_vline(x=x, line_width=1, line_dash="dot", color="gray")
        fig.add_trace(go.Scatter(x=[x], y=[-1], mode='markers', marker=dict(symbol='triangle-up', size=12, color='gray'), showlegend=False))

    for i, L in enumerate(spans):
        xs, xe = cum_len[i], cum_len[i+1]
        data = design_data[i]
        
        if data['bot_bars']:
            fig.add_trace(go.Scatter(x=[xs+0.2, xe-0.2], y=[2, 2], mode="lines+text", line=dict(color="#1565C0", width=3), 
                                     text=[f"{data['bot_bars']}", ""], textposition="top center", showlegend=False))
        if data['top_bars']:
            cut = L/3.5
            fig.add_trace(go.Scatter(x=[xs, xs+cut], y=[8, 8], mode="lines+text", line=dict(color="#C62828", width=3),
                                     text=[f"{data['top_bars']}", ""], textposition="bottom center", showlegend=False))
            fig.add_trace(go.Scatter(x=[xe-cut, xe], y=[8, 8], mode="lines", line=dict(color="#C62828", width=3), showlegend=False))
        
        if data['stirrups']:
            fig.add_annotation(x=xs+L/2, y=5, text=f"{data['stirrups']}", showarrow=False, font=dict(size=9, color="green"))

    fig.update_xaxes(visible=False, range=[-0.5, total_len+0.5])
    fig.update_yaxes(visible=False, range=[-2, 12])
    fig.update_layout(height=250, title="<b>Longitudinal Reinforcement Profile</b>", margin=dict(t=30,b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
