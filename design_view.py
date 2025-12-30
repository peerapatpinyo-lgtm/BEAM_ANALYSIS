import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import rc_design 

def add_peak_annotations(fig, x_data, y_data, row, col, unit):
    try:
        y_vals = np.array([float(y) for y in y_data])
        x_vals = np.array([float(x) for x in x_data])
        
        # 1. Find Max (Positive Peak)
        idx_max = np.argmax(y_vals)
        val_max = y_vals[idx_max]
        x_at_max = x_vals[idx_max]

        # 2. Find Min (Negative Peak)
        idx_min = np.argmin(y_vals)
        val_min = y_vals[idx_min]
        x_at_min = x_vals[idx_min]
        
        # Threshold to ignore near-zero noise
        threshold = 1.0 # Ignore small values like 0.0001
        
        # Draw Max Label
        if val_max > threshold:
            fig.add_annotation(
                x=x_at_max, y=val_max,
                text=f"<b>Max: {val_max:,.0f}<br>@ {x_at_max:.2f}m</b>",
                showarrow=False, yshift=15,
                font=dict(size=10, color="#1B5E20"), bgcolor="rgba(255,255,255,0.8)",
                row=row, col=col
            )
            fig.add_trace(go.Scatter(x=[x_at_max], y=[val_max], mode='markers', marker=dict(color='#1B5E20', size=6), showlegend=False, hoverinfo='skip'), row=row, col=col)

        # Draw Min Label (Only if significantly different from Max or negative)
        if val_min < -threshold:
            fig.add_annotation(
                x=x_at_min, y=val_min,
                text=f"<b>Min: {val_min:,.0f}<br>@ {x_at_min:.2f}m</b>",
                showarrow=False, yshift=-15,
                font=dict(size=10, color="#B71C1C"), bgcolor="rgba(255,255,255,0.8)",
                row=row, col=col
            )
            fig.add_trace(go.Scatter(x=[x_at_min], y=[val_min], mode='markers', marker=dict(color='#B71C1C', size=6), showlegend=False, hoverinfo='skip'), row=row, col=col)
            
    except Exception as e:
        print(f"Annotation Error: {e}")

def draw_diagrams(df, spans, supports, loads, u_force, u_len):
    cum_len = [0] + list(np.cumsum(spans))
    L_total = cum_len[-1]
    
    # Scale calculation for drawing supports
    viz_scale = max(L_total / 15.0, 1.0)
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("<b>Structure & Loads</b>", f"<b>Shear Force ({u_force})</b>", f"<b>Bending Moment ({u_force}-{u_len})</b>"),
        row_heights=[0.25, 0.375, 0.375]
    )
    
    # --- 1. Structure Plot ---
    # Beam Line
    fig.add_trace(go.Scatter(x=[0, L_total], y=[0, 0], mode='lines', line=dict(color='black', width=4), hoverinfo='none'), row=1, col=1)

    # Supports & Grid
    for i, x in enumerate(cum_len):
        # Grid line (Vertical Dash)
        fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="gray", opacity=0.3)
        
        # Support Icon
        try: stype = supports.iloc[i]['type']
        except: stype = "Pin"
        
        fig.add_annotation(x=x, y=-viz_scale*0.6, text=f"<b>N{i+1}</b>", showarrow=False, font=dict(size=10), row=1, col=1)
        
        if stype == "Pin":
            fig.add_trace(go.Scatter(x=[x], y=[-viz_scale/4], mode='markers', marker=dict(symbol='triangle-up', size=14, color='#90A4AE', line=dict(color='black', width=1.5)), showlegend=False, hovertext="Pin"), row=1, col=1)
        elif stype == "Roller":
            fig.add_trace(go.Scatter(x=[x], y=[-viz_scale/4], mode='markers', marker=dict(symbol='circle', size=14, color='white', line=dict(color='black', width=1.5)), showlegend=False, hovertext="Roller"), row=1, col=1)
            fig.add_shape(type="line", x0=x-viz_scale/3, y0=-viz_scale/2, x1=x+viz_scale/3, y1=-viz_scale/2, line=dict(color="black", width=2), row=1, col=1)
        elif stype == "Fixed":
            fig.add_shape(type="rect", x0=x-viz_scale/6, y0=-viz_scale/2, x1=x+viz_scale/6, y1=viz_scale/2, line=dict(color="black", width=2), fillcolor="gray", row=1, col=1)

    # Loads
    try:
        load_vals = [float(l.get('w',0)) for l in loads if l['type']=='U'] + [float(l.get('P',0)) for l in loads if l['type']=='P']
        max_load_val = max(load_vals) if load_vals else 1.0
    except: max_load_val = 100
    
    arrow_scale = viz_scale * 1.5
    
    for l in loads:
        if l['type'] == 'U':
            w = float(l['w'])
            x1, x2 = cum_len[l['span_idx']], cum_len[l['span_idx']+1]
            # Height proportional to load
            h = arrow_scale * (w / max_load_val) if max_load_val > 0 else arrow_scale*0.5
            h = max(h, viz_scale*0.3) # Minimum visibility
            
            fig.add_trace(go.Scatter(x=[x1, x2, x2, x1], y=[0, 0, h, h], fill="toself", fillcolor="rgba(33, 150, 243, 0.2)", line_width=0, hoverinfo='skip'), row=1, col=1)
            fig.add_annotation(x=(x1+x2)/2, y=h, text=f"w={w:.0f}", showarrow=False, yshift=5, font=dict(color="#1565C0", size=10), row=1, col=1)
            
        elif l['type'] == 'P':
            p = float(l['P'])
            px = cum_len[l['span_idx']] + float(l['x'])
            fig.add_annotation(x=px, y=0, ax=0, ay=-50, text=f"P={p:.0f}", arrowhead=2, arrowwidth=2, arrowcolor="#D32F2F", font=dict(color="#D32F2F", size=10), row=1, col=1)

    fig.update_yaxes(visible=False, range=[-viz_scale, arrow_scale*2], row=1, col=1)

    # --- 2. Shear Force ---
    # ใช้ step='hv' หรือ line ปกติ แต่เนื่องจากเรามีจุด shear discontinuity (+-epsilon) แล้ว ใช้ line ปกติจะเห็นเส้นดิ่งสวยงาม
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#E53935', width=2), name="Shear", hovertemplate='V: %{y:,.0f}'), row=2, col=1)
    add_peak_annotations(fig, df['x'].values, df['shear'].values, 2, 1, u_force)
    fig.update_yaxes(title_text=f"Shear ({u_force})", showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black', row=2, col=1)

    # --- 3. Bending Moment ---
    # Invert Y axis for Moment Diagram? (Optional: American standard flips, but typically we plot positive up in simple software. Let's keep normal but clear)
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1E88E5', width=2), name="Moment", hovertemplate='M: %{y:,.0f}'), row=3, col=1)
    add_peak_annotations(fig, df['x'].values, df['moment'].values, 3, 1, f"{u_force}-{u_len}")
    fig.update_yaxes(title_text=f"Moment", showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black', row=3, col=1)

    fig.update_layout(height=800, template="plotly_white", showlegend=False, hovermode="x unified", margin=dict(t=30, b=30, l=60, r=20))
    st.plotly_chart(fig, use_container_width=True, key="main_diagram")

# (ส่วน render_design_results และอื่นๆ เหมือนเดิม ไม่ต้องแก้)
# ... Include render_design_results ...
# ... Include render_combined_section ...
# ... Include render_longitudinal_view ...
# COPY ฟังก์ชัน design view เดิมมาต่อท้ายตรงนี้ได้เลยครับ 
# แต่เพื่อให้โค้ดสมบูรณ์ ผมจะใส่ Stub ไว้ให้ ถ้าคุณมีโค้ดเดิมอยู่แล้วใช้ต่อได้เลย

def render_combined_section(b, h, cover, top_bars, bot_bars):
    # (ใช้ Code เดิมจากคำตอบก่อนหน้า)
    try: b, h, cover = float(b), float(h), float(cover)
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
            for x in xs: fig.add_shape(type="circle", x0=x-r, y0=y_center-r, x1=x+r, y1=y_center+r, fillcolor=color, line=dict(color="black"))
            fig.add_annotation(x=b/2, y=y_center+label_pos_offset, text=f"<b>{bars}</b>", showarrow=False, font=dict(color=color, size=14))
    draw_rebars(bot_bars, cover + 2.0, "#1565C0", 6) 
    draw_rebars(top_bars, h - cover - 2.0, "#C62828", -6)
    fig.update_xaxes(visible=False, range=[-5, b+5])
    fig.update_yaxes(visible=False, range=[-5, h+5], scaleanchor="x", scaleratio=1)
    fig.update_layout(width=200, height=200, margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def render_design_results(df, params, spans, span_props_list, supports):
    # (ใช้ Code เดิมจากคำตอบก่อนหน้า)
    cum_len = [0] + list(np.cumsum(spans))
    summary_data = []
    
    for i in range(len(spans)):
        sp = span_props_list[i]
        b_s, h_s, cv_s = float(sp['b']), float(sp['h']), float(sp['cv'])
        mask = (df['x'] >= cum_len[i]) & (df['x'] <= cum_len[i+1])
        sub_df = df[mask]
        if sub_df.empty: continue
        
        m_pos = float(max(0, sub_df['moment'].max()))
        m_neg = float(min(0, sub_df['moment'].min()))
        v_u = float(sub_df['shear'].abs().max())
        
        des_pos = rc_design.calculate_flexure_sdm(m_pos, "Midspan", b_s, h_s, cv_s, params)
        des_neg = rc_design.calculate_flexure_sdm(m_neg, "Support", b_s, h_s, cv_s, params)
        v_act, v_cap, stir_txt, v_log = rc_design.calculate_shear_capacity(v_u, b_s, h_s, cv_s, params)
        status_icon = "✅" if ("OK" in des_pos['Status'] and "OK" in des_neg['Status'] and v_act <= v_cap) else "⚠️"
        
        summary_data.append({
            "Span": f"{i+1}", "Size": f"{b_s:.0f}x{h_s:.0f}", 
            "Top": des_neg['Bars'], "Bot": des_pos['Bars'], "Stirrup": stir_txt, "Check": status_icon,
            "_p": des_pos, "_n": des_neg, "_v": v_log
        })

    st.table(pd.DataFrame(summary_data).drop(columns=["_p", "_n", "_v"]))

    tabs = st.tabs([f"Span {d['Span']}" for d in summary_data])
    for i, tab in enumerate(tabs):
        d = summary_data[i]
        sp = span_props_list[i]
        with tab:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.plotly_chart(render_combined_section(sp['b'], sp['h'], sp['cv'], d['_n']['Bars'], d['_p']['Bars']), use_container_width=True, key=f"s_{i}")
            with c2:
                with st.expander("Show Calculations", expanded=True):
                    st.caption(f"Design Moments: +{d['_p']['Mu']:.1f} / {d['_n']['Mu']:.1f}")
                    st.markdown(f"**Flexure:** {d['Bot']} (Bot), {d['Top']} (Top)")
                    st.markdown(f"**Shear:** {d['Stirrup']}")

    long_data = [{'bot_bars': d['Bot'], 'top_bars': d['Top'], 'stirrups': d['Stirrup']} for d in summary_data]
    render_longitudinal_view(spans, supports, long_data)

def render_longitudinal_view(spans, supports, design_data):
    # (ใช้ Code เดิมจากคำตอบก่อนหน้า)
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=total_len, y1=10, line=dict(color="black", width=2), fillcolor="#FAFAFA", layer="below")
    for x in cum_len:
        fig.add_vline(x=x, line_width=1, line_dash="dot", line_color="gray")
        fig.add_trace(go.Scatter(x=[x], y=[-1], mode='markers', marker=dict(symbol='triangle-up', size=12, color='gray'), showlegend=False))

    for i, L in enumerate(spans):
        xs, xe = cum_len[i], cum_len[i+1]
        data = design_data[i]
        if data['bot_bars']:
            fig.add_trace(go.Scatter(x=[xs+0.2, xe-0.2], y=[2, 2], mode="lines+text", line=dict(color="#1565C0", width=3), text=[f"{data['bot_bars']}", ""], textposition="top center", showlegend=False))
        if data['top_bars']:
            cut = L/3.5
            fig.add_trace(go.Scatter(x=[xs, xs+cut], y=[8, 8], mode="lines+text", line=dict(color="#C62828", width=3), text=[f"{data['top_bars']}", ""], textposition="bottom center", showlegend=False))
            fig.add_trace(go.Scatter(x=[xe-cut, xe], y=[8, 8], mode="lines", line=dict(color="#C62828", width=3), showlegend=False))
        if data['stirrups']:
            fig.add_annotation(x=xs+L/2, y=5, text=f"{data['stirrups']}", showarrow=False, font=dict(size=9, color="green"))

    fig.update_xaxes(visible=False, range=[-0.5, total_len+0.5])
    fig.update_yaxes(visible=False, range=[-2, 12])
    fig.update_layout(height=200, margin=dict(t=30,b=10), showlegend=False, title="Longitudinal Profile")
    st.plotly_chart(fig, use_container_width=True, key="long_profile_fin")
