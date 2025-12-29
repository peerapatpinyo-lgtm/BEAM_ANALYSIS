import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import rc_design 

def draw_diagrams(df, spans, supports, loads, u_force, u_len):
    cum_len = [0] + list(np.cumsum(spans))
    L_total = cum_len[-1]
    
    val_list = [abs(l['w']) for l in loads if l['type']=='U'] + [abs(l['P']) for l in loads if l['type']=='P']
    max_load = max(val_list) if val_list else 100
    
    viz_h = max_load * 1.5
    sup_sz = max(0.2, L_total * 0.025) 
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=(f"<b>Loading Diagram</b>", 
                                        f"<b>Shear Force ({u_force})</b>", 
                                        f"<b>Bending Moment ({u_force}-{u_len})</b>"),
                        row_heights=[0.3, 0.35, 0.35])
    
    # 1. Loading Diagram (Beam Line Added)
    fig.add_trace(go.Scatter(x=[0, L_total], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=5), hoverinfo='none'), row=1, col=1)

    # Supports
    for i, x in enumerate(cum_len):
        stype = supports.iloc[i]['type']
        if stype == "Pin":
            fig.add_trace(go.Scatter(x=[x, x-sup_sz/2, x+sup_sz/2, x], 
                                     y=[0, -sup_sz, -sup_sz, 0], 
                                     fill="toself", fillcolor="#90A4AE", line_color="black", showlegend=False, hoverinfo='text', text="Pin"), row=1, col=1)
        elif stype == "Roller":
            fig.add_trace(go.Scatter(x=[x], y=[-sup_sz/2], mode="markers", 
                                     marker=dict(size=14, color="white", line=dict(color="black", width=2)), 
                                     showlegend=False, hoverinfo='text', text="Roller"), row=1, col=1)
            fig.add_trace(go.Scatter(x=[x-sup_sz/2, x+sup_sz/2], y=[-sup_sz, -sup_sz], mode="lines", 
                                     line=dict(color="black", width=2), showlegend=False), row=1, col=1)
        elif stype == "Fixed":
            fig.add_shape(type="line", x0=x, y0=-sup_sz, x1=x, y1=sup_sz, line=dict(color="black", width=4), row=1, col=1)
            for h in np.linspace(-sup_sz, sup_sz, 5):
                fig.add_shape(type="line", x0=x, y0=h, x1=x-sup_sz/3, y1=h-sup_sz/4, line=dict(color="black", width=1), row=1, col=1)
            
    # Loads
    for l in loads:
        if l['type'] == 'U':
            x1, x2 = cum_len[l['span_idx']], cum_len[l['span_idx']+1]
            h = (abs(l['w'])/max_load) * (viz_h * 0.5)
            fig.add_trace(go.Scatter(x=[x1, x2, x2, x1], y=[0, 0, h, h], fill="toself", fillcolor="rgba(33, 150, 243, 0.3)", line_width=0, showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=[x1, x2], y=[h, h], mode='lines', line=dict(color="#1976D2", width=2), showlegend=False), row=1, col=1)
            fig.add_annotation(x=(x1+x2)/2, y=h, text=f"<b>w = {l['w']:.0f}</b>", showarrow=False, yshift=15, font=dict(color="#0D47A1"), row=1, col=1)
        elif l['type'] == 'P':
            px = cum_len[l['span_idx']] + l['x']
            h = (abs(l['P'])/max_load) * (viz_h * 0.5)
            fig.add_annotation(x=px, y=0, ax=px, ay=h if h>0 else 10, arrowcolor="#D32F2F", arrowhead=2, arrowwidth=2, row=1, col=1)
            fig.add_annotation(x=px, y=h if h>0 else 10, text=f"<b>P = {l['P']:.0f}</b>", showarrow=False, yshift=10, font=dict(color="#D32F2F"), row=1, col=1)
            
    fig.update_yaxes(visible=False, range=[-sup_sz*1.5, viz_h*1.2], row=1, col=1)
    
    # SFD & BMD
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D32F2F', width=2), fillcolor='rgba(211, 47, 47, 0.1)', name="Shear"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1565C0', width=2), fillcolor='rgba(21, 101, 192, 0.1)', name="Moment"), row=3, col=1)
    
    # Max/Min Labels
    for col, row, color, unit in [('shear', 2, '#D32F2F', u_force), ('moment', 3, '#1565C0', f"{u_force}-{u_len}")]:
        arr = df[col].to_numpy()
        mx, mn = np.max(arr), np.min(arr)
        imx, imn = np.argmax(arr), np.argmin(arr)
        
        for val, idx, pos in [(mx, imx, "top"), (mn, imn, "bottom")]:
            if abs(val) > 1e-3: 
                ys = 15 if pos=="top" else -15
                fig.add_annotation(
                    x=df['x'].iloc[idx], y=val, 
                    text=f"<b>{val:,.2f}</b>", 
                    showarrow=False, 
                    bgcolor="rgba(255,255,255,0.9)", bordercolor=color, borderwidth=1, borderpad=2,
                    font=dict(color=color, size=10), yshift=ys, row=row, col=1
                )
        
        rng = mx - mn
        if rng == 0: rng = 10
        pad = rng * 0.25
        fig.update_yaxes(range=[mn-pad, mx+pad], row=row, col=1)

    fig.update_layout(height=850, template="plotly_white", margin=dict(t=40, b=40, l=50, r=40), showlegend=False)
    fig.update_xaxes(showgrid=True, gridcolor='#ECEFF1', title=f"Distance ({u_len})", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

def render_cross_section(b, h, cover, top_bars, bot_bars):
    fig = go.Figure()
    # Concrete
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=2), fillcolor="#E0E0E0")
    # Stirrup
    fig.add_shape(type="rect", x0=cover, y0=cover, x1=b-cover, y1=h-cover, line=dict(color="#4CAF50", width=2, dash="dash"))
    
    def draw_bars(bar_str, y_pos):
        parsed = rc_design.parse_bars(bar_str)
        if parsed:
            num, db = parsed
            eff_width = b - 2*cover
            if num > 1:
                spacing = eff_width / (num - 1)
                xs = [cover + i*spacing for i in range(num)]
            else:
                xs = [b/2]
            for x in xs:
                radius = (db/10)/2 
                fig.add_shape(type="circle", x0=x-radius, y0=y_pos-radius, x1=x+radius, y1=y_pos+radius, 
                              fillcolor="#D32F2F", line_color="black")
    
    draw_bars(bot_bars, cover + 1.5)
    draw_bars(top_bars, h - cover - 1.5)
    
    fig.update_xaxes(visible=False, range=[-5, b+5])
    fig.update_yaxes(visible=False, range=[-5, h+5], scaleanchor="x", scaleratio=1)
    fig.update_layout(width=180, height=220, margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    return fig

def render_design_results(df, params):
    st.markdown('<div class="section-header">4️⃣ Reinforced Concrete Design</div>', unsafe_allow_html=True)
    
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    u_moment = f"{params['u_force']}-{params['u_len']}"
    
    if "SDM" in params['method']:
        res_pos = rc_design.calculate_flexure_sdm(m_max, "Mid-Span (+M)", params)
        res_neg = rc_design.calculate_flexure_sdm(m_min, "Support (-M)", params)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="card">
                <h4>{res_pos['Type']}</h4>
                <div style="font-size:1.5em; font-weight:bold; color:#1565C0;">{res_pos['Mu']:,.2f} <span style="font-size:0.6em">{u_moment}</span></div>
                <hr>
                <p>Req. As: <b>{res_pos['As_req']:,.2f} cm²</b></p>
                <p style="color:{'red' if 'Over' in res_pos['Status'] else 'green'}"><b>{res_pos['Status']}</b></p>
                <div style="background:#E3F2FD; padding:8px; border-radius:5px;"><b>Use:</b> {res_pos['Bars']}</div>
                <small style="color:gray">Logs: {', '.join(res_pos.get('Logs',[]))}</small>
            </div>
            """, unsafe_allow_html=True)
            if "Over" not in res_pos['Status']:
                st.plotly_chart(render_cross_section(params['b'], params['h'], params['cv'], "", res_pos['Bars']), use_container_width=True)

        with c2:
            st.markdown(f"""
            <div class="card">
                <h4>{res_neg['Type']}</h4>
                <div style="font-size:1.5em; font-weight:bold; color:#C62828;">{res_neg['Mu']:,.2f} <span style="font-size:0.6em">{u_moment}</span></div>
                <hr>
                <p>Req. As: <b>{res_neg['As_req']:,.2f} cm²</b></p>
                <p style="color:{'red' if 'Over' in res_neg['Status'] else 'green'}"><b>{res_neg['Status']}</b></p>
                <div style="background:#FFEBEE; padding:8px; border-radius:5px;"><b>Use:</b> {res_neg['Bars']}</div>
                <small style="color:gray">Logs: {', '.join(res_neg.get('Logs',[]))}</small>
            </div>
            """, unsafe_allow_html=True)
            if "Over" not in res_neg['Status']:
                st.plotly_chart(render_cross_section(params['b'], params['h'], params['cv'], res_neg['Bars'], ""), use_container_width=True)
    else:
        st.info("WSD Method not implemented in this version.")

    st.markdown("---")
    st.markdown(f"#### Shear Design ({params['u_force']})")
    v_max = df['shear'].abs().max()
    vu_val, phi_vc = rc_design.calculate_shear_capacity(v_max, params)
    
    color = "green" if vu_val <= phi_vc/2 else ("orange" if vu_val <= phi_vc else "red")
    req_stirrup = "None" if vu_val <= phi_vc/2 else ("Min Stirrups" if vu_val <= phi_vc else "Design Stirrups Req.")
        
    st.markdown(f"""
    <div style="padding:15px; background-color:white; border-left:6px solid {color}; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
        <b>Max Shear Vu:</b> {vu_val:,.2f} {params['u_force']} <br>
        <b>Capacity ϕVc:</b> {phi_vc:,.2f} {params['u_force']} <br>
        <b>Status:</b> <span style="color:{color}; font-weight:bold">{req_stirrup}</span>
    </div>
    """, unsafe_allow_html=True)
