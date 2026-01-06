import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# ==========================================
# 1. CAPACITY CHECK (แบบแถบสี Safe Zone)
# ==========================================
def plot_capacity_vs_demand(df_span, phi_Mn_pos, phi_Mn_neg):
    """
    สร้างกราฟแบบ 'Envelope' ดูง่าย
    พื้นที่ระบายสี = Capacity
    เส้นกราฟ = Demand
    """
    mu_kNm = df_span['moment'] / 1000.0
    x = df_span['x']
    
    fig = go.Figure()

    # --- 1. สร้างแถบ Capacity (Safe Zone) ---
    # เทคนิค: สร้างเส้นล่าง (-Mn) แบบไม่โชว์เส้น แล้วสร้างเส้นบน (+Mn) แล้วระบายสีใส่กัน
    
    # เส้นขอบล่าง (Negative Capacity) - Invisible
    fig.add_trace(go.Scatter(
        x=[x.min(), x.max()], 
        y=[-abs(phi_Mn_neg), -abs(phi_Mn_neg)],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # เส้นขอบบน (Positive Capacity) - Fill ลงไปหาเส้นล่าง
    fig.add_trace(go.Scatter(
        x=[x.min(), x.max()], 
        y=[phi_Mn_pos, phi_Mn_pos],
        mode='lines',
        line=dict(width=0),
        fill='tonexty', # ระบายสีลงไปหาเส้นล่าง
        fillcolor='rgba(46, 204, 113, 0.2)', # สีเขียวจางๆ
        name='Safe Zone (Capacity)',
        hoverinfo='skip'
    ))

    # เส้นขอบ Capacity (เส้นประ) เพื่อให้เห็นขอบเขตชัดเจน
    fig.add_trace(go.Scatter(
        x=[x.min(), x.max()], y=[phi_Mn_pos, phi_Mn_pos],
        mode='lines', line=dict(color='green', dash='dash', width=1), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[x.min(), x.max()], y=[-abs(phi_Mn_neg), -abs(phi_Mn_neg)],
        mode='lines', line=dict(color='green', dash='dash', width=1), showlegend=False
    ))

    # --- 2. Plot Demand (Mu) ---
    fig.add_trace(go.Scatter(
        x=x, y=mu_kNm, 
        mode='lines', 
        name='Applied Moment (Mu)',
        line=dict(color='#2980B9', width=3) # สีน้ำเงินเข้ม
    ))
    
    # --- Annotations (Max Values) ---
    max_mu = mu_kNm.max()
    min_mu = mu_kNm.min()
    
    if max_mu > 0.05:
        fig.add_annotation(x=df_span.loc[mu_kNm.idxmax(), 'x'], y=max_mu, text=f"{max_mu:.2f}", showarrow=True, arrowhead=1, yshift=10)
    if min_mu < -0.05:
        fig.add_annotation(x=df_span.loc[mu_kNm.idxmin(), 'x'], y=min_mu, text=f"{min_mu:.2f}", showarrow=True, arrowhead=1, ay=30)

    # Layout
    fig.update_layout(
        title="<b>Capacity Check:</b> Is Blue Line inside Green Zone?",
        xaxis_title="Distance (m)",
        yaxis_title="Moment (kNm)",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.05),
        hovermode="x unified"
    )
    return fig

# ==========================================
# 2. ANALYSIS DIAGRAMS (แก้ไข Support & Grid Lines)
# ==========================================
def draw_interactive_diagrams(df, reac, spans, sup_df, loads, dl_factor=1.4, ll_factor=1.7):
    # --- Data Prep ---
    df_plot = df.copy()
    df_plot['shear_kn'] = df_plot['shear'] / 1000.0
    df_plot['moment_knm'] = df_plot['moment'] / 1000.0
    df_plot['deflection_mm'] = df_plot['deflection'] * 1000.0
    df_plot['moment_plot'] = -df_plot['moment_knm'] # Tension Side

    cum_spans = [0] + list(np.cumsum(spans))
    total_len = cum_spans[-1] if cum_spans else 0
    
    # Load Prep
    clean_loads = []
    if loads:
        for l in loads:
            try:
                span_idx = int(l.get('span_index', 0))
                local_x = float(l.get('x', 0))
                abs_x = cum_spans[span_idx] + local_x if span_idx < len(cum_spans)-1 else local_x
                clean_loads.append({
                    'mag': float(l.get('mag', 0)),
                    'global_x': abs_x,
                    'type': str(l.get('type', 'P')),
                    'case': str(l.get('case', 'DL')),
                    'dist': float(l.get('dist', 0))
                })
            except: continue

    # --- Setup Subplots ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.06,
        subplot_titles=("Structure Model", "Shear (kN)", "Moment (kNm)", "Deflection (mm)"),
        row_heights=[0.20, 0.25, 0.25, 0.30] # เพิ่มความสูง row 1 เพื่อให้ Support มีที่หายใจ
    )

    # ==========================
    # ROW 1: STRUCTURE
    # ==========================
    
    # 1. วาดเส้นคาน (Beam) ที่ y=0
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], line=dict(color='black', width=6), hoverinfo='skip'), row=1, col=1)
    
    # 2. Loop วาด Support และ Grid Lines
    for i, x_pos in enumerate(cum_spans):
        
        # --- A. เส้นประแนวดิ่ง (Vertical Grid) ---
        # ใช้ row='all' เพื่อลากผ่านทุกกราฟ !
        fig.add_vline(x=x_pos, line_width=1, line_dash="dash", line_color="gray", opacity=0.5, row="all", col=1)

        # --- B. Support Marker & Label ---
        # หาข้อมูล Support
        sup_type = "Pin"
        if not sup_df.empty:
            match = sup_df[sup_df['id'] == i]
            if not match.empty:
                sup_type = match.iloc[0]['type']

        # เลือกรูปทรง
        stype = sup_type.lower()
        if 'pin' in stype:
            sym, col = 'triangle-up', '#2C3E50'
        elif 'roller' in stype:
            sym, col = 'circle', '#27AE60'
        elif 'fix' in stype:
            sym, col = 'square', '#000000'
        else:
            sym, col = 'triangle-up', 'gray'

        # สร้าง Label (เช่น Pin, R=50.2)
        label_text = sup_type.capitalize()
        if reac and i in reac:
             r_val = reac[i] / 1000.0 # Convert to kN
             if abs(r_val) > 0.01:
                 label_text += f"<br>R={r_val:.2f}"

        # Plot จุด Support (ขยับ y ลงมาหน่อย จะได้ดูเหมือนรองรับอยู่ข้างล่าง)
        fig.add_trace(go.Scatter(
            x=[x_pos], y=[-0.15], 
            mode='markers+text', 
            marker=dict(symbol=sym, size=22, color=col, line=dict(width=2, color='black')), 
            text=[label_text],
            textposition="bottom center", # ให้ตัวหนังสืออยู่ใต้รูป
            textfont=dict(size=11, color='#333'),
            hoverinfo="text",
            name=sup_type
        ), row=1, col=1)

    # 3. วาด Loads
    for l in clean_loads:
        color = "#C0392B" if l['case'] == 'LL' else "#7F8C8D"
        if l['type'] == 'P':
            # Arrow Load
            fig.add_annotation(
                x=l['global_x'], y=0, ax=0, ay=-50, arrowhead=2, arrowwidth=2, arrowcolor=color,
                text=f"{l['mag']}", font=dict(color=color, size=11, family="Arial Black"), row=1, col=1
            )
        elif l['type'] == 'U':
            # UDL Block
            x_end = l['global_x'] + l['dist']
            fig.add_shape(type="rect", x0=l['global_x'], x1=x_end, y0=0.05, y1=0.3, fillcolor=color, opacity=0.3, line_width=0, row=1, col=1)
            fig.add_annotation(x=(l['global_x']+x_end)/2, y=0.35, text=f"{l['mag']}", showarrow=False, font=dict(color=color, size=10), row=1, col=1)

    # ==========================
    # ROW 2-4: DIAGRAMS
    # ==========================
    
    # Shear
    fig.add_trace(go.Scatter(x=df_plot['x'], y=df_plot['shear_kn'], fill='tozeroy', line=dict(color='#D35400'), name="Shear"), row=2, col=1)
    v_max, v_min = df_plot['shear_kn'].max(), df_plot['shear_kn'].min()
    fig.add_annotation(x=df_plot.loc[df_plot['shear_kn'].idxmax(), 'x'], y=v_max, text=f"{v_max:.2f}", showarrow=False, yshift=10, row=2, col=1)
    fig.add_annotation(x=df_plot.loc[df_plot['shear_kn'].idxmin(), 'x'], y=v_min, text=f"{v_min:.2f}", showarrow=False, yshift=-10, row=2, col=1)

    # Moment
    fig.add_trace(go.Scatter(x=df_plot['x'], y=df_plot['moment_plot'], fill='tozeroy', line=dict(color='#2980B9'), name="Moment"), row=3, col=1)
    m_sag, m_hog = df_plot['moment_knm'].max(), df_plot['moment_knm'].min()
    if m_sag > 0.05:
        fig.add_annotation(x=df_plot.loc[df_plot['moment_knm'].idxmax(), 'x'], y=-m_sag, text=f"{m_sag:.2f}", arrowhead=1, ay=30, row=3, col=1)
    if m_hog < -0.05:
        fig.add_annotation(x=df_plot.loc[df_plot['moment_knm'].idxmin(), 'x'], y=-m_hog, text=f"{m_hog:.2f}", arrowhead=1, ay=-30, row=3, col=1)

    # Deflection
    fig.add_trace(go.Scatter(x=df_plot['x'], y=df_plot['deflection_mm'], fill='tozeroy', line=dict(color='#27AE60'), name="Deflection"), row=4, col=1)
    d_max_idx = df_plot['deflection_mm'].abs().idxmax()
    d_val = df_plot.loc[d_max_idx, 'deflection_mm']
    if abs(d_val) > 0.001:
        fig.add_annotation(x=df_plot.loc[d_max_idx, 'x'], y=d_val, text=f"{d_val:.2f}", arrowhead=1, ay=30 if d_val<0 else -30, row=4, col=1)

    # ==========================
    # FINAL LAYOUT & AXIS FIX
    # ==========================
    fig.update_layout(height=900, showlegend=False, template="plotly_white", hovermode="x unified")
    
    # *** KEY FIX: ปรับ Range แกน Y ของรูปแรกให้ Support ไม่ลอย ***
    # ให้ y=0 อยู่ค่อนไปทางบน (0.4) และข้างล่างเปิดกว้าง (-0.6) ไว้ใส่ Text Support
    fig.update_yaxes(range=[-0.6, 0.4], showticklabels=False, row=1, col=1, title="Model")
    
    # Labels
    fig.update_yaxes(title_text="V (kN)", row=2, col=1)
    fig.update_yaxes(title_text="M (kNm)", row=3, col=1)
    fig.update_yaxes(title_text="δ (mm)", row=4, col=1)
    fig.update_xaxes(title_text="Distance (m)", row=4, col=1)

    return fig
