import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. HELPER: LOAD TABLE
# ==========================================
def render_load_table(params):
    st.markdown("### 📋 Design Load Parameters")
    dl_f = params.get('dl_factor', 1.4)
    ll_f = params.get('ll_factor', 1.7)
    inc_sw = params.get('include_sw', True)
    
    data = [
        {"Load Case": "Dead Load (DL)", "Factor": f"{dl_f:.2f}", "Description": "Superimposed Dead Load"},
        {"Load Case": "Live Load (LL)", "Factor": f"{ll_f:.2f}", "Description": "Live Load (Occupancy)"}
    ]
    if inc_sw:
        data.insert(0, {"Load Case": "Self-Weight (SW)", "Factor": f"{dl_f:.2f}", "Description": "Beam Self-Weight"})
        
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    st.divider()

# ==========================================
# 2. BOQ CALCULATION
# ==========================================
def calculate_boq_summary(design_res, spans):
    total_concrete_vol = 0.0
    total_formwork_area = 0.0
    total_steel_weight = 0.0
    
    for i, res in enumerate(design_res):
        L = spans[i]
        b_m = (res.get('b') or 300) / 1000.0
        h_m = (res.get('h') or 500) / 1000.0
        
        total_concrete_vol += b_m * h_m * L
        total_formwork_area += (2 * h_m + b_m) * L
        
        w_span = 0.0
        def calc_w(n, db, length): return n * (db**2 / 162) * length if n > 0 else 0

        if 'top' in res and 'all_layers' in res['top']:
             for l in res['top']['all_layers']: w_span += calc_w(l['n'], l['db'], L * 1.1)
        if 'bot' in res and 'all_layers' in res['bot']:
             for l in res['bot']['all_layers']: w_span += calc_w(l['n'], l['db'], L * 1.1)

        stir_db = res.get('shear', {}).get('db', 6)
        stir_s = (res.get('shear', {}).get('s', 200)) / 1000.0
        if stir_s > 0:
            n_stir = int(L / stir_s) + 1
            len_stir = 2 * (b_m + h_m)
            w_span += n_stir * (stir_db**2 / 162) * len_stir
            
        total_steel_weight += w_span

    return pd.DataFrame([
        {"Item": "Concrete (240 ksc)", "Quantity": float(f"{total_concrete_vol:.2f}"), "Unit": "m³"},
        {"Item": "Formwork", "Quantity": float(f"{total_formwork_area:.2f}"), "Unit": "m²"},
        {"Item": "Rebar (DB+RB)", "Quantity": float(f"{total_steel_weight:.2f}"), "Unit": "kg"}
    ])

# ==========================================
# 3. PLOTLY ANALYSIS GRAPH (DATA COORDINATE SYSTEM)
# ==========================================
def plot_analysis_results(res_df, spans, supports, loads, reactions):
    """
    Final Corrected FBD:
    - Uses 'Data Coordinates' (ayref='y') instead of pixels.
    - Guaranteed contact with beam (y=0).
    - Perfect relative scaling between UDL and Point Load.
    """
    
    # --- 1. DATA PREP & SCALING FACTOR ---
    if isinstance(loads, pd.DataFrame):
        load_list = loads.to_dict('records')
    elif isinstance(loads, list):
        load_list = loads
    else:
        load_list = []

    # หาค่าแรงสูงสุดเพื่อกำหนดสเกลความสูงกราฟ (Scaling Base)
    # ทั้ง UDL และ Point Load จะถูกหารด้วยค่านี้เพื่อให้สัดส่วนถูกต้อง
    all_mags = [l['mag'] for l in load_list] if load_list else [1]
    max_load_val = max(all_mags) if all_mags else 1.0
    if max_load_val == 0: max_load_val = 1.0
    
    # กำหนดความสูงสูงสุดของกราฟ (Visual Height Limit)
    VISUAL_Y_MAX = 2.0 

    # --- 2. SETUP PLOT ---
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("<b>1. Free Body Diagram (FBD)</b>", "<b>2. Shear Force Diagram (SFD)</b>", "<b>3. Bending Moment Diagram (BMD)</b>", "<b>4. Deflection Diagram</b>"),
        row_heights=[0.35, 0.22, 0.22, 0.21]
    )

    total_L = sum(spans)
    cum_dist = [0] + list(np.cumsum(spans))
    
    # Beam (y=0)
    fig.add_trace(go.Scatter(x=[0, total_L], y=[0, 0], mode='lines', line=dict(color='black', width=5), hoverinfo='skip'), row=1, col=1)
    
    # Supports
    for idx, row in supports.iterrows():
        sym = "square" if row['type'] == 'Fixed' else ("circle" if row['type'] == 'Roller' else "triangle-up")
        fig.add_trace(go.Scatter(
            x=[row['x']], y=[-0.05], mode='markers+text',
            marker=dict(symbol=sym, size=12, color='white', line=dict(width=2, color='black')),
            text=[row['type'][0]], textposition="bottom center", hoverinfo='name', name="Support"
        ), row=1, col=1)

    # --- 3. DRAW LOADS (SCALED BY DATA) ---
    
    # LAYER 1: UDL
    for l in load_list:
        if l['type'] == 'U':
            span_idx = int(l['span_index'])
            start_x = cum_dist[span_idx] + float(l.get('d_start', 0))
            end_x = start_x + float(l['dist'])
            mag = l['mag']
            
            # คำนวณความสูงตามสัดส่วนจริง
            # ขั้นต่ำ 0.4 หน่วย เพื่อให้มองเห็นชัดแม้แรงน้อย
            ratio = mag / max_load_val
            h_visual = 0.4 + (ratio * (VISUAL_Y_MAX - 0.4)) * 0.6  # ปรับตัวคูณ 0.6 เพื่อให้ UDL เตี้ยกว่า Point Load ที่แรงเท่ากันนิดหน่อยตามธรรมเนียม
            
            color = '#e74c3c' if l.get('case') == 'LL' else '#2980b9'
            
            # Draw Block
            fig.add_trace(go.Scatter(
                x=[start_x, end_x, end_x, start_x], y=[0, 0, h_visual, h_visual],
                fill='toself', fillcolor=color, opacity=0.15, line=dict(width=0), hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            # Top Line
            fig.add_trace(go.Scatter(
                x=[start_x, end_x], y=[h_visual, h_visual],
                mode='lines', line=dict(color=color, width=1, dash='dot'), hoverinfo='skip'
            ), row=1, col=1)
            # Label
            label_txt = f"w={mag/1000:.2f}" if l.get('case')!='SW' else f"SW={mag/1000:.2f}"
            fig.add_annotation(
                x=(start_x+end_x)/2, y=h_visual, text=label_txt, showarrow=False, yshift=10,
                font=dict(color=color, size=10), row=1, col=1
            )
            # Internal Arrows (Scale with block)
            n_arrows = max(3, int(float(l['dist']) * 2.0))
            for ax_x in np.linspace(start_x, end_x, n_arrows + 2)[1:-1]:
                 # ใช้ ayref='y' เพื่อยึดหางลูกศรกับความสูงกราฟ (Data Coordinates)
                 fig.add_annotation(
                    x=ax_x, y=0,           # หัวลูกศรที่คาน
                    ax=ax_x, ay=h_visual,  # หางลูกศรที่ความสูง Block
                    axref='x', ayref='y',  # *** KEY FIX: ใช้แกนข้อมูลจริง ไม่ใช่ Pixel ***
                    xref='x', yref='y',
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor=color,
                    row=1, col=1
                )

    # LAYER 2: POINT LOAD
    for l in load_list:
        if l['type'] == 'P':
            span_idx = int(l['span_index'])
            x_loc = cum_dist[span_idx] + float(l['d_start'])
            mag = l['mag']
            color = '#c0392b' if l.get('case') == 'LL' else '#2980b9'
            
            # คำนวณความสูงตามสัดส่วนจริง (Direct Ratio)
            # Point Load จะสูงกว่า UDL เสมอถ้าแรงเท่ากัน เพื่อความเด่น
            ratio = mag / max_load_val
            h_arrow = 0.5 + (ratio * (VISUAL_Y_MAX - 0.5))
            
            # Label Offset (อยู่เหนือหางลูกศรนิดหน่อย)
            text_y = h_arrow + 0.2
            
            # *** KEY FIX: Arrow using Data Coordinates ***
            fig.add_annotation(
                x=x_loc, y=0,          # หัวลูกศรแตะคาน (y=0) เป๊ะ
                ax=x_loc, ay=h_arrow,  # หางลูกศรอยู่ที่ความสูงคำนวณ (Data Coords)
                xref='x', yref='y',    # อ้างอิงแกนกราฟ
                axref='x', ayref='y',  # อ้างอิงแกนกราฟ (ทำให้ไม่ลอย ไม่เพี้ยนเมื่อซูม)
                
                showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor=color,
                text=f"<b>P={mag/1000:.2f}</b>",
                
                # ตำแหน่ง Text
                xanchor='center', yanchor='bottom',
                yshift=10, # ขยับ Text ขึ้นจากหางนิดนึง
                font=dict(color=color, size=12, family="Arial Black"),
                row=1, col=1
            )

    # --- 4. DIAGRAMS & LAYOUT ---
    # SFD
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['shear']/1000, mode='lines', line=dict(color='#e74c3c', width=2), fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'), row=2, col=1)
    # BMD
    fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['moment']/1000, mode='lines', line=dict(color='#27ae60', width=2), fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.1)'), row=3, col=1)
    # Deflection
    fig.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['deflection'], mode='lines', line=dict(color='#8e44ad', width=2)), row=4, col=1)

    # Layout Updates
    for x_pos in cum_dist:
        fig.add_vline(x=x_pos, line_width=1, line_dash="dash", line_color="gray", opacity=0.3)

    fig.update_layout(height=1100, showlegend=False, template="plotly_white", hovermode="x unified", margin=dict(t=50, b=40, l=60, r=20))
    
    # Scale Y-Axis to fit Arrows (Important!)
    # ปรับแกน Y ให้สูงพอที่จะรับลูกศรที่ยาวที่สุดได้
    fig.update_yaxes(range=[-0.5, VISUAL_Y_MAX * 1.3], showgrid=False, visible=False, row=1, col=1)
    
    fig.update_yaxes(title_text="Shear (kN)", showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="Moment (kNm)", autorange="reversed", showgrid=True, row=3, col=1)
    fig.update_yaxes(title_text="Def. (mm)", showgrid=True, row=4, col=1)
    fig.update_xaxes(title_text="Length (m)", row=4, col=1)

    return fig
