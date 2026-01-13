import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. HELPER: LOAD TABLE (ตาราง Load Combination)
# ==========================================
def render_load_table(params):
    """
    แสดงตาราง Load Combination แยกออกมาเพื่อให้เรียกใช้ง่ายและจัด layout ได้สวยงาม
    """
    st.markdown("### 📋 Design Load Parameters")
    
    dl_f = params.get('dl_factor', 1.4)
    ll_f = params.get('ll_factor', 1.7)
    inc_sw = params.get('include_sw', True)
    
    data = [
        {
            "Load Case": "Dead Load (DL)", 
            "Factor": f"{dl_f:.2f}", 
            "Description": "Superimposed Dead Load"
        },
        {
            "Load Case": "Live Load (LL)", 
            "Factor": f"{ll_f:.2f}", 
            "Description": "Live Load (Occupancy)"
        }
    ]
    
    if inc_sw:
        data.insert(0, {
            "Load Case": "Self-Weight (SW)", 
            "Factor": f"{dl_f:.2f}", 
            "Description": "Beam Self-Weight (approx. 2400 kg/m³)"
        })
        
    df = pd.DataFrame(data)
    
    st.dataframe(
        df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Load Case": st.column_config.TextColumn("Case", width="small"),
            "Factor": st.column_config.TextColumn("Safety Factor", width="small"),
            "Description": st.column_config.TextColumn("Detail", width="large")
        }
    )
    st.caption(f"ℹ️ **Ultimate Load Equation:** U = {dl_f}DL + {ll_f}LL")
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
        
        # Concrete
        total_concrete_vol += b_m * h_m * L
        # Formwork
        total_formwork_area += (2 * h_m + b_m) * L
        
        # Steel
        w_span = 0.0
        def calc_w(n, db, length): return n * (db**2 / 162) * length if n > 0 else 0

        # Main Bars
        if 'top' in res and 'all_layers' in res['top']:
             for l in res['top']['all_layers']: w_span += calc_w(l['n'], l['db'], L * 1.1)
        if 'bot' in res and 'all_layers' in res['bot']:
             for l in res['bot']['all_layers']: w_span += calc_w(l['n'], l['db'], L * 1.1)

        # Stirrups
        stir_db = res.get('shear', {}).get('db', 6)
        stir_s = (res.get('shear', {}).get('s', 200)) / 1000.0
        if stir_s > 0:
            n_stir = int(L / stir_s) + 1
            len_stir = 2 * (b_m + h_m)
            w_span += n_stir * (stir_db**2 / 162) * len_stir
            
        total_steel_weight += w_span

    data = [
        {"Item": "Concrete (240 ksc)", "Quantity": float(f"{total_concrete_vol:.2f}"), "Unit": "m³"},
        {"Item": "Formwork", "Quantity": float(f"{total_formwork_area:.2f}"), "Unit": "m²"},
        {"Item": "Rebar (DB+RB)", "Quantity": float(f"{total_steel_weight:.2f}"), "Unit": "kg"}
    ]
    return pd.DataFrame(data)

# ==========================================
# 3. PLOTLY ANALYSIS GRAPH (PROFESSIONAL FBD)
# ==========================================
def plot_analysis_results(res_df, spans, supports, loads, reactions):
    """
    Standard Engineering FBD:
    - Uses Pixel-Based Scaling for arrows (Fixes distortion).
    - Uses Vertical Stacking (Point loads ride on top of UDLs).
    """
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>1. Free Body Diagram (FBD)</b>", 
            "<b>2. Shear Force Diagram (SFD)</b>", 
            "<b>3. Bending Moment Diagram (BMD)</b>",
            "<b>4. Deflection Diagram</b>"
        ),
        row_heights=[0.3, 0.23, 0.23, 0.24]
    )

    # --- ROW 1: FREE BODY DIAGRAM ---
    total_L = sum(spans)
    cum_dist = [0] + list(np.cumsum(spans))
    
    # 1.1 Beam Line
    fig.add_trace(go.Scatter(
        x=[0, total_L], y=[0, 0], 
        mode='lines', line=dict(color='black', width=6), 
        hoverinfo='skip'
    ), row=1, col=1)
    
    # 1.2 Supports
    for idx, row in supports.iterrows():
        sym = "triangle-up"
        if row['type'] == 'Fixed': sym = "square"
        elif row['type'] == 'Roller': sym = "circle"
        
        fig.add_trace(go.Scatter(
            x=[row['x']], y=[-0.05], 
            mode='markers+text',
            marker=dict(symbol=sym, size=14, color='white', line=dict(width=2, color='black')),
            text=[row['type'][0]], textposition="bottom center",
            hoverinfo='name', name="Support"
        ), row=1, col=1)

    # 1.3 Loads (Fix: Robust Data Conversion)
    if isinstance(loads, pd.DataFrame):
        load_list = loads.to_dict('records')
    elif isinstance(loads, list):
        load_list = loads
    else:
        load_list = []
    
    # 1.4 Visual Constants
    UDL_VISUAL_H = 0.5  # ความสูงสมมติของ Block UDL ในกราฟ
    ARROW_PX = 50       # ความยาวลูกศร Point Load (Pixel)
    
    # --- STEP 1: วาด UDL ก่อน (Layer ล่าง) ---
    for l in load_list:
        if l['type'] == 'U':
            span_idx = int(l['span_index'])
            start_x = cum_dist[span_idx] + float(l.get('d_start', 0))
            dist_val = float(l['dist'])
            end_x = start_x + dist_val
            
            mag_label = l['mag'] / 1000.0
            color = '#c0392b' if l.get('case') == 'LL' else '#2980b9'
            
            # วาด Block
            fig.add_trace(go.Scatter(
                x=[start_x, end_x, end_x, start_x], 
                y=[0, 0, UDL_VISUAL_H, UDL_VISUAL_H],
                fill='toself', fillcolor=color, opacity=0.15, line=dict(width=0),
                hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            
            # เส้นขอบบน
            fig.add_trace(go.Scatter(
                x=[start_x, end_x], y=[UDL_VISUAL_H, UDL_VISUAL_H],
                mode='lines', line=dict(color=color, width=1.5), hoverinfo='skip'
            ), row=1, col=1)
            
            # ลูกศรย่อยๆ
            n_arrows = max(2, int(dist_val * 1.5))
            for ax_x in np.linspace(start_x, end_x, n_arrows + 2)[1:-1]:
                fig.add_annotation(
                    x=ax_x, y=0, ax=0, ay=-30, ayref='pixel', # Fixed pixel size
                    xref="x1", yref="y1", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor=color,
                    row=1, col=1
                )
            
            # Label ตรงกลาง
            label_txt = f"<b>w={mag_label:.2f}</b>"
            if l.get('case') == 'SW': label_txt = f"SW={mag_label:.2f}"
            fig.add_annotation(
                x=(start_x+end_x)/2, y=UDL_VISUAL_H, 
                text=label_txt, showarrow=False, yshift=15, 
                font=dict(color=color, size=10), row=1, col=1
            )

    # --- STEP 2: วาด Point Load ทีหลัง (Layer บน) ---
    # ตามตำรา: ถ้ามี UDL ให้วาด Point Load เริ่มที่ความสูงของ UDL (Stacking)
    for l in load_list:
        if l['type'] == 'P':
            span_idx = int(l['span_index'])
            x_loc = cum_dist[span_idx] + float(l['d_start'])
            mag_label = l['mag'] / 1000.0
            color = '#c0392b' if l.get('case') == 'LL' else '#2980b9'
            
            # Check overlap with UDL logic (Simplified: Always lift a bit to be safe/consistent)
            # หรือถ้ารู้ว่าตรงนี้มี UDL ให้ยกขึ้น แต่เพื่อความง่าย เราจะยก Point Load ขึ้นไปที่ระดับ UDL เลย
            # เพื่อให้เป็นมาตรฐานเดียวกันว่า "Point Load อยู่ชั้นบน"
            
            y_base = UDL_VISUAL_H  # ยกหางลูกศรไปเริ่มที่ความสูง UDL
            
            fig.add_annotation(
                x=x_loc, 
                y=0, # หัวลูกศรชี้ลงที่คานเหมือนเดิม (ถูกต้องทาง Physics)
                # แต่เราจะวาด "ก้าน" ให้ยาวทะลุ UDL ลงมา หรือวาดแค่ส่วนบน?
                # ตามตำรา FBD: ลูกศรควรพุ่งลงมาที่คาน แต่ถ้ามี UDL ขวาง ให้วาดทับได้เลย
                # แต่เพื่อให้ "เห็นชัด" (Visibility) เราจะขยับ *หาง* ขึ้นไปสูงๆ
                
                ax=0, ay=-(ARROW_PX + 30), # เพิ่มความยาวหางลูกศรให้พ้น UDL (50+30 px)
                ayref='pixel',
                xref="x1", yref="y1",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=color,
                
                text=f"<b>P={mag_label:.2f}</b>", 
                yshift=(ARROW_PX + 35), # ขยับ Text ตามขึ้นไป
                font=dict(color=color, size=11, family="Arial Black"), 
                row=1, col=1
            )

    # --- ROW 2-4: DIAGRAMS (Standard) ---
    # Shear
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['shear']/1000, 
        mode='lines', name='Shear', line=dict(color='#e74c3c', width=2),
        fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'
    ), row=2, col=1)
    
    v_vals = res_df['shear']/1000
    for val in [v_vals.max(), v_vals.min()]:
        if abs(val) > 0.01:
            idx = (v_vals - val).abs().idxmin()
            fig.add_annotation(x=res_df['x'].iloc[idx], y=val, text=f"<b>{val:.2f}</b>", showarrow=False, yshift=10 if val>0 else -10, font=dict(color='#e74c3c', size=11), row=2, col=1)

    # Moment
    fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['moment']/1000, 
        mode='lines', name='Moment', line=dict(color='#27ae60', width=2),
        fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.1)'
    ), row=3, col=1)
    
    m_vals = res_df['moment']/1000
    for val in [m_vals.max(), m_vals.min()]:
        if abs(val) > 0.01:
            idx = (m_vals - val).abs().idxmin()
            fig.add_annotation(x=res_df['x'].iloc[idx], y=val, text=f"<b>{val:.2f}</b>", showarrow=True, arrowhead=1, ay=20 if val>0 else -20, font=dict(color='#27ae60', size=11), row=3, col=1)

    # Deflection
    fig.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['deflection'], 
        mode='lines', name='Deflection', line=dict(color='#8e44ad', width=2)
    ), row=4, col=1)
    
    if not res_df['deflection'].empty:
        idx_max_def = res_df['deflection'].abs().idxmax()
        max_def = res_df['deflection'].iloc[idx_max_def]
        if abs(max_def) > 0.001:
             fig.add_annotation(x=res_df['x'].iloc[idx_max_def], y=max_def, text=f"<b>Max: {max_def:.2f} mm</b>", showarrow=True, arrowhead=1, ay=30 if max_def < 0 else -30, font=dict(color='#8e44ad', size=11), row=4, col=1)

    # --- LAYOUT ---
    for x_pos in cum_dist:
        fig.add_vline(x=x_pos, line_width=1, line_dash="dash", line_color="gray", opacity=0.3)

    fig.update_layout(
        height=1000, showlegend=False, template="plotly_white", hovermode="x unified",
        margin=dict(t=50, b=40, l=60, r=20)
    )
    
    # [PROFESSIONAL FIX] Lock FBD Scale & Headroom
    # ให้พื้นที่ด้านบน (1.5 - 2.0) สำหรับหัวลูกศร Point Load ที่ยกสูงขึ้น
    fig.update_yaxes(range=[-0.5, 2.0], showgrid=False, visible=False, row=1, col=1)
    
    fig.update_yaxes(title_text="Shear (kN)", showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="Moment (kNm)", autorange="reversed", showgrid=True, row=3, col=1)
    fig.update_yaxes(title_text="Def. (mm)", showgrid=True, row=4, col=1)
    fig.update_xaxes(title_text="Length (m)", row=4, col=1)

    return fig
