import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def draw_interactive_diagrams(df, reac, spans, sup_df, loads, dl_factor=1.4, ll_factor=1.7):
    
    # --- 0. Setup ---
    st.markdown("### ⚙️ Design Criteria & Load Combination")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1: st.metric(label="Dead Load Factor (DL)", value=f"{dl_factor:.2f}")
    with col2: st.metric(label="Live Load Factor (LL)", value=f"{ll_factor:.2f}")
    with col3: st.info(f"**Factored Load Analysis:**\n\nAll results (V, M, Deflection) shown below include safety factors.")
    st.markdown("---")

    # Data Sanitization
    if isinstance(spans, (pd.DataFrame, pd.Series)): spans_val = spans.values.flatten().tolist()
    elif isinstance(spans, list): spans_val = spans
    else: spans_val = []

    cum_spans = [0] + list(np.cumsum(spans_val))
    total_len = cum_spans[-1] if cum_spans else 0
    priority_x = set(cum_spans)
    clean_loads = []
    
    if loads is not None:
        if isinstance(loads, pd.DataFrame): loads_data = loads.to_dict('records')
        else: loads_data = loads
        
        for l in loads_data:
            try:
                # Resolve Position
                span_idx = int(l.get('span_index', l.get('span_idx', 0)))
                local_x = float(l.get('x', 0))
                # ถ้า User ใส่มาแล้วยังไม่ได้แปลงเป็น Global (กรณีมาจาก list state)
                if span_idx < len(cum_spans) - 1:
                    abs_x = cum_spans[span_idx] + local_x
                else:
                    abs_x = local_x
                    
                priority_x.add(abs_x)
                if str(l.get('type')).upper().startswith('U'):
                    priority_x.add(abs_x + float(l.get('dist', 0)))

                clean_loads.append({
                    'span_idx': span_idx,
                    'mag': float(l.get('mag', 0)),
                    'x': local_x,
                    'global_x': abs_x,
                    'type': str(l.get('type', 'P')),
                    'case': str(l.get('case', 'DL')),
                    'dist': float(l.get('dist', 0))
                })
            except: continue
    
    for i, slen in enumerate(spans_val):
        priority_x.add(cum_spans[i] + slen/2)

    # --- 1. Load List Table ---
    st.markdown("### 📋 Applied Loads List (Unfactored Input)")
    if len(clean_loads) > 0:
        load_table_data = []
        for i, l in enumerate(clean_loads):
            span_num = l['span_idx'] + 1
            mag = l['mag']
            factored_mag = mag * (dl_factor if l['case'] == 'DL' else ll_factor)
            
            if l['type'] == 'P':
                type_lbl = "Point (P)"
                pos_lbl = f"@ x = {l['x']:.2f} m (Span {span_num})"
            elif l['type'] == 'U':
                type_lbl = "Uniform (w)"
                x_end = l['x'] + l['dist']
                pos_lbl = f"From x={l['x']:.2f} to {x_end:.2f} m"
            elif l['type'] == 'M':
                type_lbl = "Moment (M)"
                pos_lbl = f"@ x = {l['x']:.2f} m"
            
            load_table_data.append([i+1, type_lbl, l['case'], f"{mag}", f"{factored_mag:.2f}", pos_lbl])
        
        st.table(pd.DataFrame(load_table_data, columns=["No.", "Type", "Case", f"Service Load", f"Factored Load", "Position Detail"]))
    else: 
        st.info("No loads applied yet.")

    if df is None or df.empty: return

    # Fix Column Names
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    st.markdown("---")
    st.markdown("### 📊 Structural Analysis Diagrams (Ultimate Limit State)")

    # Auto-Scale
    max_defl = df['deflection'].abs().max() if not df['deflection'].empty else 0
    defl_unit = "m"
    defl_scale = 1.0
    if max_defl > 0 and max_defl < 0.01:
        defl_unit = "mm"
        defl_scale = 1000.0
    df['deflection_plot'] = df['deflection'] * defl_scale

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("<b>Structure Model</b>", f"<b>Shear Force (Vu)</b>", f"<b>Bending Moment (Mu)</b>", f"<b>Deflection (δ)</b>"),
        row_heights=[0.20, 0.26, 0.26, 0.28]
    )

    # --- Row 1: Structure ---
    if isinstance(sup_df, list): sup_df = pd.DataFrame(sup_df)
    sup_map = {}
    # สร้าง Map จาก Global X -> Type (เพราะ Solver อาจแก้ x มาแล้ว)
    # แต่เพื่อความง่าย ใช้ Logic Node ID เดิมจาก app.py
    for s in sup_df.to_dict('records'):
        if 'id' in s: sup_map[int(s['id'])] = s.get('type')
        
    for i, x in enumerate(cum_spans):
        fig.add_annotation(x=x, y=-0.55, text=f"Node {i+1}", showarrow=False, font=dict(size=10, color="gray"), row=1, col=1)
        stype = sup_map.get(i, None)
        if stype == 'Fixed':
            fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=-0.3, line=dict(width=4, color='black'), row=1, col=1)
            fig.add_shape(type="line", x0=x-0.15, y0=-0.3, x1=x+0.15, y1=-0.3, line=dict(width=4, color='black'), row=1, col=1)
        elif stype == 'Pin':
            fig.add_trace(go.Scatter(x=[x], y=[-0.15], mode='markers', marker=dict(symbol='triangle-up', size=18, color='white', line=dict(color='black', width=2)), showlegend=False, hoverinfo='skip'), row=1, col=1)
        elif stype == 'Roller':
            fig.add_trace(go.Scatter(x=[x], y=[-0.15], mode='markers', marker=dict(symbol='circle', size=18, color='white', line=dict(color='black', width=2)), showlegend=False, hoverinfo='skip'), row=1, col=1)

    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], line=dict(color='black', width=5), hoverinfo='skip', showlegend=False), row=1, col=1)

    # Loads
    for l in clean_loads:
        l_color = "#E74C3C" if l['case'] == 'LL' else "#555555"
        if l['type'] == 'P':
            fig.add_annotation(x=l['global_x'], y=0, ax=0, ay=-50, arrowhead=2, arrowwidth=2, arrowcolor=l_color, text=f"P={l['mag']}", font=dict(color=l_color, size=10), yshift=10, row=1, col=1)
        elif l['type'] == 'U':
            x_end = l['global_x'] + l['dist']
            fig.add_shape(type="rect", x0=l['global_x'], x1=x_end, y0=0.05, y1=0.20, fillcolor=l_color, opacity=0.15, line_width=0, row=1, col=1)
            fig.add_shape(type="line", x0=l['global_x'], y0=0.20, x1=x_end, y1=0.20, line=dict(color=l_color, width=2), row=1, col=1)
            fig.add_annotation(x=(l['global_x']+x_end)/2, y=0.20, text=f"w={l['mag']}", showarrow=False, yshift=10, font=dict(color=l_color, size=10), row=1, col=1)

    # --- SMART LABELS (NO ARROWS) ---
    def add_eng_labels(x_data, y_data, row_idx, color_code, unit_suffix, invert_sign=False):
        y_arr = np.array(y_data, dtype=float)
        x_arr = np.array(x_data, dtype=float)
        if len(y_arr) == 0: return
        
        # Determine Max/Min
        val_max = np.max(y_arr)
        val_min = np.min(y_arr)
        
        # Display Logic: Show Max Amplitude (Pos or Neg)
        # Or show both Max Pos and Max Neg if significant
        
        indices_to_show = []
        if abs(val_max) > 1e-4: indices_to_show.append(np.argmax(y_arr))
        if abs(val_min) > 1e-4: indices_to_show.append(np.argmin(y_arr))
        
        # Style: NO ARROW, Box Background
        style = dict(
            showarrow=False,  # 3. เอาลูกศรออกตามที่ขอ
            font=dict(color=color_code, size=11),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=color_code, borderwidth=1, borderpad=3
        )

        seen_x = set()
        for idx in indices_to_show:
            if idx in seen_x: continue
            seen_x.add(idx)
            
            x_val = x_arr[idx]
            y_val = y_arr[idx]
            
            # Smart Snap X for display
            disp_x = x_val
            if priority_x:
                closest_p = min(priority_x, key=lambda p: abs(x_val - p))
                if abs(x_val - closest_p) < 0.05: disp_x = closest_p
            
            val_display = -y_val if invert_sign else y_val
            txt = f"<b>{val_display:.2f}</b><br><span style='font-size:9px'>@ {disp_x:.2f}m</span>"
            
            # Shift Y slightly to not cover the line
            y_shift = 15 if y_val >= 0 else -15
            if invert_sign: y_shift *= -1
            
            fig.add_annotation(x=x_val, y=y_val if not invert_sign else -y_val, text=txt, yshift=y_shift, row=row_idx, col=1, **style)

    # Row 2: Shear
    c_shear = '#E67E22'
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color=c_shear, width=2, shape='hv'), name="Shear"), row=2, col=1) # shape='hv' helps visual but data is now dense enough
    add_eng_labels(df['x'], df['shear'], 2, c_shear, "N")

    # Row 3: Moment
    c_moment = '#2980B9'
    moment_plot = -df['moment']
    fig.add_trace(go.Scatter(x=df['x'], y=moment_plot, fill='tozeroy', line=dict(color=c_moment, width=2), name="Moment"), row=3, col=1)
    add_eng_labels(df['x'], df['moment'], 3, c_moment, "Nm", invert_sign=True)

    # Row 4: Deflection
    c_defl = '#27AE60'
    fig.add_trace(go.Scatter(x=df['x'], y=df['deflection_plot'], fill='tozeroy', line=dict(color=c_defl, width=2), name="Deflection"), row=4, col=1)
    
    # Custom Label for Deflection (Max Abs)
    if not df.empty:
        y_scaled = df['deflection_plot'].values
        idx_max = np.argmax(np.abs(y_scaled))
        max_val = y_scaled[idx_max]
        if abs(max_val) > 1e-5:
             x_max = df['x'].values[idx_max]
             txt = f"<b>Max: {max_val:.3f} {defl_unit}</b><br>@ {x_max:.2f}m"
             fig.add_annotation(
                x=x_max, y=max_val, text=txt, 
                showarrow=False, yshift=20 if max_val>0 else -20,
                font=dict(color=c_defl, size=11), bgcolor="rgba(255,255,255,0.9)", bordercolor=c_defl, borderwidth=1,
                row=4, col=1
            )

    # Grid
    for node_x in cum_spans:
        for r in [2,3,4]: fig.add_vline(x=node_x, line_dash="dash", line_color="gray", opacity=0.3, row=r, col=1)

    fig.update_layout(height=900, showlegend=False, template="plotly_white", margin=dict(l=50, r=20, t=40, b=40), hovermode="x unified")
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_yaxes(title_text="Shear (N)", row=2, col=1)
    fig.update_yaxes(title_text="Moment (Nm)", row=3, col=1)
    fig.update_yaxes(title_text=f"Defl ({defl_unit})", row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans):
    pass # ใช้ Table เดิมจาก app.py
