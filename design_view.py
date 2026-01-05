import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def draw_interactive_diagrams(df, reac, spans, sup_df, loads, unit_force="kg", unit_len="m", dl_factor=1.4, ll_factor=1.7):
    
    # --- 0. Setup ---
    st.markdown("### ⚙️ Design Criteria & Load Combination")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1: st.metric(label="Dead Load Factor (DL)", value=f"{dl_factor:.2f}")
    with col2: st.metric(label="Live Load Factor (LL)", value=f"{ll_factor:.2f}")
    with col3: st.info(f"**Factored Load Analysis:**\n\nAll results (V, M, Deflection) shown below include safety factors.")
    st.markdown("---")

    # Data Sanitization
    if isinstance(spans, (pd.DataFrame, pd.Series)): 
        spans_val = spans.values.flatten().tolist()
    elif isinstance(spans, list): 
        spans_val = spans
    else: 
        spans_val = []

    cum_spans = [0] + list(np.cumsum(spans_val))
    total_len = cum_spans[-1] if cum_spans else 0

    # [Priority List Creation] for Smart Snapping
    priority_x = set(cum_spans)
    clean_loads = []
    
    if loads is not None:
        # Normalize loads to list of dicts
        if isinstance(loads, pd.DataFrame): 
            loads_data = loads.to_dict('records')
        else: 
            loads_data = loads
        
        if len(loads_data) > 0:
            for l in loads_data:
                if isinstance(l, dict):
                    try:
                        # Calculate absolute X based on span index
                        span_idx = int(l.get('span_idx', 0))
                        if span_idx < len(cum_spans):
                            abs_x = cum_spans[span_idx] + float(l.get('x', 0))
                            priority_x.add(abs_x)
                            
                            # Handle UDL End position for priority snapping
                            if str(l.get('type')) == 'U':
                                dist = float(l.get('dist', 0)) if l.get('dist') is not None else 0
                                if dist == 0 and span_idx < len(spans_val): 
                                    dist = spans_val[span_idx] - float(l.get('x', 0))
                                priority_x.add(abs_x + dist)

                            clean_loads.append({
                                'span_idx': span_idx,
                                'mag': float(l.get('mag', 0)),
                                'x': float(l.get('x', 0)),
                                'type': str(l.get('type', 'P')),
                                'case': str(l.get('case', 'DL')),
                                'dist': float(l.get('dist', 0)) if l.get('dist') is not None else None
                            })
                    except (ValueError, TypeError): 
                        continue
    
    # Add mid-spans to priority
    for i, slen in enumerate(spans_val):
        priority_x.add(cum_spans[i] + slen/2)

    # --- 1. Load List Table ---
    st.markdown("### 📋 Applied Loads List (Unfactored Input)")
    if len(clean_loads) > 0:
        load_table_data = []
        for i, l in enumerate(clean_loads):
            span_num = l['span_idx'] + 1
            mag = l['mag']
            x_local = l['x']
            l_type = l['type']
            l_case = l['case']
            
            # Calculate Factored Load
            factor = dl_factor if l_case == 'DL' else ll_factor
            factored_mag = mag * factor
            
            if l_type == 'P':
                type_lbl = "Point (P)"
                pos_lbl = f"@ x = {x_local:.2f} m (Span {span_num})"
            elif l_type == 'U':
                type_lbl = "Uniform (w)"
                dist = l['dist']
                if dist is None or dist == 0: 
                    span_len = spans_val[l['span_idx']]
                    dist = span_len - x_local
                x_end = x_local + dist
                pos_lbl = f"From x={x_local:.2f} to x={x_end:.2f} m (Len={dist:.2f})"
            elif l_type == 'M':
                type_lbl = "Moment (M)"
                pos_lbl = f"@ x = {x_local:.2f} m (Span {span_num})"
            
            load_table_data.append([i+1, type_lbl, l_case, f"{mag}", f"{factored_mag:.2f}", pos_lbl])
        
        st.table(pd.DataFrame(load_table_data, columns=["No.", "Type", "Case", f"Service Load", f"Factored Load", "Position Detail"]))
    else: 
        st.info("No loads applied yet.")

    # Check if results exist
    if df is None or (isinstance(df, pd.DataFrame) and df.empty): return

    st.markdown("---")
    st.markdown("### 📊 Structural Analysis Diagrams (Ultimate Limit State)")

    # --- 2. Auto-Scale Deflection ---
    max_defl = df['deflection'].abs().max() if not df['deflection'].empty else 0
    defl_unit = unit_len
    defl_scale = 1.0
    if max_defl > 0 and max_defl < 0.01:
        defl_unit = "mm"
        defl_scale = 1000.0
    df['deflection_plot'] = df['deflection'] * defl_scale

    # --- 3. Plot Setup ---
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("<b>Structure Model & Loads</b>", f"<b>Shear Force (Vu)</b>", f"<b>Bending Moment (Mu) - Tension Side Positive</b>", f"<b>Deflection (δ)</b>"),
        row_heights=[0.20, 0.26, 0.26, 0.28]
    )

    # --- Row 1: Structure Visualization ---
    if isinstance(sup_df, list): sup_df = pd.DataFrame(sup_df)
    sup_map = {}
    if sup_df is not None and not sup_df.empty and 'id' in sup_df.columns:
        sup_map = {int(r['id']): r['type'] for _, r in sup_df.iterrows()}

    # Draw Supports & Nodes
    for i, x in enumerate(cum_spans):
        fig.add_annotation(x=x, y=-0.55, text=f"Node {i+1}", showarrow=False, font=dict(size=10, color="gray"), row=1, col=1)
        if i in sup_map:
            stype = sup_map[i]
            if stype == 'Fixed':
                fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=-0.3, line=dict(width=4, color='black'), row=1, col=1)
                fig.add_shape(type="line", x0=x-0.15, y0=-0.3, x1=x+0.15, y1=-0.3, line=dict(width=4, color='black'), row=1, col=1)
                for h in np.linspace(x-0.15, x+0.15, 5):
                      fig.add_shape(type="line", x0=h, y0=-0.3, x1=h-0.05, y1=-0.4, line=dict(width=1, color='black'), row=1, col=1)
            elif stype == 'Pin':
                fig.add_trace(go.Scatter(x=[x], y=[-0.15], mode='markers', marker=dict(symbol='triangle-up', size=18, color='white', line=dict(color='black', width=2)), showlegend=False, hoverinfo='skip'), row=1, col=1)
                fig.add_shape(type="line", x0=x-0.2, y0=-0.25, x1=x+0.2, y1=-0.25, line=dict(width=2, color='black'), row=1, col=1)
            elif stype == 'Roller':
                fig.add_trace(go.Scatter(x=[x], y=[-0.15], mode='markers', marker=dict(symbol='circle', size=18, color='white', line=dict(color='black', width=2)), showlegend=False, hoverinfo='skip'), row=1, col=1)
                fig.add_shape(type="line", x0=x-0.2, y0=-0.30, x1=x+0.2, y1=-0.30, line=dict(width=2, color='black'), row=1, col=1)

    # Beam Line
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], line=dict(color='black', width=5), hoverinfo='skip', showlegend=False), row=1, col=1)

    # Draw Loads on Structure
    for l in clean_loads:
        x_abs_start = cum_spans[l['span_idx']] + l['x']
        mag = l['mag']; l_type = l['type']; l_case = l['case']
        l_color = "#E74C3C" if l_case == 'LL' else "#555555"
        
        if l_type == 'P':
            fig.add_annotation(x=x_abs_start, y=0, ax=0, ay=-50, arrowhead=2, arrowwidth=2, arrowcolor=l_color, text=f"P={mag}", font=dict(color=l_color, size=10, family="Arial"), yshift=10, row=1, col=1)
        elif l_type == 'U':
            span_idx = l['span_idx']; span_len = spans_val[span_idx]
            dist = l.get('dist')
            if dist is None or dist == 0: dist = span_len - l['x']
            dist = min(dist, span_len - l['x'])
            x_abs_end = x_abs_start + dist
            
            # Draw UDL Block
            fig.add_shape(type="rect", x0=x_abs_start, x1=x_abs_end, y0=0.05, y1=0.20, fillcolor=l_color, opacity=0.15, line_width=0, row=1, col=1)
            fig.add_shape(type="line", x0=x_abs_start, y0=0.20, x1=x_abs_end, y1=0.20, line=dict(color=l_color, width=2), row=1, col=1)
            mid_x = (x_abs_start + x_abs_end) / 2
            fig.add_annotation(x=mid_x, y=0.20, text=f"w={mag}", showarrow=False, yshift=10, font=dict(color=l_color, size=10), row=1, col=1)
        elif l_type == 'M':
            fig.add_annotation(x=x_abs_start, y=0, text=f"M={mag}", showarrow=True, arrowhead=1, ax=0, ay=-40, arrowcolor='purple', font=dict(size=10), row=1, col=1)

    # ==========================================
    # SMART LABELS LOGIC (Function)
    # ==========================================
    def add_eng_labels(x_data, y_data, row_idx, color_code, unit_suffix, invert_sign=False):
        y_arr = np.array(y_data, dtype=float)
        x_arr = np.array(x_data, dtype=float)
        if len(y_arr) == 0: return
        
        def get_smart_info(is_max):
            target_val = np.max(y_arr) if is_max else np.min(y_arr)
            # Invert comparison if we are inverting sign for display
            if invert_sign: 
                target_val = np.min(y_arr) if is_max else np.max(y_arr)
            
            candidates = np.where(np.isclose(y_arr, target_val, rtol=1e-4))[0]
            if len(candidates) == 0: return 0, 0 

            best_idx = candidates[0]
            snapped_x = x_arr[best_idx]
            min_dist = float('inf')
            
            # Smart Snapping Loop
            for idx in candidates:
                curr_x = x_arr[idx]
                
                # Check distance to ANY priority point
                curr_dist = 100
                best_px = curr_x
                if priority_x:
                    # Find closest priority point
                    closest_px = min(priority_x, key=lambda p: abs(curr_x - p))
                    curr_dist = abs(curr_x - closest_px)
                    best_px = closest_px
                
                # Tolerance 5cm
                if curr_dist < 0.05:
                    return idx, best_px # Return snapped X immediately if match found
                
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    best_idx = idx
                    snapped_x = curr_x # Fallback to original if not snapped
            
            return best_idx, snapped_x

        # Calculate Max/Min
        if invert_sign:
            max_idx, max_x_display = get_smart_info(is_max=False) # Logic inverted inside
            min_idx, min_x_display = get_smart_info(is_max=True)
        else:
            max_idx, max_x_display = get_smart_info(is_max=True)
            min_idx, min_x_display = get_smart_info(is_max=False)
        
        style = dict(
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
            font=dict(color=color_code, size=10),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=color_code, borderwidth=1, borderpad=2
        )

        def plot_lbl(idx, x_display, is_top):
            val = y_arr[idx]
            # Format text
            txt = f"<b>{val:.2f}</b><br><span style='font-size:9px'>@ {x_display:.2f}m</span>" if abs(val) >= 0.01 else f"<b>{val:.4f}</b>"
            
            y_plot = -val if invert_sign else val
            ay_val = -40 if is_top else 40 
            # Note: Arrow X uses real data x_arr[idx], Text shows x_display
            fig.add_annotation(x=x_arr[idx], y=y_plot, text=txt, ax=0, ay=ay_val, row=row_idx, col=1, **style)

        # Plot labels if significant
        if abs(y_arr[max_idx]) > 1e-9: plot_lbl(max_idx, max_x_display, True)
        if abs(y_arr[min_idx]) > 1e-9 and abs(x_arr[max_idx] - x_arr[min_idx]) > 0.1:
             plot_lbl(min_idx, min_x_display, False)

    # --- ROW 2: SHEAR FORCE ---
    c_shear = '#E67E22'
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color=c_shear, width=2), name="Shear"), row=2, col=1)
    add_eng_labels(df['x'], df['shear'], 2, c_shear, unit_force)

    # --- ROW 3: BENDING MOMENT ---
    c_moment = '#2980B9'
    moment_plot = -df['moment'] # Invert for tension positive
    fig.add_trace(go.Scatter(x=df['x'], y=moment_plot, fill='tozeroy', line=dict(color=c_moment, width=2), name="Moment"), row=3, col=1)
    add_eng_labels(df['x'], df['moment'], 3, c_moment, f"{unit_force}-{unit_len}", invert_sign=True)

    # --- ROW 4: DEFLECTION ---
    c_defl = '#27AE60'
    fig.add_trace(go.Scatter(x=df['x'], y=df['deflection_plot'], fill='tozeroy', line=dict(color=c_defl, width=2), name="Deflection"), row=4, col=1)
    
    # Custom Logic for Deflection Max Label
    if not df.empty:
        y_scaled = df['deflection_plot'].values
        # Use same smart logic
        target_val = np.max(np.abs(y_scaled))
        candidates = np.where(np.isclose(np.abs(y_scaled), target_val, rtol=1e-4))[0]
        
        if len(candidates) > 0:
            best_d_idx = candidates[0]
            d_x_display = df['x'].values[best_d_idx]

            for idx in candidates:
                curr_x = df['x'].values[idx]
                if priority_x:
                    closest_px = min(priority_x, key=lambda p: abs(curr_x - p))
                    curr_dist = abs(curr_x - closest_px)
                    if curr_dist < 0.05:
                        best_d_idx = idx
                        d_x_display = closest_px
                        break
            
            d_val_scaled = y_scaled[best_d_idx]
            d_x_real = df['x'].values[best_d_idx]
            
            if abs(d_val_scaled) > 1e-5:
                 fig.add_annotation(
                    x=d_x_real, y=d_val_scaled,
                    text=f"<b>Max: {d_val_scaled:.2f} {defl_unit}</b><br><span style='font-size:9px'>@ {d_x_display:.2f}m</span>",
                    showarrow=True, arrowhead=2, ax=0, ay=35 if d_val_scaled < 0 else -35,
                    font=dict(color=c_defl, size=10),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor=c_defl, borderwidth=1, borderpad=2,
                    row=4, col=1
                )

    # Grids & Styling
    for node_x in cum_spans:
        for r_idx in [2, 3, 4]:
            fig.add_vline(x=node_x, line_width=1, line_dash="dash", line_color="gray", opacity=0.5, row=r_idx, col=1)
    
    for i, span_len in enumerate(spans_val):
        mid_x = cum_spans[i] + (span_len / 2)
        for r_idx in [2, 3, 4]:
            fig.add_vline(x=mid_x, line_width=1, line_dash="dash", line_color="gray", opacity=0.5, row=r_idx, col=1)

    fig.update_layout(height=1000, showlegend=False, template="plotly_white", margin=dict(l=60, r=30, t=40, b=50), hovermode="x unified")
    fig.update_yaxes(visible=False, range=[-0.65, 0.8], row=1, col=1)
    fig.update_yaxes(title_text=f"Vu ({unit_force})", row=2, col=1)
    fig.update_yaxes(title_text=f"Mu ({unit_force}-{unit_len})", row=3, col=1)
    fig.update_yaxes(title_text=f"δ ({defl_unit})", row=4, col=1, autorange=True)
    fig.update_xaxes(title_text=f"Distance ({unit_len})", row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans, unit_force="kg", unit_len="m"):
    st.markdown("### 📍 Support Reactions (Factored)")
    reac_data = []
    if reac is not None and len(reac) > 0 and spans is not None:
        if isinstance(spans, (pd.DataFrame, pd.Series)): spans_list = spans.values.flatten().tolist()
        else: spans_list = spans
        num_nodes = len(spans_list) + 1
        for i in range(num_nodes):
            try:
                idx_ry = 2 * i; idx_mz = 2 * i + 1
                if idx_mz < len(reac):
                    ry = float(reac[idx_ry]); mz = float(reac[idx_mz])
                    if abs(ry) > 1e-4 or abs(mz) > 1e-4:
                        reac_data.append({"Node": i+1, f"Ry (Vertical) [{unit_force}]": f"{ry:.2f}", f"Mz (Moment) [{unit_force}-{unit_len}]": f"{mz:.2f}"})
            except: continue
        if len(reac_data) > 0: st.table(pd.DataFrame(reac_data))
        else: st.write("No significant reactions.")
    else: st.info("Reaction data is not available.")
