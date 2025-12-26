def create_engineering_plots(df, vis_spans, vis_supports, loads, unit_sys):
    """
    Creates professional structural engineering diagrams (Textbook Style).
    """
    cum_len = [0] + list(np.cumsum(vis_spans))
    total_len = cum_len[-1]
    
    # Define Units
    force_unit = "kg" if unit_sys == "Metric (kg/cm)" else "kN"
    dist_unit = "m"
    moment_unit = "kg-m" if unit_sys == "Metric (kg/cm)" else "kN-m"
    
    # --- 1. PRE-CALCULATE SCALING ---
    # Find max load value to scale arrows proportionally
    max_load_val = 1.0 # default baseline
    for load in loads:
        val = max(abs(load.get('w', 0)), abs(load.get('P', 0)))
        if val > max_load_val: max_load_val = val
    
    # Scale factor for visualization height
    load_plot_height = max_load_val * 1.5 

    # --- 2. CREATE SUBPLOTS ---
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        subplot_titles=("<b>Loading Diagram</b>", "<b>Shear Force Diagram (SFD)</b>", "<b>Bending Moment Diagram (BMD)</b>"),
        row_heights=[0.3, 0.35, 0.35]
    )

    # ==================================================
    # ROW 1: LOADING DIAGRAM (Textbook Style)
    # ==================================================
    
    # 1.1 Draw The Beam
    fig.add_shape(type="line", x0=0, y0=0, x1=total_len, y1=0, 
                  line=dict(color="black", width=4), row=1, col=1)
    
    # 1.2 Draw Supports (Triangles/Pins)
    for i, x in enumerate(cum_len):
        # Check support type if available, else default to Pin
        if i < len(vis_supports):
            sup_type = vis_supports.iloc[i]['type']
            if sup_type != "None":
                # Draw Triangle Support
                fig.add_trace(go.Scatter(
                    x=[x], y=[-load_plot_height*0.05], 
                    mode='markers', 
                    marker=dict(symbol="triangle-up", size=15, color="#212121"), 
                    hoverinfo='text', text=f"Support: {sup_type}"
                ), row=1, col=1)

    # 1.3 Draw Loads
    for load in loads:
        span_idx = load.get('span_idx', 0)
        x_start = cum_len[span_idx]
        x_end = cum_len[span_idx+1]
        
        # --- CASE A: DISTRIBUTED LOAD (UDL) ---
        if 'w' in load and load['w'] != 0:
            w = load['w']
            # Height proportional to max load
            h = (w / max_load_val) * (load_plot_height * 0.7)
            
            # 1. Fill Area (Optional, for clarity)
            fig.add_trace(go.Scatter(
                x=[x_start, x_end, x_end, x_start], 
                y=[0, 0, h, h], 
                fill='toself', fillcolor='rgba(255, 152, 0, 0.2)', 
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ), row=1, col=1)
            
            # 2. Top Line (The "Comb" back)
            fig.add_trace(go.Scatter(
                x=[x_start, x_end], y=[h, h],
                mode='lines', line=dict(color='#EF6C00', width=2),
                showlegend=False, hoverinfo='text', text=f"UDL: {w} {force_unit}/{dist_unit}"
            ), row=1, col=1)
            
            # 3. Multiple Arrows (The "Comb" teeth)
            # Calculate number of arrows based on span length (approx every 0.5-1m visually)
            n_arrows = max(3, int((x_end - x_start) * 2)) 
            arrow_x = np.linspace(x_start, x_end, n_arrows)
            
            for ax in arrow_x:
                fig.add_annotation(
                    x=ax, y=0, ax=ax, ay=h,
                    xref="x1", yref="y1", axref="x1", ayref="y1",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#EF6C00",
                    row=1, col=1
                )
            
            # 4. Label
            fig.add_annotation(
                x=(x_start+x_end)/2, y=h, 
                text=f"<b>w = {w}</b>", 
                showarrow=False, yshift=15, font=dict(color="#EF6C00", size=11),
                row=1, col=1
            )

        # --- CASE B: POINT LOAD ---
        # Assuming input dict has keys like 'P' and 'x' (relative distance)
        # Handle logic compatible with potential input structures
        elif 'P' in load and load['P'] != 0:
            P = load['P']
            # Relative x from start of span usually
            rel_x = load.get('x', 0.5 * (x_end - x_start)) # Default to mid if missing
            load_x = x_start + rel_x
            
            # Height logic
            h = (P / max_load_val) * (load_plot_height * 0.8)
            
            # Draw Single Thick Arrow
            fig.add_annotation(
                x=load_x, y=0, ax=load_x, ay=h,
                xref="x1", yref="y1", axref="x1", ayref="y1",
                showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor="#D32F2F",
                row=1, col=1
            )
            
            # Label
            fig.add_annotation(
                x=load_x, y=h, 
                text=f"<b>P = {P}</b>", 
                showarrow=False, yshift=15, font=dict(color="#D32F2F", size=12, weight="bold"),
                row=1, col=1
            )

    # Force Y-Range for Load Plot to look nice
    fig.update_yaxes(range=[-load_plot_height*0.2, load_plot_height*1.3], visible=False, row=1, col=1)


    # ==================================================
    # ROW 2: SHEAR FORCE DIAGRAM (SFD)
    # ==================================================
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'], 
        mode='lines', line=dict(color='#D32F2F', width=2), 
        fill='tozeroy', fillcolor='rgba(211, 47, 47, 0.1)',
        name="Shear", hovertemplate="%{y:.2f} " + force_unit
    ), row=2, col=1)

    # Annotate Peaks (Only significant ones)
    for i in range(len(vis_spans)):
        span_data = df[df['span_id'] == i]
        if not span_data.empty:
            # Find local min/max
            v_max = span_data['shear'].max()
            v_min = span_data['shear'].min()
            
            # Plot Annotation
            for val, color in [(v_max, 'red'), (v_min, 'red')]:
                if abs(val) > 0.1: # Filter noise
                    # Find X position for this value
                    match_row = span_data.iloc[(span_data['shear']-val).abs().argsort()[:1]]
                    vx = match_row['x'].values[0]
                    
                    fig.add_annotation(
                        x=vx, y=val, text=f"{val:.2f}",
                        showarrow=False, yshift=10 if val>0 else -10,
                        font=dict(color=color, size=10), bgcolor="white",
                        row=2, col=1
                    )

    # ==================================================
    # ROW 3: BENDING MOMENT DIAGRAM (BMD)
    # ==================================================
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['moment'], 
        mode='lines', line=dict(color='#1976D2', width=2), 
        fill='tozeroy', fillcolor='rgba(25, 118, 210, 0.1)',
        name="Moment", hovertemplate="%{y:.2f} " + moment_unit
    ), row=3, col=1)
    
    # Annotate Peaks
    for i in range(len(vis_spans)):
        span_data = df[df['span_id'] == i]
        if not span_data.empty:
            m_max = span_data['moment'].max() # Sagging (+)
            m_min = span_data['moment'].min() # Hogging (-)
            
            for val in [m_max, m_min]:
                if abs(val) > 0.1:
                    match_row = span_data.iloc[(span_data['moment']-val).abs().argsort()[:1]]
                    mx = match_row['x'].values[0]
                    
                    fig.add_annotation(
                        x=mx, y=val, text=f"{val:.2f}",
                        showarrow=False, yshift=10 if val>0 else -10,
                        font=dict(color="#1976D2", size=10), bgcolor="white",
                        row=3, col=1
                    )

    # ==================================================
    # FINAL LAYOUT
    # ==================================================
    # Grid lines & Zero lines
    for r in [2, 3]:
        fig.add_hline(y=0, line_width=1, line_color="black", row=r, col=1)
        for x in cum_len:
            fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="gray", opacity=0.5, row=r, col=1)

    fig.update_layout(
        height=800, 
        showlegend=False, 
        plot_bgcolor="white",
        hovermode="x unified",
        margin=dict(t=30, b=30, l=50, r=20)
    )
    
    # Axis Titles
    fig.update_yaxes(title_text=f"V ({force_unit})", row=2, col=1)
    fig.update_yaxes(title_text=f"M ({moment_unit})", row=3, col=1)
    fig.update_xaxes(title_text=f"Position ({dist_unit})", row=3, col=1)

    return fig
