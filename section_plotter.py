import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

def plot_section(b_m, h_m, cover_mm, bar_db, n_top, n_bot, stirrup_txt, fc, fy):
    """
    Plots the cross-section of the RC Beam.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Convert dimensions to mm for plotting logic, but keep labels correct
    b = b_m * 1000
    h = h_m * 1000
    cover = cover_mm
    
    # Draw Concrete Section
    rect = patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='#f0f0f0')
    ax.add_patch(rect)
    
    # Draw Stirrup (simplified box)
    stirrup_inset = cover
    st_w = b - 2*stirrup_inset
    st_h = h - 2*stirrup_inset
    if st_w > 0 and st_h > 0:
        rect_st = patches.Rectangle((stirrup_inset, stirrup_inset), st_w, st_h, 
                                    linewidth=1.5, edgecolor='darkblue', facecolor='none', linestyle='--')
        ax.add_patch(rect_st)
    
    # Draw Rebars
    # Top Bars
    if n_top > 0:
        spacing = st_w / (n_top + 1) if n_top > 0 else 0
        for i in range(n_top):
            cx = stirrup_inset + (i+1)*spacing
            cy = h - stirrup_inset - (bar_db/2) # Simple positioning
            circle = patches.Circle((cx, cy), bar_db/2, color='red', zorder=10)
            ax.add_patch(circle)
            
    # Bottom Bars
    if n_bot > 0:
        spacing = st_w / (n_bot + 1) if n_bot > 0 else 0
        for i in range(n_bot):
            cx = stirrup_inset + (i+1)*spacing
            cy = stirrup_inset + (bar_db/2)
            circle = patches.Circle((cx, cy), bar_db/2, color='blue', zorder=10)
            ax.add_patch(circle)
            
    # Annotations
    ax.text(b/2, h + 20, f"{n_top}-DB{bar_db}", ha='center', color='red', fontsize=12, fontweight='bold')
    ax.text(b/2, -40, f"{n_bot}-DB{bar_db}", ha='center', color='blue', fontsize=12, fontweight='bold')
    ax.text(b + 20, h/2, stirrup_txt, rotation=270, va='center', fontsize=10)
    
    # Dimensions
    ax.text(-30, h/2, f"H={h:.0f}", rotation=90, va='center')
    ax.text(b/2, -80, f"B={b:.0f}", ha='center')

    ax.set_xlim(-50, b + 50)
    ax.set_ylim(-100, h + 50)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def plot_longitudinal_section(spans, sup_df, design_res, h_beam, cover=40):
    """
    (Legacy) Simple longitudinal plot.
    Keeping it just in case, but app.py uses the detailed one below.
    """
    return plot_longitudinal_section_detailed(spans, sup_df, design_res, h_beam, cover)

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_beam, cover=40):
    """
    Plots a detailed longitudinal section of the beam with accurate supports and rebar.
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # [FIX] Ensure spans is treated as a list/array for pandas
    cum_dist = [0] + list(pd.Series(spans).cumsum())
    total_length = cum_dist[-1]
    h_m = h_beam
    cover_m = cover / 1000.0
    
    # 1. Draw Beam Body
    beam_rect = patches.Rectangle((0, 0), total_length, h_m, linewidth=1.5, edgecolor='black', facecolor='#FFFFF0')
    ax.add_patch(beam_rect)
    
    # 2. Draw Supports (Realistic)
    # Map node_id or id to type
    if 'id' in sup_df.columns:
        sup_types = sup_df.set_index('id')['type'].to_dict()
    else:
        sup_types = sup_df.set_index('node_id')['type'].to_dict()

    for i, x in enumerate(cum_dist):
        sType = sup_types.get(i, 'Pin')
        
        # Support triangle geometry
        tri_w = max(0.2, total_length * 0.025)
        tri_h = h_m * 0.3
        
        if sType == 'Pin':
            # Triangle
            triangle = patches.Polygon([[x, 0], [x - tri_w/2, -tri_h], [x + tri_w/2, -tri_h]], closed=True, color='gray', ec='black')
            ax.add_patch(triangle)
        elif sType == 'Roller':
            # Triangle
            triangle = patches.Polygon([[x, 0], [x - tri_w/2, -tri_h], [x + tri_w/2, -tri_h]], closed=True, color='gray', ec='black')
            ax.add_patch(triangle)
            # Circles underneath
            circle1 = patches.Circle((x - tri_w/3, -tri_h - tri_w/4), tri_w/4, color='black')
            circle2 = patches.Circle((x + tri_w/3, -tri_h - tri_w/4), tri_w/4, color='black')
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            # Ground line
            ax.plot([x - tri_w, x + tri_w], [-tri_h - tri_w/2, -tri_h - tri_w/2], color='black', lw=1)
        elif sType == 'Fixed':
            # Vertical Wall Line
            ax.plot([x, x], [-tri_h, h_m + tri_h], color='black', lw=3) 
            # Hatches
            hatch_w = tri_w / 2
            for h_y in np.arange(-tri_h, h_m + tri_h, tri_h/2):
                if i == 0: # Left side fixed
                    ax.plot([x, x-hatch_w], [h_y, h_y-hatch_w], color='black', lw=1)
                else: # Right side fixed
                    ax.plot([x, x+hatch_w], [h_y, h_y-hatch_w], color='black', lw=1)


    # 3. Draw Rebar (Realistic based on design_res)
    for i, d_data in enumerate(design_res):
        x_start = cum_dist[i]
        x_end = cum_dist[i+1]
        span_len = spans[i]
        
        # --- Bottom Bars (Positive Moment) ---
        n_bot = d_data['pos']['n']
        db_bot = d_data.get('db', 16) # Default if missing
        
        if n_bot > 0:
            y_bot = cover_m
            # Anchorage offset
            anchorage = span_len * 0.05 
            ax.plot([x_start + anchorage, x_end - anchorage], [y_bot, y_bot], 
                    color='#1f77b4', lw=3, solid_capstyle='round') # Blue
            # Label
            ax.text(x_start + span_len/2, y_bot + 0.15*h_m, f"{n_bot}-DB{db_bot}\n(Bot)", 
                    ha='center', va='bottom', color='#1f77b4', fontsize=9, fontweight='bold')

    # --- Top Bars (Negative Moment over interior supports) ---
    for i in range(1, len(cum_dist) - 1):
        x_sup = cum_dist[i]
        
        # Check adjacent spans
        if i-1 < len(design_res) and i < len(design_res):
            span_prev = design_res[i-1]
            span_next = design_res[i]
            
            # Use max bars from adjacent
            n_top = max(span_prev['neg']['n'], span_next['neg']['n'])
            
            if n_top > 0:
                y_top = h_m - cover_m
                # Extend L/3
                len_prev = spans[i-1] / 3.0 
                len_next = spans[i] / 3.0
                
                ax.plot([x_sup - len_prev, x_sup + len_next], [y_top, y_top], 
                        color='#d62728', lw=3, solid_capstyle='round') # Red
                # Label
                ax.text(x_sup, y_top - 0.2*h_m, f"{n_top}-DB16\n(Top)", 
                        ha='center', va='top', color='#d62728', fontsize=9, fontweight='bold')

    # Setting Axes
    ax.set_xlim(-total_length*0.1, total_length*1.1)
    ax.set_ylim(-h_m*0.5, h_m*1.5)
    ax.set_aspect('equal')
    ax.axis('off') # Hide axis for cleaner drawing
    
    plt.tight_layout()
    return fig
