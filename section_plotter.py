import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

def plot_section(b_m, h_m, cover_mm, bar_db, n_top, n_bot, stirrup_txt, fc, fy):
    """
    Plots the cross-section (Compact Fixed Scale).
    """
    # Keep compact size as requested
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    
    b = b_m * 1000
    h = h_m * 1000
    cover = cover_mm
    stirrup_dia = 6 
    
    # Concrete
    rect = patches.Rectangle((0, 0), b, h, linewidth=1.5, edgecolor='black', facecolor='#f9f9f9')
    ax.add_patch(rect)
    
    # Stirrup
    inset = cover
    if b > 2*inset and h > 2*inset:
        rect_st = patches.FancyBboxPatch((inset, inset), b-2*inset, h-2*inset,
                                         boxstyle="round,pad=0,rounding_size=10",
                                         linewidth=1, edgecolor='darkblue', facecolor='none', linestyle='--')
        ax.add_patch(rect_st)
    
    # Rebar Logic
    def draw_bars_row(n, y_center, color):
        if n <= 0: return
        start_x = cover + stirrup_dia + (bar_db/2)
        end_x = b - (cover + stirrup_dia + (bar_db/2))
        width_avail = end_x - start_x
        
        if n == 1: xs = [b/2]
        else:
            gap = width_avail / (n - 1)
            xs = [start_x + i*gap for i in range(n)]
            
        for cx in xs:
            circle = patches.Circle((cx, y_center), bar_db/2, color=color, zorder=5, ec='black', lw=0.5)
            ax.add_patch(circle)

    draw_bars_row(n_top, h - (cover + stirrup_dia + bar_db/2), '#d62728')
    draw_bars_row(n_bot, cover + stirrup_dia + bar_db/2, '#1f77b4')
            
    # Labels
    ax.text(b/2, h + 30, f"{n_top}-DB{bar_db}", ha='center', fontsize=9, color='#d62728', fontweight='bold')
    ax.text(b/2, -50, f"{n_bot}-DB{bar_db}", ha='center', fontsize=9, color='#1f77b4', fontweight='bold')
    
    # Dimensions
    ax.text(-40, h/2, f"{h:.0f}", rotation=90, va='center', fontsize=8)
    ax.text(b/2, -90, f"{b:.0f}", ha='center', fontsize=8)

    ax.set_xlim(-60, b + 60)
    ax.set_ylim(-110, h + 80)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.subplots_adjust(left=0.15, right=0.85, top=0.90, bottom=0.15)
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_beam, cover=40):
    """
    Plots the longitudinal section (Wide & Large for bottom view).
    """
    # [FIX] เพิ่มขนาดให้ใหญ่ ชัดเจน (Wide Format)
    fig, ax = plt.subplots(figsize=(15, 3.5))
    
    cum_dist = [0] + list(pd.Series(spans).cumsum())
    total_length = cum_dist[-1]
    h_m = h_beam
    cover_m = cover / 1000.0
    
    # Beam body
    beam_rect = patches.Rectangle((0, 0), total_length, h_m, linewidth=1.5, edgecolor='black', facecolor='#FFFFF0')
    ax.add_patch(beam_rect)
    
    # Supports
    if 'id' in sup_df.columns: sup_types = sup_df.set_index('id')['type'].to_dict()
    else: sup_types = sup_df.set_index('node_id')['type'].to_dict()

    for i, x in enumerate(cum_dist):
        sType = sup_types.get(i, 'Pin')
        tri_w = max(0.2, total_length * 0.02)
        tri_h = h_m * 0.25
        if sType in ['Pin', 'Roller']:
            ax.add_patch(patches.Polygon([[x, 0], [x - tri_w/2, -tri_h], [x + tri_w/2, -tri_h]], closed=True, color='gray', ec='black'))
            if sType == 'Roller':
                ax.plot([x - tri_w, x + tri_w], [-tri_h - tri_w/4, -tri_h - tri_w/4], color='black', lw=1)
        elif sType == 'Fixed':
            ax.plot([x, x], [-tri_h, h_m + tri_h], color='black', lw=3)

    # Rebars
    for i, d_data in enumerate(design_res):
        x_start, x_end = cum_dist[i], cum_dist[i+1]
        span_len = spans[i]
        
        # Bottom Bar
        n_bot = d_data['pos']['n']
        if n_bot > 0:
            y_bot = cover_m
            anchorage = span_len * 0.05 
            ax.plot([x_start + anchorage, x_end - anchorage], [y_bot, y_bot], color='#1f77b4', lw=4)
            ax.text(x_start + span_len/2, y_bot + 0.15*h_m, f"{n_bot}-DB{d_data.get('db',16)}", ha='center', color='#1f77b4', fontsize=11, fontweight='bold')

    # Top Bars (Continuous Logic Simplified)
    for i in range(1, len(cum_dist) - 1):
        x_sup = cum_dist[i]
        if i-1 < len(design_res) and i < len(design_res):
            n_top = max(design_res[i-1]['neg']['n'], design_res[i]['neg']['n'])
            if n_top > 0:
                y_top = h_m - cover_m
                len_prev, len_next = spans[i-1]/3.0, spans[i]/3.0
                ax.plot([x_sup - len_prev, x_sup + len_next], [y_top, y_top], color='#d62728', lw=4)
                ax.text(x_sup, y_top - 0.2*h_m, f"{n_top}-DB16", ha='center', color='#d62728', fontsize=11, fontweight='bold')

    # Add Dimensions Text
    for i, length in enumerate(spans):
        mid_x = cum_dist[i] + length/2
        ax.text(mid_x, -h_m*0.6, f"L = {length:.2f} m", ha='center', va='top', fontsize=10)

    ax.set_xlim(-total_length*0.05, total_length*1.05)
    ax.set_ylim(-h_m*0.8, h_m*1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig
