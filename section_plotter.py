import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

def plot_section(b_m, h_m, cover_mm, bar_db, n_top, n_bot, stirrup_txt, fc, fy):
    """
    Plots the cross-section of the RC Beam (Compact Version).
    """
    # [FIX] ลดขนาดรูปลงจาก (4,4) เป็น (3,3)
    fig, ax = plt.subplots(figsize=(3, 3))
    
    b = b_m * 1000
    h = h_m * 1000
    cover = cover_mm
    
    # Draw Concrete
    rect = patches.Rectangle((0, 0), b, h, linewidth=1.5, edgecolor='black', facecolor='#f9f9f9')
    ax.add_patch(rect)
    
    # Draw Stirrup
    inset = cover
    if b > 2*inset and h > 2*inset:
        rect_st = patches.Rectangle((inset, inset), b-2*inset, h-2*inset, 
                                    linewidth=1, edgecolor='darkblue', facecolor='none', linestyle='--')
        ax.add_patch(rect_st)
    
    # Rebars
    # Top
    if n_top > 0:
        sp = (b - 2*inset) / (n_top + 1)
        for i in range(n_top):
            circle = patches.Circle((inset + (i+1)*sp, h - inset - bar_db/2), bar_db/2, color='#d62728', zorder=5)
            ax.add_patch(circle)
            
    # Bot
    if n_bot > 0:
        sp = (b - 2*inset) / (n_bot + 1)
        for i in range(n_bot):
            circle = patches.Circle((inset + (i+1)*sp, inset + bar_db/2), bar_db/2, color='#1f77b4', zorder=5)
            ax.add_patch(circle)
            
    # Labels (Simplified)
    ax.text(b/2, h + 15, f"{n_top}-DB{bar_db}", ha='center', fontsize=9, color='#d62728', fontweight='bold')
    ax.text(b/2, -30, f"{n_bot}-DB{bar_db}", ha='center', fontsize=9, color='#1f77b4', fontweight='bold')
    
    # Dimensions (Side)
    ax.text(-25, h/2, f"{h:.0f}", rotation=90, va='center', fontsize=8)
    ax.text(b/2, -60, f"{b:.0f}", ha='center', fontsize=8)

    ax.set_xlim(-40, b + 40)
    ax.set_ylim(-80, h + 40)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # [FIX] ปรับ Margin ให้ชิดที่สุด
    plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1)
    
    return fig

# ... (ส่วน plot_longitudinal_section_detailed เหมือนเดิม ใช้ตัวเดิมได้เลยครับ) ...
# เพื่อความชัวร์ ให้ copy plot_longitudinal_section_detailed จากคำตอบก่อนหน้ามาแปะต่อท้ายตรงนี้
def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_beam, cover=40):
    fig, ax = plt.subplots(figsize=(12, 3))
    
    cum_dist = [0] + list(pd.Series(spans).cumsum())
    total_length = cum_dist[-1]
    h_m = h_beam
    cover_m = cover / 1000.0
    
    # Beam
    beam_rect = patches.Rectangle((0, 0), total_length, h_m, linewidth=1.5, edgecolor='black', facecolor='#FFFFF0')
    ax.add_patch(beam_rect)
    
    # Supports
    if 'id' in sup_df.columns:
        sup_types = sup_df.set_index('id')['type'].to_dict()
    else:
        sup_types = sup_df.set_index('node_id')['type'].to_dict()

    for i, x in enumerate(cum_dist):
        sType = sup_types.get(i, 'Pin')
        tri_w = max(0.2, total_length * 0.025)
        tri_h = h_m * 0.3
        
        if sType == 'Pin':
            triangle = patches.Polygon([[x, 0], [x - tri_w/2, -tri_h], [x + tri_w/2, -tri_h]], closed=True, color='gray', ec='black')
            ax.add_patch(triangle)
        elif sType == 'Roller':
            triangle = patches.Polygon([[x, 0], [x - tri_w/2, -tri_h], [x + tri_w/2, -tri_h]], closed=True, color='gray', ec='black')
            ax.add_patch(triangle)
            ax.add_patch(patches.Circle((x - tri_w/3, -tri_h - tri_w/4), tri_w/4, color='black'))
            ax.add_patch(patches.Circle((x + tri_w/3, -tri_h - tri_w/4), tri_w/4, color='black'))
            ax.plot([x - tri_w, x + tri_w], [-tri_h - tri_w/2, -tri_h - tri_w/2], color='black', lw=1)
        elif sType == 'Fixed':
            ax.plot([x, x], [-tri_h, h_m + tri_h], color='black', lw=3) 
            hatch_w = tri_w / 2
            for h_y in np.arange(-tri_h, h_m + tri_h, tri_h/2):
                if i == 0: ax.plot([x, x-hatch_w], [h_y, h_y-hatch_w], color='black', lw=1)
                else: ax.plot([x, x+hatch_w], [h_y, h_y-hatch_w], color='black', lw=1)

    # Rebar
    for i, d_data in enumerate(design_res):
        x_start = cum_dist[i]
        x_end = cum_dist[i+1]
        span_len = spans[i]
        
        # Bot
        n_bot = d_data['pos']['n']
        db_bot = d_data.get('db', 16)
        if n_bot > 0:
            y_bot = cover_m
            anchorage = span_len * 0.05 
            ax.plot([x_start + anchorage, x_end - anchorage], [y_bot, y_bot], color='#1f77b4', lw=3, solid_capstyle='round')
            ax.text(x_start + span_len/2, y_bot + 0.15*h_m, f"{n_bot}-DB{db_bot}", ha='center', va='bottom', color='#1f77b4', fontsize=9, fontweight='bold')

    # Top Bars
    for i in range(1, len(cum_dist) - 1):
        x_sup = cum_dist[i]
        if i-1 < len(design_res) and i < len(design_res):
            span_prev = design_res[i-1]
            span_next = design_res[i]
            n_top = max(span_prev['neg']['n'], span_next['neg']['n'])
            if n_top > 0:
                y_top = h_m - cover_m
                len_prev = spans[i-1] / 3.0 
                len_next = spans[i] / 3.0
                ax.plot([x_sup - len_prev, x_sup + len_next], [y_top, y_top], color='#d62728', lw=3, solid_capstyle='round')
                ax.text(x_sup, y_top - 0.2*h_m, f"{n_top}-DB16", ha='center', va='top', color='#d62728', fontsize=9, fontweight='bold')

    ax.set_xlim(-total_length*0.1, total_length*1.1)
    ax.set_ylim(-h_m*0.5, h_m*1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig
