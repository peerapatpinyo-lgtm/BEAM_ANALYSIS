import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b, h, cover, db, n_top, n_bot, stir_info, fc, fy):
    """
    Plots the cross-section of the beam using Matplotlib.
    All units in meters except db (mm).
    """
    fig, ax = plt.subplots(figsize=(4, 5))
    
    # Concrete Outline
    rect = patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='gray', facecolor='#f0f0f0')
    ax.add_patch(rect)
    
    # Stirrup
    cover_m = cover / 1000
    stir_w = b - 2*cover_m
    stir_h = h - 2*cover_m
    rect_stir = patches.Rectangle((cover_m, cover_m), stir_w, stir_h, linewidth=1.5, edgecolor='blue', facecolor='none', linestyle='--')
    ax.add_patch(rect_stir)
    
    # Rebars
    db_m = db / 1000
    
    # Bottom Bars
    spacing_bot = stir_w / (n_bot + 1) if n_bot > 1 else stir_w/2
    for i in range(n_bot):
        x = cover_m + (i if n_bot > 1 else 0.5) * (stir_w / (n_bot - 1) if n_bot > 1 else 1)
        if n_bot == 1: x = b/2
        
        circ = patches.Circle((x, cover_m + db_m/2), db_m/2, color='red', zorder=10)
        ax.add_patch(circ)
        
    # Top Bars
    for i in range(n_top):
        x = cover_m + (i if n_top > 1 else 0.5) * (stir_w / (n_top - 1) if n_top > 1 else 1)
        if n_top == 1: x = b/2
        
        circ = patches.Circle((x, h - cover_m - db_m/2), db_m/2, color='red', zorder=10)
        ax.add_patch(circ)

    ax.set_xlim(-0.05, b+0.05)
    ax.set_ylim(-0.05, h+0.05)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"{int(b*100)}x{int(h*100)} cm\n{n_bot}-DB{db} (Bot)", fontsize=10)
    
    return fig

def plot_longitudinal_section_detailed(spans, supports, design_res, h, cover):
    """
    Plots side view of beam with reinforcement.
    """
    total_L = sum(spans)
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Beam Body
    rect = patches.Rectangle((0, 0), total_L, h, linewidth=1, edgecolor='black', facecolor='#f9f9f9')
    ax.add_patch(rect)
    
    # Supports
    cum_x = [0] + list(np.cumsum(spans))
    for x in cum_x:
        poly = patches.Polygon([[x, 0], [x-0.1, -0.2], [x+0.1, -0.2]], closed=True, color='black')
        ax.add_patch(poly)
        
    # Rebar Lines (Simplified)
    cover_m = cover / 1000
    
    # Bottom Rebar (Continuous line for viz)
    ax.plot([0, total_L], [cover_m, cover_m], color='red', linewidth=2, linestyle='-')
    
    # Top Rebar
    ax.plot([0, total_L], [h-cover_m, h-cover_m], color='red', linewidth=2, linestyle='--')
    
    # Annotate Steel
    start_x = 0
    for i, span_len in enumerate(spans):
        mid_x = start_x + span_len/2
        
        # Get design info for this span
        if i < len(design_res):
            d = design_res[i]
            txt_bot = f"{d['pos']['n']}-DB{d['db']}"
            txt_top = f"{d['neg']['n']}-DB{d['db']}"
            
            ax.text(mid_x, cover_m + 0.05, txt_bot, color='red', ha='center', fontsize=8)
            ax.text(mid_x, h - cover_m - 0.1, txt_top, color='red', ha='center', fontsize=8)
            
        start_x += span_len

    ax.set_xlim(-0.5, total_L + 0.5)
    ax.set_ylim(-0.5, h + 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    return fig
