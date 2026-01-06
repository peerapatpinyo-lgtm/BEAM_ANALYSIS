import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b, h, cover_mm, db_mm, n_top, n_bot, stirrup_info, fc=None, fy=None):
    """
    วาด Cross Section คานคอนกรีตเสริมเหล็ก (เหมือนเดิม)
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Concrete Section
    rect = patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='#2C3E50', facecolor='#F4F6F7')
    ax.add_patch(rect)
    
    cover = cover_mm / 1000
    db = db_mm / 1000
    
    # Stirrup
    stirrup_w = b - 2*cover
    stirrup_h = h - 2*cover
    if stirrup_w > 0 and stirrup_h > 0:
        rect_stir = patches.Rectangle((cover, cover), stirrup_w, stirrup_h, 
                                      linewidth=1.5, edgecolor='#C0392B', facecolor='none', linestyle='--')
        ax.add_patch(rect_stir)

    # Main Bars
    def draw_bars(n, y_pos, color):
        if n <= 0: return
        start_x = cover + db/2
        end_x = b - cover - db/2
        
        if n == 1:
            x_positions = [b/2]
        else:
            gap = (end_x - start_x) / (n - 1)
            x_positions = [start_x + i*gap for i in range(n)]
            
        for x in x_positions:
            circle = patches.Circle((x, y_pos), radius=db/2, edgecolor=color, facecolor=color)
            ax.add_patch(circle)

    y_top = h - cover - db/2
    draw_bars(int(n_top), y_top, '#E74C3C') # Red for Top
    
    y_bot = cover + db/2
    draw_bars(int(n_bot), y_bot, '#2980B9') # Blue for Bot

    # Labels
    ax.text(b/2, h + 0.05, f"{n_top}-DB{db_mm}", ha='center', color='#E74C3C', fontweight='bold')
    ax.text(b/2, -0.05, f"{n_bot}-DB{db_mm}", ha='center', color='#2980B9', fontweight='bold')
    
    # Stirrup Text
    try:
        s_text = stirrup_info.split('@')[1].strip()
        stir_lbl = f"Stirrup RB9 @ {s_text}"
    except:
        stir_lbl = stirrup_info
    ax.text(b+0.02, h/2, stir_lbl, rotation=270, va='center', color='#C0392B', fontsize=10)

    # Specs
    if fc and fy:
        spec_text = f"$f'_c$={fc:.0f}\n$f_y$={fy:.0f}"
        ax.text(b*0.1, h*0.5, spec_text, fontsize=8, alpha=0.5)

    ax.set_xlim(-0.1, b + 0.2)
    ax.set_ylim(-0.1, h + 0.15)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig

def plot_longitudinal_section(spans, supports, design_data, h, cover_mm):
    """
    วาดรูปตัดตามยาว (Longitudinal Elevation)
    """
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    cover = cover_mm / 1000.0
    
    # Adjust Figure size based on length
    fig_w = min(12, max(8, total_len * 0.8))
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    
    # 1. Draw Concrete Beam
    beam_rect = patches.Rectangle((0, 0), total_len, h, linewidth=2, edgecolor='black', facecolor='#ECF0F1')
    ax.add_patch(beam_rect)
    
    # 2. Draw Supports
    for i, x in enumerate(cum_spans):
        # Determine support type from data if available, else generic triangle
        # Draw Triangle Support
        tri = patches.Polygon([[x, -0.1], [x-0.1, -0.25], [x+0.1, -0.25]], closed=True, color='#2C3E50')
        ax.add_patch(tri)
        ax.text(x, -0.35, f"Sup {i+1}", ha='center', fontsize=9)

    # 3. Draw Reinforcement (Simplified for Visualization)
    # Logic: 
    # - Bottom Bars (Positive Moment) -> Run full span (Visual simplification)
    # - Top Bars (Negative Moment) -> Run over supports
    
    y_top = h - cover
    y_bot = cover
    
    # Draw per span
    for i, span_len in enumerate(spans):
        start_x = cum_spans[i]
        end_x = cum_spans[i+1]
        data = design_data[i]
        
        # --- A. Bottom Bars (Midspan Design) ---
        n_bot = data['pos']['n']
        txt_bot = f"{n_bot}-DB{data['db']}"
        
        # Draw Blue Line (Bottom)
        ax.plot([start_x + 0.05, end_x - 0.05], [y_bot, y_bot], color='#2980B9', linewidth=3)
        # Label Bottom
        ax.text((start_x + end_x)/2, y_bot + 0.05, txt_bot, color='#2980B9', ha='center', fontsize=9, fontweight='bold')
        
        # --- B. Top Bars (Left Support) ---
        # Note: We need to handle continuity. For now, draw based on 'neg' design of this span
        # Usually 'neg' is taken as max of left/right moment.
        # Let's visualize the reinforcement required at the supports.
        
        n_top = data['neg']['n']
        txt_top = f"{n_top}-DB{data['db']}"
        
        # Draw Red Line (Top) - Visualizing continuity over support
        # We draw a segment centered at the support (or left side of span)
        # This is schematic.
        
        # Left Support Region of this span
        if n_top > 0:
            # Draw segment covering 1/4 of span
            seg_len = span_len / 3.5
            ax.plot([start_x, start_x + seg_len], [y_top, y_top], color='#E74C3C', linewidth=3)
            ax.text(start_x + seg_len/2, y_top - 0.08, txt_top, color='#E74C3C', ha='center', fontsize=9, fontweight='bold')
            
        # Also need to check Right Support for the last span or continuity
        # Ideally, we should pass Support Moments design. 
        # For this simplified view, we mirror the pattern at the end of span if it's the last one
        if i == len(spans) - 1:
             ax.plot([end_x - seg_len, end_x], [y_top, y_top], color='#E74C3C', linewidth=3)
    
    # 4. Stirrups Indication (Vertical lines)
    # Draw a few representative stirrups
    for i, span_len in enumerate(spans):
        start_x = cum_spans[i]
        # Draw 5 stirrups at each end
        s_spacing = 0.15 # Schematic
        for k in range(5):
            sx = start_x + 0.1 + k*s_spacing
            ax.plot([sx, sx], [cover, h-cover], color='#C0392B', linestyle='--', linewidth=0.8)
            
            ex = cum_spans[i+1] - 0.1 - k*s_spacing
            ax.plot([ex, ex], [cover, h-cover], color='#C0392B', linestyle='--', linewidth=0.8)
            
    # Settings
    ax.set_ylim(-0.4, h + 0.2)
    ax.set_xlim(-0.5, total_len + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Longitudinal Section (Reinforcement Profile)", pad=10, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig
