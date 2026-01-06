import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_section(b, h, cover_mm, db_mm, n_top, n_bot, stirrup_info, fc=None, fy=None):
    """
    วาด Cross Section คานคอนกรีตเสริมเหล็ก พร้อม Material Specs
    """
    # Create Figure
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # 1. Concrete Section
    rect = patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='#2C3E50', facecolor='#F4F6F7')
    ax.add_patch(rect)
    
    cover = cover_mm / 1000
    db = db_mm / 1000
    
    # 2. Stirrup
    stirrup_w = b - 2*cover
    stirrup_h = h - 2*cover
    if stirrup_w > 0 and stirrup_h > 0:
        rect_stir = patches.Rectangle((cover, cover), stirrup_w, stirrup_h, 
                                      linewidth=1.5, edgecolor='#C0392B', facecolor='none', linestyle='--')
        ax.add_patch(rect_stir)

    # 3. Main Bars
    def draw_bars(n, y_pos):
        if n <= 0: return
        start_x = cover + db/2
        end_x = b - cover - db/2
        
        if n == 1:
            x_positions = [b/2]
        else:
            gap = (end_x - start_x) / (n - 1)
            x_positions = [start_x + i*gap for i in range(n)]
            
        for x in x_positions:
            circle = patches.Circle((x, y_pos), radius=db/2, edgecolor='#2980B9', facecolor='#2980B9')
            ax.add_patch(circle)

    y_top = h - cover - db/2
    draw_bars(int(n_top), y_top)
    
    y_bot = cover + db/2
    draw_bars(int(n_bot), y_bot)

    # 4. Dimensions & Labels
    # Dimension Lines H
    ax.plot([-0.05, -0.05], [0, h], color='black', linewidth=1)
    ax.text(-0.08, h/2, f"H={h:.2f}m", rotation=90, va='center', ha='right', fontsize=11)
    
    # Dimension Lines B
    ax.plot([0, b], [-0.05, -0.05], color='black', linewidth=1)
    ax.text(b/2, -0.08, f"B={b:.2f}m", va='top', ha='center', fontsize=11)
    
    # Rebar Text
    if n_top > 0:
        ax.text(b/2, h + 0.03, f"{n_top}-DB{db_mm} (Top)", ha='center', color='#2980B9', fontsize=10, fontweight='bold')
    if n_bot > 0:
        ax.text(b/2, -0.15, f"{n_bot}-DB{db_mm} (Bot)", ha='center', color='#2980B9', fontsize=10, fontweight='bold')
    
    # Stirrup Text
    try:
        s_text = stirrup_info.split('@')[1].strip()
        stir_lbl = f"Stirrup @ {s_text}"
    except:
        stir_lbl = stirrup_info
    ax.text(b+0.04, h/2, stir_lbl, rotation=270, va='center', color='#C0392B', fontsize=10)

    # 5. Material Specs Box (Updated)
    if fc and fy:
        spec_text = f"SPEC:\n$f'_c$={fc:.0f} MPa\n$f_y$={fy:.0f} MPa"
        ax.text(b*1.3, h*0.9, spec_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='#BDC3C7', alpha=0.9))

    # Settings
    ax.set_xlim(-0.15, b + 0.3)
    ax.set_ylim(-0.25, h + 0.15)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig
