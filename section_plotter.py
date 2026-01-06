import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_section(b, h, cover_mm, db_mm, n_top, n_bot, stirrup_info):
    """
    วาด Cross Section คานคอนกรีตเสริมเหล็ก (Professional Style)
    """
    # Create Figure
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # 1. Draw Concrete Section (Gray Border, White Fill)
    rect = patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='#333333', facecolor='#F0F2F6')
    ax.add_patch(rect)
    
    # Parameters needed
    cover = cover_mm / 1000
    db = db_mm / 1000
    
    # 2. Draw Stirrup (Red Line)
    # สมมติเหล็กปลอกห่างจากผิวคอนกรีต = cover
    stirrup_w = b - 2*cover
    stirrup_h = h - 2*cover
    if stirrup_w > 0 and stirrup_h > 0:
        rect_stir = patches.Rectangle((cover, cover), stirrup_w, stirrup_h, 
                                      linewidth=1.5, edgecolor='#E74C3C', facecolor='none', linestyle='--')
        ax.add_patch(rect_stir)

    # 3. Draw Main Bars (Blue Dots)
    # Function to distribute bars evenly
    def draw_bars(n, y_pos):
        if n <= 0: return
        
        # Calculate X positions
        # Space available inside stirrups (approx)
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

    # Draw Top Bars
    y_top = h - cover - db/2
    draw_bars(int(n_top), y_top)
    
    # Draw Bottom Bars
    y_bot = cover + db/2
    draw_bars(int(n_bot), y_bot)

    # 4. Annotations & Dimensions
    # Dimensions text
    ax.text(-0.05, h/2, f"H={h:.2f}m", rotation=90, verticalalignment='center', horizontalalignment='right', fontsize=10)
    ax.text(b/2, -0.05, f"B={b:.2f}m", verticalalignment='top', horizontalalignment='center', fontsize=10)
    
    # Rebar Labels
    ax.text(b/2, h + 0.02, f"{n_top}-DB{db_mm} (Top)", ha='center', color='#2980B9', fontsize=9, fontweight='bold')
    ax.text(b/2, -0.12, f"{n_bot}-DB{db_mm} (Bot)", ha='center', color='#2980B9', fontsize=9, fontweight='bold')
    
    # Stirrup Label (Parse string "RB9 @ 0.15 m")
    try:
        s_text = stirrup_info.split('@')[1].strip()
        stir_lbl = f"Stirrup @ {s_text}"
    except:
        stir_lbl = stirrup_info

    ax.text(b+0.02, h/2, stir_lbl, rotation=270, verticalalignment='center', color='#C0392B', fontsize=9)

    # Settings
    ax.set_xlim(-0.1, b + 0.1)
    ax.set_ylim(-0.2, h + 0.2)
    ax.set_aspect('equal')
    ax.axis('off') # Hide axes
    
    plt.tight_layout()
    return fig
