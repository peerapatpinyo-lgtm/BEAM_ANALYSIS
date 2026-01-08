import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, stir_text, fc, fy, title="Section"):
    """
    Plots the beam cross-section with different Top and Bottom bar sizes.
    """
    b = b_m * 1000
    h = h_m * 1000
    cover = cover_mm
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 1. Concrete Face
    rect = patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='#f0f0f0')
    ax.add_patch(rect)
    
    # 2. Stirrup (Simplified as a box inside cover)
    stir_w = b - 2*cover
    stir_h = h - 2*cover
    stirrup = patches.Rectangle((cover, cover), stir_w, stir_h, linewidth=1.5, edgecolor='blue', facecolor='none', linestyle='--')
    ax.add_patch(stirrup)
    
    # 3. Rebar Plotting Helper
    def plot_bars(n_bars, y_center, db, color='red'):
        if n_bars < 2: n_bars = 2 # Minimum visual
        radius = db / 2
        
        # Spacing logic
        start_x = cover + db # Offset from stirrup roughly
        end_x = b - cover - db
        if n_bars == 1:
            x_positions = [b/2]
        else:
            x_positions = np.linspace(start_x, end_x, int(n_bars))
            
        for x in x_positions:
            circle = patches.Circle((x, y_center), radius, edgecolor='black', facecolor=color, zorder=10)
            ax.add_patch(circle)
            
    # Draw Top Bars
    # y position = height - cover - stirrup_dia (approx 9) - radius
    y_top = h - cover - 9 - (db_top_mm/2)
    plot_bars(n_top, y_top, db_top_mm, color='#d62728') # Red
    
    # Draw Bottom Bars
    y_bot = cover + 9 + (db_bot_mm/2)
    plot_bars(n_bot, y_bot, db_bot_mm, color='#1f77b4') # Blue
    
    # 4. Annotation
    # Dimension lines
    ax.annotate(f"{b:.0f}", xy=(b/2, -40), ha='center', va='top', arrowprops=dict(arrowstyle='|-|'))
    ax.annotate(f"{h:.0f}", xy=(-40, h/2), ha='right', va='center', rotation=90, arrowprops=dict(arrowstyle='|-|'))
    
    # Text Details
    info_text = (
        f"Size: {b:.0f}x{h:.0f} mm\n"
        f"Cover: {cover} mm\n"
        f"fc': {fc} MPa\n"
        f"fy: {fy} MPa"
    )
    ax.text(b*1.1, h*0.9, info_text, fontsize=10, bbox=dict(boxstyle="round", fc="white"))
    
    # Rebar Labels
    ax.text(b/2, h + 30, f"{int(n_top)}-DB{db_top_mm} (Top)", ha='center', color='#d62728', fontweight='bold')
    ax.text(b/2, -90, f"{int(n_bot)}-DB{db_bot_mm} (Bottom)", ha='center', color='#1f77b4', fontweight='bold')
    ax.text(b/2, h/2, f"Stirrup: {stir_text}", ha='center', va='center', color='blue', fontsize=9, backgroundcolor='white')

    ax.set_xlim(-100, b + 150)
    ax.set_ylim(-150, h + 150)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Plots the longitudinal profile of the beam showing spans, supports, and simplified rebar.
    """
    n_spans = len(spans)
    total_length = sum(spans)
    h_mm = h_m * 1000
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    current_x = 0
    
    # 1. Beam Body
    beam_rect = patches.Rectangle((0, 0), total_length, h_mm, linewidth=2, edgecolor='black', facecolor='#f9f9f9')
    ax.add_patch(beam_rect)
    
    # 2. Supports
    for _, row in sup_df.iterrows():
        # --- 🔴 แก้ไขจุดที่ Error ตรงนี้ครับ (เปลี่ยนจาก row['position'] เป็น row['x']) ---
        sx = row['x'] 
        
        # Draw triangle support
        triangle = patches.Polygon([[sx-0.2, -100], [sx+0.2, -100], [sx, 0]], closed=True, edgecolor='black', facecolor='grey')
        ax.add_patch(triangle)
        
        # Check if 'support_id' exists, if not use index or generic name
        s_id = row.get('support_id', 'Sup')
        ax.text(sx, -150, str(s_id), ha='center', fontsize=10, fontweight='bold')

    # 3. Reinforcement Visualization (Simplified)
    offsets = [0] + list(np.cumsum(spans))
    
    for i in range(n_spans):
        start = offsets[i]
        end = offsets[i+1]
        length = end - start
        
        res = design_res[i]
        
        # Bottom Bar (Blue) - Span center
        ax.plot([start + 0.2, end - 0.2], [50, 50], color='#1f77b4', linewidth=3)
        ax.text(start + length/2, 80, f"{res['pos']['n']}-DB{res['bot_db']}", ha='center', color='#1f77b4', fontsize=9)
        
        # Top Bar (Red) - Supports
        ax.plot([start, end], [h_mm-50, h_mm-50], color='#d62728', linewidth=3)
        ax.text(start + length/2, h_mm-90, f"{res['neg']['n']}-DB{res['top_db']}", ha='center', color='#d62728', fontsize=9)
        
        # Stirrups info
        ax.text(start + length/2, h_mm/2, f"Stir: RB{res['stir_db']}@{int(res['shear']['s'])}", ha='center', color='blue', fontsize=8, alpha=0.7)

    # Decoration
    ax.set_xlim(-1, total_length + 1)
    ax.set_ylim(-200, h_mm + 100)
    ax.set_aspect('equal', adjustable='box') 
    ax.axis('off')
    ax.set_title("Longitudinal Section (Reinforcement Layout)", fontsize=12, fontweight='bold')
    
    return fig
