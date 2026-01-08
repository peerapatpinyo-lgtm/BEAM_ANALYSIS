import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 📏 Config ---
DPI_VALUE = 100     
GLOBAL_FONT = 9

def _setup_white_canvas(figsize):
    """ Helper to force pure white background """
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI_VALUE)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    return fig, ax

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, stir_text, fc, fy, title="SECTION A-A"):
    """ 
    Plots Cross Section with Dynamic Scaling & Top/Bottom Bar distinction.
    """
    SECTION_SCALE = 60 # Scale factor (Higher = Smaller image on screen)
    
    b = b_m * 1000
    h = h_m * 1000
    cover = cover_mm
    
    # Viewport calculation (Auto-size)
    view_left, view_right = -300, b + 500
    view_bottom, view_top = -400, h + 400
    
    width_inches = (view_right - view_left) / SECTION_SCALE 
    height_inches = (view_top - view_bottom) / SECTION_SCALE
    
    # Clamp min size to avoid errors
    width_inches = max(4, width_inches)
    height_inches = max(4, height_inches)
    
    fig, ax = _setup_white_canvas((width_inches, height_inches))
    
    # 1. Concrete Outline
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='#2c3e50', facecolor='#FFFFFF', zorder=0))
    
    # 2. Stirrup Line
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=1, edgecolor='#7f8c8d', ls='--', zorder=1))
    
    # 3. Main Bars Helper
    def draw_bars(n_bars, y_center, db, color):
        if n_bars < 2: n_bars = 2 # Minimum visual
        radius = db / 2
        
        # Calculate X positions
        # Space out between stirrup corners (approx)
        start_x = cover + db
        end_x = b - cover - db
        
        if n_bars == 1:
            x_positions = [b/2]
        else:
            if end_x < start_x: # Case where beam is very narrow
                x_positions = [b/2] * int(n_bars)
            else:
                x_positions = np.linspace(start_x, end_x, int(n_bars))
            
        for x in x_positions:
            circle = plt.Circle((x, y_center), radius, facecolor=color, edgecolor='black', linewidth=0.5, zorder=10)
            ax.add_patch(circle)

    # Draw Top Bars (Red)
    # Position: Top - Cover - Stirrup(approx 9) - radius
    y_top = h - cover - 9 - (db_top_mm/2)
    draw_bars(n_top, y_top, db_top_mm, '#d62728') 
    
    # Draw Bottom Bars (Blue)
    y_bot = cover + 9 + (db_bot_mm/2)
    draw_bars(n_bot, y_bot, db_bot_mm, '#1f77b4') 

    # 4. Annotations & Dimensions
    # Top Leader
    ax.plot([b/2, b+150], [y_top, y_top+100], color='#d62728', lw=1)
    ax.text(b+160, y_top+100, f"{int(n_top)}-DB{db_top_mm} (Top)", va='center', color='#d62728', fontweight='bold', fontsize=GLOBAL_FONT)
    
    # Bottom Leader
    ax.plot([b/2, b+150], [y_bot, y_bot-100], color='#1f77b4', lw=1)
    ax.text(b+160, y_bot-100, f"{int(n_bot)}-DB{db_bot_mm} (Bot)", va='center', color='#1f77b4', fontweight='bold', fontsize=GLOBAL_FONT)
    
    # Stirrup Label
    ax.text(b+50, h/2, f"Stirrup: {stir_text}", va='center', color='#2c3e50', fontsize=GLOBAL_FONT-1)

    # Dimensions (Width)
    ax.plot([0, b], [-100, -100], color='black', lw=0.8)
    ax.plot([0, 0], [-80, -120], color='black', lw=0.8)
    ax.plot([b, b], [-80, -120], color='black', lw=0.8)
    ax.text(b/2, -150, f"{int(b)} mm", ha='center', va='top', fontsize=GLOBAL_FONT)
    
    # Dimensions (Height)
    ax.plot([-100, -100], [0, h], color='black', lw=0.8)
    ax.plot([-80, -120], [0, 0], color='black', lw=0.8)
    ax.plot([-80, -120], [h, h], color='black', lw=0.8)
    ax.text([-150], h/2, f"{int(h)} mm", ha='right', va='center', rotation=90, fontsize=GLOBAL_FONT)
    
    # Material Box
    mat_text = f"fc' = {fc} MPa\nfy = {fy} MPa"
    ax.text(view_right-50, view_bottom+50, mat_text, ha='right', va='bottom', fontsize=GLOBAL_FONT-2, color='gray', 
            bbox=dict(facecolor='#FFFFFF', alpha=1.0, edgecolor='#cccccc'))

    # Title
    ax.text(b/2, h + 150, title, ha='center', fontweight='bold', fontsize=GLOBAL_FONT+2)

    ax.set_xlim(view_left, view_right)
    ax.set_ylim(view_bottom, view_top)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Plots the longitudinal profile showing spans, supports, and simplified rebar.
    Using 'mm' units for correct aspect ratio.
    """
    spans_mm = [s * 1000 for s in spans]
    total_length_mm = sum(spans_mm)
    h_mm = h_m * 1000
    
    # Dynamic width based on beam length
    # Approx 1000mm per inch width roughly
    width_inches = max(8, total_length_mm / 1000)
    height_inches = 4
    
    fig, ax = _setup_white_canvas((width_inches, height_inches))
    
    # 1. Beam Body
    beam_rect = patches.Rectangle((0, 0), total_length_mm, h_mm, linewidth=2, edgecolor='black', facecolor='#fcfcfc')
    ax.add_patch(beam_rect)
    
    # 2. Supports
    sup_w = 300 # mm
    sup_h = 300 # mm
    
    for _, row in sup_df.iterrows():
        sx = row['x'] * 1000 # Convert m to mm
        
        # Triangle Support
        triangle = patches.Polygon(
            [[sx - sup_w/2, -sup_h], [sx + sup_w/2, -sup_h], [sx, 0]], 
            closed=True, edgecolor='black', facecolor='#bdc3c7'
        )
        ax.add_patch(triangle)
        
        # Base Line
        ax.plot([sx - sup_w, sx + sup_w], [-sup_h, -sup_h], color='black', lw=2)
        
        # Label
        s_id = row.get('id', '')
        ax.text(sx, -sup_h - 100, f"Supp {s_id}", ha='center', fontsize=GLOBAL_FONT)

    # 3. Reinforcement
    offsets_mm = [0] + list(np.cumsum(spans_mm))
    
    for i in range(len(spans)):
        start = offsets_mm[i]
        end = offsets_mm[i+1]
        length = end - start
        res = design_res[i]
        
        # Bottom Bar (Blue) - with hooks concept (shortened slightly from ends)
        bot_y = cover_mm + 20
        ax.plot([start + cover_mm, end - cover_mm], [bot_y, bot_y], color='#1f77b4', linewidth=3, zorder=5)
        ax.text(start + length/2, bot_y + 40, f"{res['pos']['n']}-DB{res['bot_db']}", 
                ha='center', color='#1f77b4', fontsize=GLOBAL_FONT-1, fontweight='bold')
        
        # Top Bar (Red)
        top_y = h_mm - cover_mm - 20
        ax.plot([start, end], [top_y, top_y], color='#d62728', linewidth=3, zorder=5)
        ax.text(start + length/2, top_y - 60, f"{res['neg']['n']}-DB{res['top_db']}", 
                ha='center', color='#d62728', fontsize=GLOBAL_FONT-1, fontweight='bold')
        
        # Stirrup Info Box
        ax.text(start + length/2, h_mm/2, f"Stir: RB{res['stir_db']}@{int(res['shear']['s'])}", 
                ha='center', va='center', color='#2c3e50', fontsize=GLOBAL_FONT-2, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 4. Dimensions
    # Total Length
    ax.plot([0, total_length_mm], [h_mm + 300, h_mm + 300], color='black', lw=1)
    ax.text(total_length_mm/2, h_mm + 350, f"Total Length = {total_length_mm/1000:.2f} m", ha='center', fontsize=GLOBAL_FONT)
    
    # Scale adjustment
    ax.set_xlim(-500, total_length_mm + 500)
    ax.set_ylim(-600, h_mm + 600)
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')
    
    return fig
