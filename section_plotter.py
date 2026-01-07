import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b, h, cover, db, n_top, n_bot, stir_label, fc, fy):
    """ 
    Engineering Cross Section
    - Corner bars are MANDATORY
    - Clean look
    """
    fig, ax = plt.subplots(figsize=(5, 6))
    
    # Units: mm
    B, H = b*1000, h*1000
    c = cover
    
    # 1. Concrete Face
    rect = patches.Rectangle((0, 0), B, H, linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(rect)
    
    # 2. Stirrup Line (Assume RB6 or RB9)
    db_stir = 6 # mm estimation
    w_s = B - 2*c
    h_s = H - 2*c
    stir = patches.Rectangle((c, c), w_s, h_s, linewidth=2, edgecolor='#0000FF', facecolor='none', linestyle='-')
    ax.add_patch(stir)
    
    # Hook detail (Schematic)
    ax.plot([c+10, c-5], [H-c-10, H-c+5], color='blue', linewidth=2)
    
    # 3. Rebars Logic (MUST BE AT CORNERS)
    # Top Bars
    y_top = H - c - db/2 - db_stir
    if n_top >= 2:
        # Linspace ensures bars are at start and end (corners) of the effective width
        x_tops = np.linspace(c + db_stir + db/2, B - c - db_stir - db/2, n_top)
    else:
        x_tops = [B/2] # Single bar (rare)

    for x in x_tops:
        circle = patches.Circle((x, y_top), db/2, edgecolor='black', facecolor='#FF0000', zorder=10)
        ax.add_patch(circle)
        
    # Bot Bars
    y_bot = c + db_stir + db/2
    if n_bot >= 2:
        x_bots = np.linspace(c + db_stir + db/2, B - c - db_stir - db/2, n_bot)
    else:
        x_bots = [B/2]

    for x in x_bots:
        circle = patches.Circle((x, y_bot), db/2, edgecolor='black', facecolor='#FF0000', zorder=10)
        ax.add_patch(circle)

    # 4. Dimensions & Labels
    ax.text(B/2, H + 20, f"{int(B)}", ha='center', va='bottom', fontsize=12)
    ax.text(-20, H/2, f"{int(H)}", ha='right', va='center', rotation=90, fontsize=12)
    
    # Leader lines
    ax.annotate(f"{n_top}-DB{db}", xy=(x_tops[0], y_top), xytext=(-50, H),
                arrowprops=dict(arrowstyle='->'), fontsize=10)
    ax.annotate(f"{n_bot}-DB{db}", xy=(x_bots[0], y_bot), xytext=(-50, 0),
                arrowprops=dict(arrowstyle='->'), fontsize=10)
    
    ax.text(B/2, -40, f"Stirrup: {stir_label}", ha='center', color='blue', fontsize=10)

    ax.set_xlim(-80, B + 80)
    ax.set_ylim(-80, H + 80)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_detailed(span_len, h, cover, n_top, n_bot, db, s_stir, span_id):
    """
    CAD-Style Longitudinal Profile
    - Aspect Ratio fixed
    - Dimension lines
    """
    L_mm = span_len * 1000
    h_mm = h * 1000
    
    # Aspect ratio correction: If beam is too long, we scale X to fit
    # But user wants "real look". Let's keep aspect but wide figure.
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # 1. Beam Outline
    ax.plot([0, L_mm], [0, 0], 'k-', linewidth=2) # Bot
    ax.plot([0, L_mm], [h_mm, h_mm], 'k-', linewidth=2) # Top
    ax.plot([0, 0], [0, h_mm], 'k--', linewidth=1) # Left Sup line
    ax.plot([L_mm, L_mm], [0, h_mm], 'k--', linewidth=1) # Right Sup line
    
    # 2. Main Steel (Offset by cover)
    # Top
    ax.plot([cover, L_mm-cover], [h_mm-cover-10, h_mm-cover-10], 'r-', linewidth=3, label='Top')
    # Bot
    ax.plot([cover, L_mm-cover], [cover+10, cover+10], 'r-', linewidth=3, label='Bot')
    
    # 3. Stirrups (Draw actual lines based on spacing)
    s_mm = s_stir * 10
    # Start offset
    curr_x = cover + 50
    while curr_x < (L_mm - cover):
        ax.plot([curr_x, curr_x], [cover, h_mm-cover], 'b-', linewidth=1)
        curr_x += s_mm
        
    # 4. Dimensions
    # Span Text
    ax.text(L_mm/2, h_mm/2, f"L = {span_len:.2f} m", ha='center', va='center', 
            fontsize=12, bbox=dict(facecolor='white', edgecolor='none'))
    
    # Stirrup Label
    ax.annotate(f"RB6@{s_stir}cm", xy=(L_mm/4, h_mm/2), xytext=(L_mm/4, h_mm/2 + 100),
                arrowprops=dict(arrowstyle='->', color='blue'), color='blue', ha='center')
    
    # Main Bar Labels
    ax.text(cover, h_mm+20, f"{n_top}-DB{db}", color='red', ha='left', va='bottom', fontsize=11, fontweight='bold')
    ax.text(cover, -20, f"{n_bot}-DB{db}", color='red', ha='left', va='top', fontsize=11, fontweight='bold')

    # Supports
    ax.plot([0], [-20], marker='^', markersize=10, color='black', clip_on=False)
    ax.plot([L_mm], [-20], marker='^', markersize=10, color='black', clip_on=False)

    # Clean up
    ax.set_ylim(-150, h_mm + 150)
    ax.set_xlim(-200, L_mm + 200)
    ax.axis('off')
    ax.set_title(f"LONGITUDINAL PROFILE: SPAN {span_id}", loc='left')
    
    return fig
