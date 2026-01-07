import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b, h, cover, db, n_top, n_bot, stir_label, fc, fy):
    """ 
    Engineering Cross Section 
    """
    fig, ax = plt.subplots(figsize=(4, 5)) # Slightly compact to fit side-by-side
    
    # Convert m to mm for drawing
    B, H = b*1000, h*1000
    c = cover
    
    # 1. Concrete
    rect = patches.Rectangle((0, 0), B, H, linewidth=2, edgecolor='black', facecolor='#f9f9f9')
    ax.add_patch(rect)
    
    # 2. Stirrup
    db_stir = 6
    w_s = B - 2*c
    h_s = H - 2*c
    stir = patches.Rectangle((c, c), w_s, h_s, linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(stir)
    
    # 3. Main Bars (Logic: Always at Corners first)
    # Top
    y_top = H - c - db/2 - db_stir
    if n_top < 2: n_top = 2 # Minimum safety
    x_tops = np.linspace(c + db_stir + db/2, B - c - db_stir - db/2, n_top)
    
    for x in x_tops:
        circle = patches.Circle((x, y_top), db/2, edgecolor='black', facecolor='red', zorder=5)
        ax.add_patch(circle)
        
    # Bot
    y_bot = c + db_stir + db/2
    if n_bot < 2: n_bot = 2
    x_bots = np.linspace(c + db_stir + db/2, B - c - db_stir - db/2, n_bot)
    
    for x in x_bots:
        circle = patches.Circle((x, y_bot), db/2, edgecolor='black', facecolor='red', zorder=5)
        ax.add_patch(circle)

    # 4. Annotations
    ax.text(B/2, H+20, f"{int(B)}", ha='center', fontsize=11)
    ax.text(-20, H/2, f"{int(H)}", va='center', rotation=90, fontsize=11)
    
    ax.text(B/2, H/2, f"{n_top}-DB{db} (Top)\n{n_bot}-DB{db} (Bot)\n{stir_label}", 
            ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), fontsize=9)

    ax.set_xlim(-50, B+50)
    ax.set_ylim(-50, H+50)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_detailed(span_len, h, cover, n_top, n_bot, db, s_stir, span_id):
    """
    Clean Longitudinal Profile
    - Fix: Stirrups won't overlap into a blob.
    - Fix: Clean dimension lines.
    """
    L_mm = span_len * 1000
    h_mm = h * 1000
    
    # Create Figure (Wide aspect)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    
    # 1. Beam Body
    ax.plot([0, L_mm], [0, 0], 'k-', linewidth=1.5)
    ax.plot([0, L_mm], [h_mm, h_mm], 'k-', linewidth=1.5)
    ax.plot([0, 0], [0, h_mm], 'k--', linewidth=1)
    ax.plot([L_mm, L_mm], [0, h_mm], 'k--', linewidth=1)
    
    # 2. Main Bars (Red)
    ax.plot([cover, L_mm-cover], [h_mm-cover-10, h_mm-cover-10], 'r-', linewidth=2, label='Top')
    ax.plot([cover, L_mm-cover], [cover+10, cover+10], 'r-', linewidth=2, label='Bot')
    
    # 3. Stirrups (Blue - Thinner & Clean)
    s_mm = s_stir * 10
    # Avoid drawing if spacing is too dense relative to pixel size, but for matplotlib vector it's fine.
    # Just make them thinner and distinct.
    x_stir = np.arange(cover + 50, L_mm - cover - 50, s_mm)
    
    # Use vlines for better performance and look
    ax.vlines(x_stir, ymin=cover, ymax=h_mm-cover, colors='blue', linewidth=0.6, alpha=0.7)
    
    # 4. Dimensions & Labels
    # Mid-span Text
    ax.text(L_mm/2, h_mm + 50, f"Span {span_id}: L = {span_len:.2f} m", 
            ha='center', fontsize=12, fontweight='bold', color='#333')
    
    # Rebar Labels (with Leader Lines)
    ax.annotate(f"{n_top}-DB{db}", xy=(L_mm*0.2, h_mm-cover), xytext=(L_mm*0.2, h_mm+100),
                arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=10)
    
    ax.annotate(f"{n_bot}-DB{db}", xy=(L_mm*0.2, cover), xytext=(L_mm*0.2, -80),
                arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=10)
    
    # Stirrup Label (Point to one stirrup)
    if len(x_stir) > 0:
        idx = len(x_stir)//2
        ax.annotate(f"RB6@{s_stir}cm", xy=(x_stir[idx], h_mm/2), xytext=(x_stir[idx]+150, h_mm/2),
                    arrowprops=dict(arrowstyle='->', color='blue'), color='blue', fontsize=10, bbox=dict(facecolor='white', edgecolor='none'))

    # Supports
    ax.plot(0, -20, marker='^', color='black', markersize=10, clip_on=False)
    ax.plot(L_mm, -20, marker='^', color='black', markersize=10, clip_on=False)

    ax.set_ylim(-150, h_mm + 200)
    ax.set_xlim(-200, L_mm + 200)
    ax.axis('off')
    
    return fig
