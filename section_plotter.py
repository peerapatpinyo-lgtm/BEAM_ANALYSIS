import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b_m, h_m, cover_mm, db_mm, n_top, n_bot, stir_info, fc, fy):
    """
    วาด Cross Section ของคาน
    """
    b = b_m * 1000  # Convert to mm
    h = h_m * 1000
    cover = cover_mm
    
    fig, ax = plt.subplots(figsize=(4, 5))
    
    # 1. Concrete Face
    rect = patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='#333', facecolor='#f0f0f0')
    ax.add_patch(rect)
    
    # 2. Stirrup (Assume RB6 or RB9)
    stir_dia = 6 # mm
    s_w = b - 2*cover
    s_h = h - 2*cover
    stirrup = patches.Rectangle((cover, cover), s_w, s_h, 
                                linewidth=2, edgecolor='#e74c3c', facecolor='none', linestyle='-')
    ax.add_patch(stirrup)
    
    # 3. Rebars (Bottom)
    # Calculate spacing
    if n_bot > 1:
        gap_bot = (s_w - stir_dia*2 - db_mm) / (n_bot - 1)
    else:
        gap_bot = 0
        
    for i in range(n_bot):
        cx = cover + stir_dia + db_mm/2 + i*gap_bot
        cy = cover + stir_dia + db_mm/2
        circle = patches.Circle((cx, cy), db_mm/2, edgecolor='black', facecolor='#2980b9')
        ax.add_patch(circle)
        
    # 4. Rebars (Top)
    if n_top > 1:
        gap_top = (s_w - stir_dia*2 - db_mm) / (n_top - 1)
    else:
        gap_top = 0
        
    for i in range(n_top):
        cx = cover + stir_dia + db_mm/2 + i*gap_top
        cy = h - (cover + stir_dia + db_mm/2)
        circle = patches.Circle((cx, cy), db_mm/2, edgecolor='black', facecolor='#2980b9')
        ax.add_patch(circle)
        
    # Annotations
    ax.text(b/2, h/2, f"{b:.0f}x{h:.0f} mm", ha='center', va='center', fontsize=12, color='#7f8c8d')
    ax.text(b/2, -50, f"Bot: {n_bot}-DB{db_mm}\nTop: {n_top}-DB{db_mm}", ha='center', va='top', fontsize=10)
    ax.text(b/2, h+20, f"Stirrup: {stir_info}", ha='center', va='bottom', fontsize=10, color='red')

    ax.set_xlim(-50, b+50)
    ax.set_ylim(-150, h+100)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig

def plot_longitudinal_detailed(L_m, h_m, cover_mm, n_top, n_bot, db_mm, s_stir_cm, span_id):
    """
    วาดรูปด้านข้างคาน (Longitudinal)
    """
    L = L_m * 1000
    h = h_m * 1000
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Concrete Beam
    rect = patches.Rectangle((0, 0), L, h, linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(rect)
    
    # Top Bar (Simplified)
    ax.plot([50, L-50], [h-cover_mm-10, h-cover_mm-10], color='blue', linewidth=3, label='Top Bar')
    
    # Bot Bar
    ax.plot([50, L-50], [cover_mm+10, cover_mm+10], color='blue', linewidth=3, label='Bot Bar')
    
    # Stirrups
    s_mm = s_stir_cm * 10
    n_stir = int((L - 100) / s_mm)
    for i in range(n_stir + 1):
        x = 50 + i*s_mm
        ax.plot([x, x], [cover_mm, h-cover_mm], color='red', linewidth=1, linestyle='--')
        
    ax.text(L/2, h/2, f"SPAN {span_id} (L={L_m:.2f}m)", ha='center', fontsize=14, alpha=0.3)
    
    # Dimensions
    ax.annotate(f"{L_m:.2f} m", xy=(L/2, -50), ha='center')
    
    ax.set_xlim(-200, L+200)
    ax.set_ylim(-100, h+100)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig
