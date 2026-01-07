import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b, h, cover, db, n_top, n_bot, stir_label, fc, fy):
    """ Cross Section View """
    fig, ax = plt.subplots(figsize=(4, 5))
    
    # Concrete Box
    rect = patches.Rectangle((0, 0), b*1000, h*1000, linewidth=2, edgecolor='#333', facecolor='#f0f0f0')
    ax.add_patch(rect)
    
    # Stirrup
    c = cover
    w_s = (b*1000) - 2*c
    h_s = (h*1000) - 2*c
    stir = patches.Rectangle((c, c), w_s, h_s, linewidth=2, edgecolor='blue', facecolor='none', linestyle='-')
    ax.add_patch(stir)
    
    # Rebars (Top)
    spacing_top = w_s / (n_top + 1) if n_top > 1 else w_s/2
    for i in range(n_top):
        cx = c + (spacing_top * (i+1)) if n_top > 1 else (b*1000)/2
        cy = (h*1000) - c - (db/2) - 6 # 6mm est stirrup dia
        circle = patches.Circle((cx, cy), db/2, color='red')
        ax.add_patch(circle)
        
    # Rebars (Bot)
    spacing_bot = w_s / (n_bot + 1) if n_bot > 1 else w_s/2
    for i in range(n_bot):
        cx = c + (spacing_bot * (i+1)) if n_bot > 1 else (b*1000)/2
        cy = c + (db/2) + 6
        circle = patches.Circle((cx, cy), db/2, color='red')
        ax.add_patch(circle)
        
    ax.set_xlim(-50, b*1000 + 50)
    ax.set_ylim(-50, h*1000 + 50)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"{int(b*100)}x{int(h*100)} cm\n{n_top}-DB{db} (Top) / {n_bot}-DB{db} (Bot)\nStirrup: {stir_label}", fontsize=10)
    return fig

def plot_longitudinal_detailed(span_len, h, cover, n_top, n_bot, db, s_stir, span_id):
    """ 
    Detailed Longitudinal Profile for a single span 
    Shows actual stirrup spacing lines.
    """
    L_mm = span_len * 1000
    h_mm = h * 1000
    c = cover
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # 1. Concrete Beam
    beam = patches.Rectangle((0, 0), L_mm, h_mm, linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(beam)
    
    # 2. Main Bars (Schematic Lines)
    # Top
    ax.plot([c, L_mm-c], [h_mm-c-10, h_mm-c-10], color='red', linewidth=3, label='Top Bars')
    # Bot
    ax.plot([c, L_mm-c], [c+10, c+10], color='red', linewidth=3, label='Bot Bars')
    
    # 3. Stirrups (Vertical Lines at spacing s)
    s_mm = s_stir * 10 # cm to mm
    # Generate stirrup positions
    x_stir = np.arange(c + 50, L_mm - c - 50, s_mm)
    
    for x in x_stir:
        ax.plot([x, x], [c, h_mm-c], color='blue', linewidth=1, linestyle='-')
        
    # 4. Annotations
    # Center text
    ax.text(L_mm/2, h_mm/2, f"SPAN {span_id} (L = {span_len:.2f} m)", 
            ha='center', va='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
    
    # Bar Labels
    ax.text(L_mm*0.1, h_mm - c - 40, f"{n_top}-DB{db}", color='red', fontsize=9, ha='left')
    ax.text(L_mm*0.1, c + 40, f"{n_bot}-DB{db}", color='red', fontsize=9, ha='left')
    
    # Stirrup Label with Arrow
    mid_idx = len(x_stir)//2
    if mid_idx < len(x_stir):
        x_lbl = x_stir[mid_idx]
        ax.annotate(f"RB6 @ {s_stir} cm", xy=(x_lbl, h_mm/2), xytext=(x_lbl+200, h_mm/2),
                    arrowprops=dict(arrowstyle='->', color='blue'), color='blue', fontsize=9)

    # 5. Supports (Triangles at ends)
    ax.plot([0], [-50], marker='^', markersize=15, color='black', clip_on=False)
    ax.plot([L_mm], [-50], marker='^', markersize=15, color='black', clip_on=False)
    
    ax.set_xlim(-200, L_mm + 200)
    ax.set_ylim(-100, h_mm + 100)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    return fig
