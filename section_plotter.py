import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    Cross Section: Clean & Professional Consultant Style
    """
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds, db = 6, db_main_mm
    
    fig, ax = plt.subplots(figsize=(5, 7))
    
    # 1. Main Concrete Section (Soft gray shade)
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='#2c3e50', facecolor='#fdfdfd', zorder=1))
    # Light Hatch for concrete texture
    ax.add_patch(patches.Rectangle((0, 0), b, h, facecolor='none', hatch='....', alpha=0.1, zorder=2))
    
    # 2. Stirrup (Solid line)
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=1.5, edgecolor='#34495e', facecolor='none', zorder=3))
    
    # 3. Main Reinforcement
    def draw_bars(n, y_pos, color):
        if n < 1: return
        spacing = (b - 2*cover - 2*ds - db) / (max(1, n - 1))
        for i in range(int(n)):
            x = (b/2) if n == 1 else (cover + ds + db/2 + i*spacing)
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=10))

    draw_bars(n_bottom, cover + ds + db/2, '#b03a2e') # Bottom Bars
    draw_bars(n_top, h - cover - ds - db/2, '#2e86c1') # Top Bars
    
    # 4. Professional Leader Lines (L-Shape)
    # Top Bars Label
    ax.annotate(f"{int(n_top)}-DB{int(db)}", xy=(b*0.3, h-cover), xytext=(-60, h+80),
                arrowprops=dict(arrowstyle='-', connectionstyle='angle,angleA=0,angleB=90,rad=0', color='#2e86c1', lw=1),
                color='#2e86c1', fontweight='bold', fontsize=10)
    
    # Bottom Bars Label
    ax.annotate(f"{int(n_bottom)}-DB{int(db)}", xy=(b*0.7, cover), xytext=(b+40, -80),
                arrowprops=dict(arrowstyle='-', connectionstyle='angle,angleA=0,angleB=90,rad=0', color='#b03a2e', lw=1),
                color='#b03a2e', fontweight='bold', fontsize=10)
    
    # Material Info (Clean bottom right)
    mat_text = f"f'c: {fc} MPa\nfy: {fy} MPa\n{stirrup_name}"
    ax.text(b + 40, h/2, mat_text, va='center', fontsize=9, color='#7f8c8d', family='sans-serif')

    # Dimensions (Architecture style)
    ax.plot([0, b], [-40, -40], color='#95a5a6', lw=0.8)
    ax.text(b/2, -70, f"{int(b)}", ha='center', fontsize=9, color='#7f8c8d')
    ax.plot([-40, -40], [0, h], color='#95a5a6', lw=0.8)
    ax.text(-80, h/2, f"{int(h)}", va='center', rotation=90, fontsize=9, color='#7f8c8d')
    
    ax.set_xlim(-150, b + 250)
    ax.set_ylim(-150, h + 200)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Longitudinal: Minimalist Engineering Profile
    """
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # 1. Beam Body
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, linewidth=1.5, edgecolor='#2c3e50', facecolor='white'))
    
    # 2. Support Columns (Clean pillars)
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        ax.add_patch(patches.Rectangle((sx-60, -h_beam*0.8), 120, h_beam*0.8, facecolor='#ecf0f1', edgecolor='#bdc3c7'))
        ax.text(sx, -h_beam*1.1, f"{sup['type']}", ha='center', fontsize=8, color='#7f8c8d')

    # 3. Bars & Stirrup Info
    for i, span_l_m in enumerate(spans):
        L_mm = span_l_m * 1000
        x_start, x_end = offsets[i], offsets[i+1]
        mid_x = (x_start + x_end) / 2
        res = design_res[i]
        
        # Continuous Bottom Steel
        ax.plot([x_start+20, x_end-20], [cover_mm, cover_mm], color='#b03a2e', lw=2, alpha=0.9)
        ax.text(mid_x, cover_mm + 20, f"{int(res['pos']['n'])}-DB{int(res['db'])}", ha='center', color='#b03a2e', fontsize=8, fontweight='bold')

        # Top Steel (Curtailment)
        ax.plot([x_start, x_start + L_mm*0.25], [h_beam-cover_mm, h_beam-cover_mm], color='#2e86c1', lw=2)
        ax.plot([x_end - L_mm*0.25, x_end], [h_beam-cover_mm, h_beam-cover_mm], color='#2e86c1', lw=2)
        
        # Stirrup Info (Clean label)
        ax.text(mid_x, h_beam + 40, f"RB6@{int(res['shear']['s'])} mm", ha='center', color='#27ae60', fontsize=8, fontweight='bold')
        # Visual Stirrups (Thin lines)
        for sx in np.linspace(x_start + 100, x_end - 100, 8):
            ax.plot([sx, sx], [cover_mm, h_beam-cover_mm], color='#27ae60', lw=0.5, alpha=0.2)

    # 4. Span Dimensions (Bottom)
    for i in range(len(spans)):
        ax.annotate('', xy=(offsets[i], -50), xytext=(offsets[i+1], -50), arrowprops=dict(arrowstyle='<->', color='#bdc3c7', lw=0.7))
        ax.text((offsets[i]+offsets[i+1])/2, -100, f"L = {spans[i]} m", ha='center', fontsize=8, color='#7f8c8d')

    ax.set_xlim(-200, total_L + 200)
    ax.set_ylim(-h_beam*1.5, h_beam*2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig
