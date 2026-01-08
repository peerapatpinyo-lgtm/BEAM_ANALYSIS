import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    World-Class Cross Section: ISO Standard Style
    """
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds, db = 6, db_main_mm
    
    fig, ax = plt.subplots(figsize=(5, 7), dpi=100)
    
    # 1. Concrete Face & Technical Hatching
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=1.5, edgecolor='#1a1a1a', facecolor='#ffffff'))
    ax.add_patch(patches.Rectangle((0, 0), b, h, facecolor='none', hatch='////', alpha=0.05))
    
    # 2. Rebar Zone (เส้นประแสดงระยะ Cover)
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, 
                                   linewidth=0.8, edgecolor='#bdc3c7', facecolor='none', linestyle='--'))
    
    # 3. Stirrup (Solid path)
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, 
                                   linewidth=1.5, edgecolor='#2c3e50', facecolor='none'))
    
    def draw_bars(n, y_pos, color):
        if n < 1: return
        spacing = (b - 2*cover - 2*ds - db) / (max(1, n - 1))
        for i in range(int(n)):
            x = (b/2) if n == 1 else (cover + ds + db/2 + i*spacing)
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=10))

    draw_bars(n_bottom, cover + ds + db/2, '#8e1c19') # Deep Crimson
    draw_bars(n_top, h - cover - ds - db/2, '#1a5276') # Navy Blue
    
    # 4. Technical Leader Lines (สะอาดและแม่นยำ)
    # Top Bars
    ax.plot([b*0.2, -40], [h-cover, h+100], color='#1a5276', lw=0.8)
    ax.text(-50, h+110, f"{int(n_top)}-DB{int(db)}", color='#1a5276', fontweight='bold', ha='right')
    
    # Bottom Bars
    ax.plot([b*0.8, b+40], [cover, -80], color='#8e1c19', lw=0.8)
    ax.text(b+50, -100, f"{int(n_bottom)}-DB{int(db)}", color='#8e1c19', fontweight='bold', ha='left')

    # 5. Dimensions with Ticks (Style: ISO)
    def draw_dim(p1, p2, text, vertical=False):
        if vertical:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', lw=0.8)
            ax.plot([p1[0]-10, p1[0]+10], [p1[1]-10, p1[1]+10], color='black', lw=1.2) # Tick
            ax.plot([p2[0]-10, p2[0]+10], [p2[1]-10, p2[1]+10], color='black', lw=1.2) # Tick
            ax.text(p1[0]-30, (p1[1]+p2[1])/2, text, va='center', ha='right', rotation=90, fontsize=9)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', lw=0.8)
            ax.plot([p1[0]-10, p1[0]+10], [p1[1]-10, p1[1]+10], color='black', lw=1.2) # Tick
            ax.plot([p2[0]-10, p2[0]+10], [p2[1]-10, p2[1]+10], color='black', lw=1.2) # Tick
            ax.text((p1[0]+p2[0])/2, p1[1]-30, text, ha='center', va='top', fontsize=9)

    draw_dim([0, -40], [b, -40], f"{int(b)}")
    draw_dim([-40, 0], [-40, h], f"{int(h)}", vertical=True)
    
    # Data Table (Minimalist bottom right)
    props = f"f'c: {fc} MPa\nfy: {fy} MPa\nST: {stirrup_name}"
    ax.text(b + 50, h/2, props, va='center', fontsize=8, color='#566573', linespacing=1.8)

    ax.set_xlim(-180, b + 220)
    ax.set_ylim(-180, h + 220)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Advanced Longitudinal Section: Construction Profile
    """
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    fig, ax = plt.subplots(figsize=(15, 4), dpi=100)
    
    # 1. Main Beam (Double line for top/bottom face)
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, lw=1.2, edgecolor='#1a1a1a', facecolor='#ffffff'))
    
    # 2. Structural Supports (Poured Concrete Columns)
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        ax.add_patch(patches.Rectangle((sx-80, -h_beam*0.7), 160, h_beam*0.7, 
                                       facecolor='#f2f4f4', edgecolor='#7f8c8d', lw=0.8))
        ax.text(sx, -h_beam*1.0, f"SUPPORT\n({sup['type']})", ha='center', fontsize=7, color='#7f8c8d')

    # 3. Bar Detailing with Precision
    for i, span_l_m in enumerate(spans):
        L_mm = span_l_m * 1000
        x_s, x_e = offsets[i], offsets[i+1]
        res = design_res[i]
        
        # Positive Reinforcement (Bottom)
        ax.plot([x_s+30, x_e-30], [cover_mm, cover_mm], color='#8e1c19', lw=2, zorder=5)
        ax.text((x_s+x_e)/2, cover_mm + 15, f"{int(res['pos']['n'])}-DB{int(res['db'])}", 
                ha='center', color='#8e1c19', fontsize=8, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, lw=0))

        # Negative Reinforcement (Top - L/4 Curtailment)
        ax.plot([x_s, x_s + L_mm*0.25], [h_beam-cover_mm, h_beam-cover_mm], color='#1a5276', lw=2, zorder=5)
        ax.plot([x_e - L_mm*0.25, x_e], [h_beam-cover_mm, h_beam-cover_mm], color='#1a5276', lw=2, zorder=5)
        
        # Shear Detailing (Stirrups) - Indicating Zones
        ax.text((x_s+x_e)/2, h_beam + 50, f"RB6 @{int(res['shear']['s'])} mm", 
                ha='center', color='#1d8348', fontsize=8, fontweight='bold')
        
        # Fine stirrup lines at supports
        stirrup_pos = list(np.linspace(x_s+50, x_s+h_beam, 4)) + list(np.linspace(x_e-h_beam, x_e-50, 4))
        for sx in stirrup_pos:
            ax.plot([sx, sx], [cover_mm, h_beam-cover_mm], color='#1d8348', lw=0.5, alpha=0.2)

    # 4. ISO Dimensioning (Span length)
    for i in range(len(spans)):
        x1, x2 = offsets[i], offsets[i+1]
        ax.plot([x1, x2], [-h_beam*0.4, -h_beam*0.4], color='#566573', lw=0.7)
        ax.plot([x1-5, x1+5], [-h_beam*0.4-5, -h_beam*0.4+5], color='#566573', lw=1)
        ax.plot([x2-5, x2+5], [-h_beam*0.4-5, -h_beam*0.4+5], color='#566573', lw=1)
        ax.text((x1+x2)/2, -h_beam*0.6, f"SPAN {i+1}: {spans[i]}m", ha='center', fontsize=8, color='#566573')

    ax.set_xlim(-400, total_L + 400)
    ax.set_ylim(-h_beam*1.5, h_beam*2)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig
