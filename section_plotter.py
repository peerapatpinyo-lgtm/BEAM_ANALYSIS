import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    Cross Section แบบ Professional: เพิ่มเส้นชี้ระบุเหล็ก (Leader Lines)
    """
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds, db = 6, db_main_mm
    
    fig, ax = plt.subplots(figsize=(5, 7))
    
    # Concrete Hatch
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='#34495e', facecolor='#f8f9fa', hatch='///', alpha=0.2))
    
    # Stirrup with Hook representation
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=1.5, edgecolor='#2c3e50', facecolor='none'))
    
    def draw_bars(n, y_pos, color):
        if n < 1: return
        spacing = (b - 2*cover - 2*ds - db) / (max(1, n - 1))
        for i in range(int(n)):
            x = (b/2) if n == 1 else (cover + ds + db/2 + i*spacing)
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=5))

    draw_bars(n_bottom, cover + ds + db/2, '#c0392b') # Bottom
    draw_bars(n_top, h - cover - ds - db/2, '#2980b9') # Top
    
    # Leader Lines (เส้นชี้เหล็ก)
    # Top Bars
    ax.annotate(f"{int(n_top)}-DB{int(db)}", xy=(b/4, h-cover-ds), xytext=(-80, h+50),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='#2980b9'),
                color='#2980b9', fontweight='bold')
    # Bottom Bars
    ax.annotate(f"{int(n_bottom)}-DB{int(db)}", xy=(b*3/4, cover+ds), xytext=(b+30, -50),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='#c0392b'),
                color='#c0392b', fontweight='bold')
    # Stirrup
    ax.annotate(f"{stirrup_name}", xy=(cover, h/2), xytext=(-100, h/2),
                arrowprops=dict(arrowstyle='->', color='#27ae60'), color='#27ae60', fontweight='bold')

    # Dimensions
    ax.annotate('', xy=(0, -25), xytext=(b, -25), arrowprops=dict(arrowstyle='<->', lw=0.8))
    ax.text(b/2, -60, f"{int(b)}", ha='center', fontsize=9)
    ax.annotate('', xy=(-25, 0), xytext=(-25, h), arrowprops=dict(arrowstyle='<->', lw=0.8))
    ax.text(-60, h/2, f"{int(h)}", va='center', rotation=90, fontsize=9)
    
    ax.set_xlim(-150, b + 150)
    ax.set_ylim(-150, h + 150)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Longitudinal แบบวิศวกร: แสดงโซนเหล็กปลอกและระยะหยุดเหล็กที่ชัดเจน
    """
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # 1. Concrete Outline
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, linewidth=2, edgecolor='black', facecolor='white', zorder=1))

    # 2. Support Columns (วาดเป็นเสาเพื่อให้ดูเหมือนแบบจริง)
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        ax.add_patch(patches.Rectangle((sx-100, -200), 200, 200, facecolor='#ecf0f1', edgecolor='#bdc3c7', hatch='...'))
        ax.text(sx, -230, f"{sup['type']}", ha='center', fontsize=8, fontweight='bold')

    # 3. Reinforcement Detailing
    for i, span_l_m in enumerate(spans):
        L_mm = span_l_m * 1000
        x_s, x_e = offsets[i], offsets[i+1]
        res = design_res[i]
        
        # Bottom Steel (Positive)
        ax.plot([x_s + 20, x_e - 20], [cover_mm, cover_mm], color='#c0392b', lw=2.5, zorder=3)
        ax.text((x_s+x_e)/2, cover_mm + 25, f"{int(res['pos']['n'])}-DB{int(res['db'])}", ha='center', color='#c0392b', fontsize=9, fontweight='bold')

        # Top Steel (Curtailment 0.3L)
        y_top = h_beam - cover_mm
        ax.plot([x_s, x_s + L_mm*0.3], [y_top, y_top], color='#2980b9', lw=2.5, zorder=3)
        ax.plot([x_e - L_mm*0.3, x_e], [y_top, y_top], color='#2980b9', lw=2.5, zorder=3)
        ax.text(x_s + 50, y_top - 35, f"{int(res['neg']['n'])}-DB{int(res['db'])}", color='#2980b9', fontsize=8)

        # Stirrup Zones (โซนถี่-ห่าง-ถี่)
        s_val = res['shear']['s']
        # โซนถี่ที่ Support
        for sx in np.linspace(x_s+50, x_s+h_beam*1.5, 4):
            ax.plot([sx, sx], [cover_mm, h_beam-cover_mm], color='#27ae60', lw=0.6, alpha=0.4)
        for sx in np.linspace(x_e-h_beam*1.5, x_e-50, 4):
            ax.plot([sx, sx], [cover_mm, h_beam-cover_mm], color='#27ae60', lw=0.6, alpha=0.4)
            
        ax.text((x_s+x_e)/2, h_beam+50, f"RB6@{int(s_val)}", ha='center', color='#27ae60', fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 4. Dimension Line
    ax.annotate('', xy=(0, -50), xytext=(total_L, -50), arrowprops=dict(arrowstyle='<->', color='black'))
    
    ax.set_xlim(-300, total_L + 300)
    ax.set_ylim(-300, h_beam + 200)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig
