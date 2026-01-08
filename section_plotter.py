import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    Matched-Scale Cross Section Detail.
    """
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds, db = 6, db_main_mm
    
    # ปรับ figsize ให้เล็กลงเพื่อไม่ให้ดูใหญ่กว่ารูปตัดตามยาวเกินไป
    fig, ax = plt.subplots(figsize=(4, 5.5)) 
    
    # Concrete Hatch
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='#f8f9fa', hatch='///', alpha=0.15))
    
    # Stirrup
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=1.2, edgecolor='#2c3e50', facecolor='none'))
    
    def draw_bars(n, y_pos, color):
        if n < 1: return
        spacing = (b - 2*cover - 2*ds - db) / (max(1, n - 1))
        for i in range(int(n)):
            x = (b/2) if n == 1 else (cover + ds + db/2 + i*spacing)
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=5))

    draw_bars(n_bottom, cover + ds + db/2, '#c0392b')
    draw_bars(n_top, h - cover - ds - db/2, '#2980b9')
    
    # Label Placement สัมพันธ์กับขนาดคาน
    label_y = h + 60
    ax.text(b/2, label_y, f"SECTION {int(b)}x{int(h)}mm\nfc' {fc} | fy {fy}", 
            ha='center', va='bottom', fontsize=9, fontweight='bold', family='monospace')
    
    # Dimensions
    ax.annotate('', xy=(0, -25), xytext=(b, -25), arrowprops=dict(arrowstyle='<->', lw=0.8))
    ax.text(b/2, -55, f"b={int(b)}", ha='center', fontsize=8)
    
    ax.set_xlim(-100, b + 100)
    ax.set_ylim(-120, h + 200)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Matched-Scale Longitudinal Detail.
    - Labels are moved outside for clarity.
    - Support symbols scaled to beam height.
    """
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    fig, ax = plt.subplots(figsize=(15, 4.5)) # ความกว้าง 15 นิ้วเพื่อให้เห็นคานยาวชัดเจน
    
    # Concrete Body
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, linewidth=2, edgecolor='#2c3e50', facecolor='#ffffff', zorder=1))

    # Support Drawing
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        if sup['type'] == 'Fixed':
            ax.add_patch(patches.Rectangle((sx-20, -10), 40, h_beam+20, facecolor='#bdc3c7', hatch='///', alpha=0.5))
        else:
            poly = plt.Polygon([[sx-80, -80], [sx+80, -80], [sx, 0]], closed=True, facecolor='#f8f9fa', edgecolor='black', lw=1)
            ax.add_patch(poly)

    # Rebar Detailing
    for i, span_l_m in enumerate(spans):
        L_mm = span_l_m * 1000
        x_s, x_e = offsets[i], offsets[i+1]
        mid_x = (x_s + x_e) / 2
        res = design_res[i]
        
        # Bottom Steel - Label moved down
        ax.plot([x_s + 30, x_e - 30], [cover_mm, cover_mm], color='#c0392b', lw=2.5, solid_capstyle='round')
        ax.text(mid_x, -50, f"{int(res['pos']['n'])}-DB{int(res['db'])}", 
                ha='center', va='top', color='#c0392b', fontsize=9, fontweight='bold')

        # Top Steel - Label moved up
        y_top = h_beam - cover_mm
        ax.plot([x_s, x_s + L_mm*0.25], [y_top, y_top], color='#2980b9', lw=2.5)
        ax.plot([x_e - L_mm*0.25, x_e], [y_top, y_top], color='#2980b9', lw=2.5)
        ax.text(x_s + 30, h_beam + 30, f"{int(res['neg']['n'])}-DB{int(res['db'])}", 
                ha='left', va='bottom', color='#2980b9', fontsize=9, fontweight='bold')

        # Stirrups - Using Leader Lines to stay outside
        ax.annotate(f"RB6 @{int(res['shear']['s'])}", xy=(mid_x, h_beam/2), xytext=(mid_x, -140),
                    arrowprops=dict(arrowstyle='-', color='#27ae60', lw=0.6, alpha=0.5),
                    ha='center', color='#27ae60', fontsize=8, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.7))
        
        # Stirrup symbols inside
        for sx in np.linspace(x_s + 150, x_e - 150, 6):
            ax.plot([sx, sx], [cover_mm, h_beam-cover_mm], color='#27ae60', lw=0.6, alpha=0.3)

    # Span Dimensions
    for i in range(len(spans)):
        ax.annotate('', xy=(offsets[i], h_beam + 100), xytext=(offsets[i+1], h_beam + 100),
                    arrowprops=dict(arrowstyle='<->', color='#bdc3c7', lw=0.8))
        ax.text((offsets[i]+offsets[i+1])/2, h_beam + 120, f"{spans[i]}m", ha='center', fontsize=8)

    ax.set_xlim(-300, total_L + 300)
    ax.set_ylim(-200, h_beam + 250)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig
