import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    แบบดั้งเดิมที่คุณชอบ - เน้นความชัดเจนและข้อมูลครบถ้วนข้างรูป
    """
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds, db = 6, db_main_mm
    
    fig, ax = plt.subplots(figsize=(5, 7))
    
    # Concrete Hatch
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='#f8f9fa', hatch='///', alpha=0.3))
    
    # Stirrup
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=1.5, edgecolor='#2c3e50', facecolor='none'))
    
    def draw_bars(n, y_pos, color):
        if n < 1: return
        spacing = (b - 2*cover - 2*ds - db) / (max(1, n - 1))
        for i in range(int(n)):
            x = (b/2) if n == 1 else (cover + ds + db/2 + i*spacing)
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=5))

    draw_bars(n_bottom, cover + ds + db/2, '#c0392b')
    draw_bars(n_top, h - cover - ds - db/2, '#2980b9')
    
    # ข้อมูลประกอบด้านข้าง (แบบเดิมที่คุณชอบ)
    info_text = (
        f"SECTION DETAIL\n"
        f"Size: {int(b)}x{int(h)} mm\n"
        f"Concrete: {fc} MPa\n"
        f"Steel: {fy} MPa\n"
        f"Top: {int(n_top)}-DB{int(db)}\n"
        f"Bottom: {int(n_bottom)}-DB{int(db)}\n"
        f"Stirrup: {stirrup_name}"
    )
    plt.text(b + 40, h, info_text, va='top', family='monospace', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#bdc3c7'))
    
    # Dimensions
    ax.annotate('', xy=(0, -30), xytext=(b, -30), arrowprops=dict(arrowstyle='<->'))
    ax.text(b/2, -70, f"{int(b)}", ha='center', size=9)
    ax.annotate('', xy=(-30, 0), xytext=(-30, h), arrowprops=dict(arrowstyle='<->'))
    ax.text(-80, h/2, f"{int(h)}", va='center', rotation=90, size=9)
    
    ax.set_xlim(-120, b + 300)
    ax.set_ylim(-150, h + 100)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    แบบดั้งเดิมที่ปรับปรุง Support และแก้ตัวหนังสือทับกัน
    """
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # ตัวคาน
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, linewidth=2, edgecolor='black', facecolor='white'))
    
    for i, span_l_m in enumerate(spans):
        L_mm = span_l_m * 1000
        x_s, x_e = offsets[i], offsets[i+1]
        mid_x = (x_s + x_e) / 2
        res = design_res[i]
        
        # เหล็กล่าง - วาดเลขไว้ "ใต้เหล็ก"
        ax.plot([x_s+30, x_e-30], [cover_mm, cover_mm], color='#c0392b', lw=2.5)
        ax.text(mid_x, cover_mm + 20, f"{int(res['pos']['n'])}-DB{int(res['db'])}", ha='center', color='#c0392b', fontsize=10, fontweight='bold')
        
        # เหล็กบน - วาดเลขไว้ "บนเหล็ก"
        ax.plot([x_s, x_s + L_mm*0.3], [h_beam-cover_mm, h_beam-cover_mm], color='#2980b9', lw=2.5)
        ax.plot([x_e - L_mm*0.3, x_e], [h_beam-cover_mm, h_beam-cover_mm], color='#2980b9', lw=2.5)
        ax.text(x_s + 50, h_beam-cover_mm-40, f"{int(res['neg']['n'])}-DB{int(res['db'])}", color='#2980b9', fontsize=9, fontweight='bold')
        
        # เหล็กปลอก (แก้ตัวหนังสือทับกันโดยการย้ายไปไว้ "ขอบบน" ของคานในแต่ละ Span)
        s_val = res['shear']['s']
        # วาดเส้นสัญลักษณ์ปลอก
        for sx in np.linspace(x_s + 200, x_e - 200, 6):
            ax.plot([sx, sx], [cover_mm, h_beam-cover_mm], color='#27ae60', lw=0.8, alpha=0.3)
        
        # [จุดที่แก้] ย้ายป้ายเหล็กปลอกไปไว้ด้านบนสุดของคาน เพื่อไม่ให้ทับกับเหล็กเสริมหลักตรงกลาง
        ax.text(mid_x, h_beam - 25, f"RB6@{int(s_val)}", ha='center', va='top', color='#27ae60', 
                fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # วาด Support ตามจริง
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        # สัญลักษณ์ Support พื้นฐานที่ดูง่าย
        ax.plot([sx, sx], [0, -80], 'k-', lw=2)
        ax.plot([sx-80, sx+80], [-80, -80], 'k-', lw=3)
        ax.text(sx, -120, f"{sup['type']}", ha='center', fontsize=8)

    ax.set_xlim(-200, total_L + 200)
    ax.set_ylim(-200, h_beam + 100)
    ax.axis('off')
    return fig
