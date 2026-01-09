import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import numpy as np

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    (คงเดิม 100% ตามต้นฉบับที่คุณส่งมา) วาดรูปตัดยาวคาน
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 350  
    
    fig_w = max(16, total_L / 350)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    
    beam = patches.Rectangle((0, 0), total_L, v_h, lw=2, ec='black', fc='white', antialiased=False, zorder=2)
    ax.add_patch(beam)
    
    curr_x = 0
    for i, s_mm in enumerate(spans_mm + [0]):
        ax.plot([curr_x, curr_x], [-600, v_h + 400], color='#7f8c8d', ls='-.', lw=1, zorder=1)
        ax.annotate(chr(65+i), xy=(curr_x, v_h + 500), ha='center', va='center',
                    bbox=dict(boxstyle='circle', fc='white', ec='black', lw=1.5), 
                    fontsize=14, fontweight='bold')
        
        if i < len(spans_mm):
            ax.annotate('', xy=(curr_x, v_h + 250), xytext=(curr_x + s_mm, v_h + 250),
                        arrowprops=dict(arrowstyle='<->', color='#2980b9', lw=1.2))
            ax.text(curr_x + s_mm/2, v_h + 300, f"{s_mm/1000:.2f} m", 
                    ha='center', color='#2980b9', fontsize=12, fontweight='bold')
            curr_x += s_mm

    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            sx = row['x'] * 1000
            stype = str(row.get('type', 'PIN')).upper()
            if stype == 'FIXED':
                ax.add_patch(patches.Rectangle((sx-100, -350), 200, 350, fc='#dfe6e9', ec='black', lw=1.5, hatch='////'))
            elif stype == 'ROLLER':
                ax.add_patch(patches.Polygon([[sx, 0], [sx-90, -180], [sx+90, -180]], fc='white', ec='black', lw=1.5))
                ax.add_patch(patches.Circle((sx, -215), 30, fc='black'))
            else: # PIN
                ax.add_patch(patches.Polygon([[sx, 0], [sx-90, -180], [sx+90, -180]], fc='#2c3e50', ec='black', lw=1.5))
            ax.text(sx, -500, f"S{row['id']}: {stype}", ha='center', fontweight='bold', fontsize=10)

    y_t, y_b = v_h * 0.82, v_h * 0.18
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        
        s_spacing = res['shear']['s']
        num_stirrups = int(span_L / s_spacing)
        for j in range(num_stirrups + 1):
            stir_x = x_curr + (j * s_spacing)
            if stir_x <= x_curr + span_L:
                ax.plot([stir_x, stir_x], [y_b - 20, y_t + 20], color='#bdc3c7', lw=0.7, alpha=0.6, zorder=3)
        
        mid = x_curr + span_L/2
        ax.text(mid, -150, f"RB{int(res['stir_db'])}@{int(s_spacing)}", color='#7f8c8d', fontsize=9, ha='center', style='italic')

        ax.plot([x_curr, x_curr + span_L], [y_t, y_t], color='#d30000', lw=3.5, zorder=10, antialiased=False)
        ax.plot([x_curr + 40, x_curr + span_L - 40], [y_b, y_b], color='#008c00', lw=3.5, zorder=10, antialiased=False)
        
        label_opt = dict(ha='center', fontweight='bold', fontsize=11, bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))
        ax.text(mid, v_h + 80, f"{int(res['neg']['n'])}-DB{int(res['top_db'])} (TOP)", color='#d30000', **label_opt)
        ax.text(mid, y_b - 50, f"{int(res['pos']['n'])}-DB{int(res['bot_db'])} (BOT)", color='#008c00', va='top', **label_opt)
        x_curr += span_L

    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-1000, total_L + 1000)
    ax.set_ylim(-800, v_h + 800)
    
    f_svg = io.StringIO()
    fig.savefig(f_svg, format="svg", bbox_inches='tight')
    svg_string = f_svg.getvalue()

    f_png = io.BytesIO()
    fig.savefig(f_png, format="png", dpi=300, bbox_inches='tight')
    png_bytes = f_png.getvalue()

    plt.close(fig)
    return svg_string, png_bytes

# --- 💎 แก้ไขฟังก์ชัน Cross Section ให้แสดงผลครบและระบุเหล็กกำกับ ---
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import numpy as np

def plot_cross_section(res):
    """
    วาดรูปตัดขวางคาน (Cross Section) โดยขยับตำแหน่งให้เห็นเหล็กล่างชัดเจน
    และกำจัดพื้นที่ว่างด้านบน (Crop ส่วนเกินออก)
    """
    b = float(res['b'])
    h = float(res['h'])
    cover = float(res['cover'])
    
    # 1. สร้าง Figure (เน้นแนวตั้ง)
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # 2. วาดหน้าตัดคอนกรีต (ใช้พิกัด 0,0 เป็นมุมซ้ายล่าง)
    # เพิ่มความหนาของเส้นขอบเพื่อให้เห็นชัดเจน
    ax.add_patch(patches.Rectangle((0, 0), b, h, facecolor='#ffffff', edgecolor='black', lw=2, zorder=1))
    
    # 3. วาดเหล็กปลอก (Stirrup)
    stir_off = cover
    ax.add_patch(patches.Rectangle((stir_off, stir_off), b-2*stir_off, h-2*stir_off, 
                                   fill=False, edgecolor='#2c3e50', lw=1.2, zorder=2))
    
    # 4. วาดและระบุเหล็กเมนบน (Top Bars)
    n_top = int(res['top']['n'])
    db_top = float(res['top_db'])
    y_top = h - stir_off - (db_top/2) - 2
    x_top = np.linspace(stir_off + 12, b - stir_off - 12, n_top) if n_top > 1 else [b/2]
    
    for x in x_top:
        ax.add_patch(patches.Circle((x, y_top), db_top/2 + 1, color='#d30000', zorder=10))
    
    # Label เหล็กบน (ชี้ออกไปทางซ้ายบน)
    ax.annotate(f"{n_top}-DB{int(db_top)}", xy=(x_top[0], y_top), xytext=(-b*0.1, h * 1.05),
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-0.1", color='#d30000'),
                fontsize=11, fontweight='bold', color='#d30000', ha='right')

    # 5. วาดและระบุเหล็กเมนล่าง (Bottom Bars)
    n_bot = int(res['bot']['n'])
    db_bot = float(res['bot_db'])
    y_bot = stir_off + (db_bot/2) + 2  # ตำแหน่งเหล็กล่าง
    x_bot = np.linspace(stir_off + 12, b - stir_off - 12, n_bot) if n_bot > 1 else [b/2]
    
    for x in x_bot:
        ax.add_patch(patches.Circle((x, y_bot), db_bot/2 + 1, color='#008c00', zorder=10))
        
    # Label เหล็กล่าง (ชี้ออกไปทางขวาข้างๆ คาน เพื่อประหยัดพื้นที่แนวตั้ง)
    ax.annotate(f"{n_bot}-DB{int(db_bot)}", xy=(x_bot[-1], y_bot), xytext=(b*1.1, y_bot),
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=0", color='#008c00'),
                fontsize=11, fontweight='bold', color='#008c00', ha='left', va='center')

    # 6. ข้อความกำกับขนาดและเหล็กปลอก (วางชิดขอบ)
    ax.text(b/2, h + (h*0.12), f"SECTION {int(b)}x{int(h)} mm", ha='center', fontweight='bold', fontsize=12)
    ax.text(b/2, - (h*0.08), f"RB{int(res['stir_db'])}@{int(res['shear']['s'])}", 
            ha='center', color='#34495e', fontsize=10, fontweight='bold')
    
    # --- ปรับแต่ง Viewport (หัวใจสำคัญของการแก้ปัญหา) ---
    ax.set_aspect('equal')
    ax.axis('off')
    
    # xlim: ให้เผื่อด้านซ้ายสำหรับ label บน และด้านขวาสำหรับ label ล่าง
    ax.set_xlim(-b*0.4, b*1.5)
    
    # ylim: บีบพื้นที่ด้านบนลง (-0.15 คือเผื่อด้านล่างให้เห็นเหล็กและข้อความครบ)
    # และ 1.25 คือเผื่อด้านบนให้เห็นข้อความหัวข้อ
    ax.set_ylim(-h*0.15, h*1.25)
    
    f = io.StringIO()
    # ใช้ bbox_inches='tight' พร้อมกำหนด pad_inches ให้เหลือน้อยที่สุด
    fig.savefig(f, format="svg", bbox_inches='tight', pad_inches=0.1, transparent=True)
    svg_string = f.getvalue()
    plt.close(fig)
    return svg_string
