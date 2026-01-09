import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import numpy as np

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    วาดรูปตัดยาวคาน (Longitudinal Section) พร้อมรายละเอียดเหล็กเสริม
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

def plot_cross_section(res):
    """
    วาดรูปตัดขวางคาน (Cross Section) ฉบับจัดระเบียบใหม่ 
    ลดพื้นที่สีขาวและเน้นความคมชัดแบบ Drawing มาตรฐาน
    """
    b, h = float(res['b']), float(res['h'])
    cover = float(res['cover'])
    
    # 1. กำหนดขนาด Figure ให้กระชับ (กว้างกว่าสูงเล็กน้อยเพื่อรองรับ Text ด้านข้าง)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    x0, y0 = -b/2, -h/2
    
    # 2. วาดหน้าตัดคอนกรีต (เส้นขอบหนาชัดเจน)
    ax.add_patch(patches.Rectangle((x0, y0), b, h, facecolor='#ffffff', edgecolor='black', lw=2.5, zorder=1))
    
    # 3. วาดเหล็กปลอก
    stir_off = cover
    ax.add_patch(patches.Rectangle((x0 + stir_off, y0 + stir_off), b - 2*stir_off, h - 2*stir_off, 
                                   fill=False, edgecolor='#34495e', lw=1.2, zorder=2))
    
    # 4. เหล็กเมนบน (Top Bars)
    n_top = int(res['top']['n'])
    db_top = float(res['top_db'])
    y_top = (h/2) - stir_off - (db_top/2) - 1.5
    x_top = np.linspace(x0 + stir_off + 10, x0 + b - stir_off - 10, n_top) if n_top > 1 else [0]
    for x in x_top:
        ax.add_patch(patches.Circle((x, y_top), db_top/2 + 1.2, color='#d30000', zorder=10))
    
    # เส้นชี้เหล็กบน (จัดวางให้ไม่หลุดขอบและประหยัดพื้นที่ขาว)
    ax.annotate(f"{n_top}-DB{int(db_top)}", xy=(x_top[0], y_top), xytext=(x0 - b*0.2, h*0.4),
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=0.1", color='#d30000', lw=1),
                fontsize=10, fontweight='bold', color='#d30000', ha='right')

    # 5. เหล็กเมนล่าง (Bottom Bars)
    n_bot = int(res['bot']['n'])
    db_bot = float(res['bot_db'])
    y_bot = (-h/2) + stir_off + (db_bot/2) + 1.5
    x_bot = np.linspace(x0 + stir_off + 10, x0 + b - stir_off - 10, n_bot) if n_bot > 1 else [0]
    for x in x_bot:
        ax.add_patch(patches.Circle((x, y_bot), db_bot/2 + 1.2, color='#008c00', zorder=10))
        
    # เส้นชี้เหล็กล่าง
    ax.annotate(f"{n_bot}-DB{int(db_bot)}", xy=(x_bot[-1], y_bot), xytext=(b/2 + b*0.2, -h*0.4),
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=0.1", color='#008c00', lw=1),
                fontsize=10, fontweight='bold', color='#008c00', ha='left')

    # 6. ข้อความประกอบ (วางชิดขอบคอนกรีตที่สุดเพื่อขยับรูปขึ้น)
    ax.text(0, h/2 + 2, f"SECTION {int(b)}x{int(h)}", ha='center', va='bottom', fontweight='black', fontsize=11)
    ax.text(0, -h/2 - 2, f"RB{int(res['stir_db'])}@{int(res['shear']['s'])}", 
            ha='center', va='top', color='#555555', fontsize=9, fontweight='bold')
    
    # --- 7. Viewport Optimization ---
    ax.set_aspect('equal')
    ax.axis('off')
    
    # บีบขอบเขตแนวตั้งให้เหลือน้อยที่สุดเพื่อให้รูปขยับขึ้น
    ax.set_ylim(-h*0.62, h*0.62)
    # ขยายขอบเขตแนวนอนให้พอดีกับเส้นชี้ (แก้ปัญหาต้นฉบับที่รูปขาด)
    ax.set_xlim(-b*0.85, b*0.85)
    
    f = io.StringIO()
    # ตัดขอบขาวทิ้งทั้งหมดด้วย pad_inches=0
    fig.savefig(f, format="svg", bbox_inches='tight', pad_inches=0.01, transparent=True)
    svg_string = f.getvalue()
    plt.close(fig)
    return svg_string
