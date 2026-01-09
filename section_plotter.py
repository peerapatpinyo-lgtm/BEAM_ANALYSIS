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
def plot_cross_section(res):
    """
    วาดรูปตัดขวางคาน (Cross Section) ให้เต็มพื้นที่ และจัดสมดุล Margin ใหม่
    """
    b = float(res['b'])
    h = float(res['h'])
    cover = float(res['cover'])
    
    # สร้าง Figure ให้กระชับขึ้น
    fig, ax = plt.subplots(figsize=(4, 5))
    
    # 1. วาดหน้าตัดคอนกรีต (ใช้พิกัดเริ่มที่ 0,0)
    ax.add_patch(patches.Rectangle((0, 0), b, h, facecolor='#fdfdfd', edgecolor='black', lw=2.5, zorder=1))
    
    # 2. วาดเหล็กปลอก (Stirrup)
    stir_off = cover
    ax.add_patch(patches.Rectangle((stir_off, stir_off), b-2*stir_off, h-2*stir_off, 
                                   fill=False, edgecolor='#34495e', lw=1.5, zorder=2))
    
    # 3. เหล็กบน (Top Bars)
    n_top = int(res['top']['n'])
    db_top = float(res['top_db'])
    y_top = h - stir_off - (db_top/2) - 2 # ขยับลงมาจากขอบปลอกเล็กน้อย
    x_top = np.linspace(stir_off + 12, b - stir_off - 12, n_top) if n_top > 1 else [b/2]
    
    for x in x_top:
        ax.add_patch(patches.Circle((x, y_top), db_top/2 + 1, color='#d30000', zorder=10))
    
    # เส้นชี้เหล็กบน (ชี้ออกด้านซ้าย)
    ax.annotate(f"{n_top}-DB{int(db_top)}", xy=(x_top[0], y_top), xytext=(-b*0.35, h * 0.9),
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=0.1", color='#d30000'),
                fontsize=10, fontweight='bold', color='#d30000', ha='right')

    # 4. เหล็กล่าง (Bottom Bars)
    n_bot = int(res['bot']['n'])
    db_bot = float(res['bot_db'])
    y_bot = stir_off + (db_bot/2) + 2 # ขยับขึ้นมาจากขอบปลอกเล็กน้อย
    x_bot = np.linspace(stir_off + 12, b - stir_off - 12, n_bot) if n_bot > 1 else [b/2]
    
    for x in x_bot:
        ax.add_patch(patches.Circle((x, y_bot), db_bot/2 + 1, color='#008c00', zorder=10))
        
    # เส้นชี้เหล็กล่าง (ชี้ออกด้านขวา)
    ax.annotate(f"{n_bot}-DB{int(db_bot)}", xy=(x_bot[-1], y_bot), xytext=(b*1.35, h * 0.1),
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=0.1", color='#008c00'),
                fontsize=10, fontweight='bold', color='#008c00', ha='left')

    # 5. ข้อความรายละเอียด (จัดให้อยู่ชิดรูปมากขึ้น)
    ax.text(b/2, h + (h*0.08), f"SECTION {int(b)}x{int(h)}", ha='center', fontweight='bold', fontsize=11)
    ax.text(b/2, - (h*0.12), f"RB{int(res['stir_db'])}@{int(res['shear']['s'])}", 
            ha='center', color='#555', fontsize=9, style='italic')
    
    # --- ปรับแต่ง Viewport ให้สมดุล ---
    ax.set_aspect('equal')
    ax.axis('off')
    
    # ตั้งค่าขอบเขต (Margins) ให้พอดีกับเส้นชี้ ไม่เหลือที่ว่างสีขาวมากเกินไป
    ax.set_xlim(-b*0.5, b*1.5) 
    ax.set_ylim(-h*0.2, h*1.2)
    
    f = io.StringIO()
    fig.savefig(f, format="svg", bbox_inches='tight', pad_inches=0.05, transparent=True)
    svg_string = f.getvalue()
    plt.close(fig)
    return svg_string
