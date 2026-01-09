import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import numpy as np

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    (คงเดิม 100% ตามที่คุณส่งมา) วาดรูปตัดยาวคาน
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

# --- 💎 ฟังก์ชันที่เพิ่มใหม่: วาดรูปตัดขวาง (Cross Section) ---
def plot_cross_section(res):
    """
    วาดรูปตัดขวางคาน (Cross Section) สัมพันธ์กับค่าที่กรอกในแต่ละ Span
    """
    b = float(res['b'])
    h = float(res['h'])
    cover = float(res['cover'])
    
    fig, ax = plt.subplots(figsize=(4, 5))
    
    # 1. วาดหน้าตัดคอนกรีต
    ax.add_patch(patches.Rectangle((0, 0), b, h, facecolor='white', edgecolor='black', lw=3, zorder=1))
    
    # 2. วาดเหล็กปลอก (Stirrup)
    stir_offset = cover + 5 
    ax.add_patch(patches.Rectangle((stir_offset, stir_offset), b-2*stir_offset, h-2*stir_offset, 
                                   fill=False, edgecolor='#7f8c8d', lw=2, zorder=2))
    
    # 3. วาดเหล็กเมนบน (Top Bars) - สีแดง
    n_top = int(res['top']['n'])
    db_top = float(res['top_db'])
    if n_top > 1:
        x_top = np.linspace(stir_offset + 10, b - stir_offset - 10, n_top)
    else:
        x_top = [b/2]
        
    for x in x_top:
        ax.add_patch(patches.Circle((x, h - stir_offset - 10), db_top/2 + 2, color='#d30000', zorder=10))
    
    # 4. วาดเหล็กเมนล่าง (Bottom Bars) - สีเขียว
    n_bot = int(res['bot']['n'])
    db_bot = float(res['bot_db'])
    if n_bot > 1:
        x_bot = np.linspace(stir_offset + 10, b - stir_offset - 10, n_bot)
    else:
        x_bot = [b/2]
        
    for x in x_bot:
        ax.add_patch(patches.Circle((x, stir_offset + 10), db_bot/2 + 2, color='#008c00', zorder=10))

    # ใส่ข้อความกำกับ
    ax.text(b/2, h + h*0.1, f"SECTION {int(b)}x{int(h)} mm", ha='center', fontweight='bold', fontsize=12)
    ax.text(b/2, -h*0.15, f"Stirrup: RB{int(res['stir_db'])}@{int(res['shear']['s'])}", ha='center', color='#7f8c8d', fontsize=10)
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-b*0.2, b*1.2)
    ax.set_ylim(-h*0.2, h*1.2)
    
    f = io.StringIO()
    fig.savefig(f, format="svg", bbox_inches='tight')
    svg_string = f.getvalue()
    plt.close(fig)
    return svg_string
