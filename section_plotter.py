import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    สร้างแบบขยายคานในรูปแบบ SVG (Vector) 
    คมชัดสูงสุด ซูมไม่แตก และสัดส่วนถูกต้องตามหลักวิศวกรรม
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 350  # สัดส่วนคานเพรียวบางระดับสากล
    
    # 1. กำหนดสัดส่วนรูปภาพ (Dynamic Figsize)
    fig_w = max(16, total_L / 350)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    
    # 2. วาดคอนกรีต (ปิด Antialiasing เพื่อขอบคมกริบแบบ CAD)
    beam = patches.Rectangle((0, 0), total_L, v_h, lw=2, ec='black', fc='white', antialiased=False, zorder=2)
    ax.add_patch(beam)
    
    # 3. วาด Grid Lines และหัวเสา (A, B, C)
    curr_x = 0
    for i, s_mm in enumerate(spans_mm + [0]):
        # เส้น Grid (Center Line)
        ax.plot([curr_x, curr_x], [-600, v_h + 400], color='#7f8c8d', ls='-.', lw=1, zorder=1)
        # หัว Grid วงกลม
        ax.annotate(chr(65+i), xy=(curr_x, v_h + 500), ha='center', va='center',
                    bbox=dict(boxstyle='circle', fc='white', ec='black', lw=1.5), 
                    fontsize=14, fontweight='black')
        
        # Dimension Line (ระยะ Span เมตร)
        if i < len(spans_mm):
            ax.annotate('', xy=(curr_x, v_h + 250), xytext=(curr_x + s_mm, v_h + 250),
                        arrowprops=dict(arrowstyle='<->', color='#2980b9', lw=1.2))
            ax.text(curr_x + s_mm/2, v_h + 300, f"{s_mm/1000:.2f} m", 
                    ha='center', color='#2980b9', fontsize=12, fontweight='black')
            curr_x += s_mm

    # 4. วาด Supports ( Engineering Symbols )
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            sx = row['x'] * 1000
            stype = str(row.get('type', 'PIN')).upper()
            
            if stype == 'FIXED':
                ax.add_patch(patches.Rectangle((sx-100, -350), 200, 350, fc='#dfe6e9', ec='black', lw=1.5, hatch='////'))
            elif stype == 'ROLLER':
                pts = [[sx, 0], [sx-90, -180], [sx+90, -180]]
                ax.add_patch(patches.Polygon(pts, fc='white', ec='black', lw=1.5))
                ax.add_patch(patches.Circle((sx, -215), 30, fc='black'))
            else: # PIN
                pts = [[sx, 0], [sx-90, -180], [sx+90, -180]]
                ax.add_patch(patches.Polygon(pts, fc='#2c3e50', ec='black', lw=1.5))
            
            ax.text(sx, -500, f"S{row['id']}: {stype}", ha='center', fontweight='black', fontsize=10)

    # 5. วาดเหล็กเสริม Main Rebars (Red=Top, Green=Bot)
    y_t, y_b = v_h * 0.82, v_h * 0.18
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # เหล็กเมน (ปิด Antialiasing เพื่อให้เส้นคมกริบ)
        ax.plot([x_curr, x_curr + span_L], [y_t, y_t], color='#d30000', lw=3.5, zorder=10, antialiased=False)
        ax.plot([x_curr + 40, x_curr + span_L - 40], [y_b, y_b], color='#008c00', lw=3.5, zorder=10, antialiased=False)
        
        # Text Labels พร้อมกล่องสีขาวกันตัวหนังสือเบลอ
        label_opt = dict(ha='center', fontweight='black', fontsize=11, 
                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.5))
        
        ax.text(mid, v_h + 80, f"{int(res['neg']['n'])}-DB{int(res['top_db'])} (TOP)", color='#d30000', **label_opt)
        ax.text(mid, y_b - 50, f"{int(res['pos']['n'])}-DB{int(res['bot_db'])} (BOT)", color='#008c00', va='top', **label_opt)
        
        # สัญลักษณ์เหล็กปลอก
        ax.text(mid, -150, f"RB{int(res['stir_db'])}@{int(res['shear']['s'])}mm", 
                color='#535c68', fontsize=9, style='italic', ha='center')

        x_curr += span_L

    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-1000, total_L + 1000)
    ax.set_ylim(-800, v_h + 800)
    
    # แปลงผลลัพธ์เป็น SVG String เพื่อส่งให้ Browser แสดงผลสดๆ
    f = io.StringIO()
    fig.savefig(f, format="svg", bbox_inches='tight')
    plt.close(fig)
    return f.getvalue()
