import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

def plot_as_svg(spans, sup_df, design_res, h_m, cover_mm):
    """
    สร้างรูปตัดตามยาวในรูปแบบ SVG String เพื่อความคมชัดระดับเลเซอร์
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 350  # คานเพรียวบางระดับพรีเมียม
    
    # 1. ปรับขนาด Figure ให้ยาวตามความจริง
    fig_w = max(16, total_L / 400)
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    
    # --- 🛠️ Drafting Styles ---
    # วาดคานคอนกรีต (ปิด Antialiasing เพื่อให้ขอบคมกริบแบบ CAD)
    beam = patches.Rectangle((0, 0), total_L, v_h, lw=1.5, ec='black', fc='white', aa=False, zorder=2)
    ax.add_patch(beam)
    
    # 2. Grid & Dimension Lines
    curr_x = 0
    for i, s_mm in enumerate(spans_mm + [0]):
        # Grid Line (เส้นแกนเสา)
        ax.plot([curr_x, curr_x], [-500, v_h + 350], color='#bdc3c7', ls='--', lw=0.8, zorder=1)
        # หัว Grid
        ax.annotate(chr(65+i), xy=(curr_x, v_h + 450), ha='center', va='center',
                    bbox=dict(boxstyle='circle', fc='white', ec='black', lw=1.2), fontsize=13)
        
        # เส้นบอกระยะ Span
        if i < len(spans_mm):
            ax.annotate('', xy=(curr_x, v_h + 200), xytext=(curr_x + s_mm, v_h + 200),
                        arrowprops=dict(arrowstyle='<->', color='#2980b9', lw=1))
            ax.text(curr_x + s_mm/2, v_h + 230, f"{s_mm/1000:.2f} m", 
                    ha='center', color='#2980b9', fontsize=11, fontweight='bold')
            curr_x += s_mm

    # 3. Supports (แสดงสัญลักษณ์พร้อมชื่อ Type)
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            sx = row['x'] * 1000
            stype = str(row.get('type', 'PIN')).upper()
            
            if stype == 'FIXED':
                ax.add_patch(patches.Rectangle((sx-80, -300), 160, 300, fc='#dfe6e9', ec='black', lw=1.2, hatch='///'))
            elif stype == 'ROLLER':
                pts = [[sx, 0], [sx-70, -150], [sx+70, -150]]
                ax.add_patch(patches.Polygon(pts, fc='white', ec='black', lw=1.2))
                ax.add_patch(patches.Circle((sx, -175), 20, fc='black'))
            else: # PIN
                pts = [[sx, 0], [sx-70, -150], [sx+70, -150]]
                ax.add_patch(patches.Polygon(pts, fc='#2c3e50', ec='black', lw=1.2))
            
            ax.text(sx, -400, f"S{row['id']}: {stype}", ha='center', fontweight='bold', fontsize=9)

    # 4. Reinforcement (เหล็กเส้นระดับ High-Definition)
    y_t, y_b = v_h * 0.8, v_h * 0.2
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # วาดเหล็กเมน (ใช้ Solid Cap เพื่อความคม)
        ax.plot([x_curr, x_curr + span_L], [y_t, y_t], color='#d63031', lw=3, zorder=10, aa=False)
        ax.plot([x_curr + 40, x_curr + span_L - 40], [y_b, y_b], color='#27ae60', lw=3, zorder=10, aa=False)
        
        # Label เหล็ก (TOP/BOT)
        ax.annotate(f"{int(res['neg']['n'])}-DB{int(res['top_db'])} (TOP)", xy=(mid, v_h + 50), 
                    ha='center', va='bottom', fontsize=11, color='#c0392b', fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        
        ax.annotate(f"{int(res['pos']['n'])}-DB{int(res['bot_db'])} (BOT)", xy=(mid, y_b - 30), 
                    ha='center', va='top', fontsize=10, color='#1e8449', fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

        x_curr += span_L

    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-800, total_L + 800)
    ax.set_ylim(-700, v_h + 600)
    
    # --- 💎 ขั้นตอนสำคัญ: แปลงเป็น SVG String ---
    f = io.StringIO()
    fig.savefig(f, format="svg", bbox_inches='tight')
    plt.close(fig)
    return f.getvalue()
