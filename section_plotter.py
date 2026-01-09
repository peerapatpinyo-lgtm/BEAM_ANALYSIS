import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    แก้ไขปัญหาคานหนาและเส้นทับกันด้วยระบบ Relative Offset
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    
    # 1. ตั้งค่ารูปภาพให้กว้างมาก (16 นิ้ว) และเตี้ย (3 นิ้ว) เพื่อบังคับให้คานผอมยาว
    fig, ax = plt.subplots(figsize=(16, 3), dpi=120)
    
    # 2. วาดคอนกรีต (Outline) - ใช้ zorder เพื่อให้อยู่ด้านหลังเหล็ก
    # วาดรูปสี่เหลี่ยมผืนผ้าที่เป็นตัวคาน
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=2, ec='black', fc='none', zorder=2))
    
    # 3. แก้ปัญหาเหล็กทับเส้น: ใช้พิกัด Y ที่ 'Offset' จากขอบเข้ามา 20% ของความสูงคาน
    # วิธีนี้จะทำให้เหล็กไม่มีทางทับเส้นขอบคานแน่นอน ไม่ว่าคานจะสูงหรือต่ำ
    inner_gap = h_mm * 0.2
    y_top = h_mm - inner_gap
    y_bot = inner_gap
    
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # วาดเหล็กบน (สีแดง) และเหล็กล่าง (สีเขียว)
        # ใช้ lw (line width) ที่หนาเพื่อให้เห็นชัด แต่ไม่ออกไปนอกเส้นขอบ
        ax.plot([x_curr + 20, x_curr + span_L - 20], [y_top, y_top], 
                color='#d63031', lw=3, label='Top Bar', zorder=5, solid_capstyle='round')
        ax.plot([x_curr + 50, x_curr + span_L - 50], [y_bot, y_bot], 
                color='#27ae60', lw=3, label='Bottom Bar', zorder=5, solid_capstyle='round')
        
        # ใส่ตัวเลขรายละเอียด (วางห่างจากตัวคาน)
        ax.text(mid, h_mm + 150, f"{res['neg']['n']}-DB{int(res['top_db'])}", 
                color='#d63031', ha='center', fontweight='bold', fontsize=9)
        ax.text(mid, -250, f"RB{int(res['stir_db'])}@{int(res['shear']['s'])}", 
                color='#636e72', ha='center', fontsize=8)
        
        x_curr += span_L

    # 4. แก้ปัญหา Support ทับคาน: วาด Support 'ใต้' เส้น y=0 ลงไปเท่านั้น
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            sx = row['x'] * 1000
            # วาดฐานรองรับ (เสา) จาก 0 ลงไปถึง -400
            ax.add_patch(patches.Rectangle((sx-80, -400), 160, 400, 
                                         fc='#ecf0f1', ec='black', lw=1, zorder=1))
            ax.text(sx, -550, f"S{row.get('id','')}", ha='center', fontweight='bold')

    # 5. ตั้งค่าการแสดงผลแบบสมส่วน
    ax.set_aspect('auto') # สำคัญ: ยืดตามแนวนอน
    ax.axis('off')
    
    # ปรับขอบเขตแกน Y ให้กว้างพอที่จะเห็น Support และ Label แต่ไม่ทำให้คานดูหนา
    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-700, h_mm + 400)
    
    plt.tight_layout()
    return fig
