import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_section(b, h, cover_mm, db_mm, n_top, n_bot, stirrup_info):
    """
    ฟังก์ชันวาดรูปหน้าตัดคาน (Cross Section)
    """
    # Create Figure
    fig, ax = plt.subplots(figsize=(4, 5))
    
    # 1. วาดคอนกรีต (Concrete Section)
    # b, h รับมาเป็นเมตร
    rect = patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='#f0f0f0')
    ax.add_patch(rect)
    
    # Convert units to meters for plotting
    cv = cover_mm / 1000
    d_bar = db_mm / 1000
    stirrup_d = 0.009 # สมมติเหล็กปลอก 9mm (RB9)
    
    # 2. วาดเหล็กปลอก (Stirrups) - เส้นประสีน้ำเงิน
    w_inner = b - 2*cv
    h_inner = h - 2*cv
    
    if w_inner > 0 and h_inner > 0:
        stirrup = patches.Rectangle((cv, cv), w_inner, h_inner, 
                                    linewidth=1.5, edgecolor='blue', facecolor='none', linestyle='--')
        ax.add_patch(stirrup)
    
    # 3. ฟังก์ชันวาดเหล็ก (Rebars) - วงกลมสีแดง
    def draw_bars(n, is_top):
        if n <= 0: return
        
        # ตำแหน่งแกน Y
        if is_top:
            y_pos = h - cv - stirrup_d - (d_bar/2)
        else:
            y_pos = cv + stirrup_d + (d_bar/2)
            
        # คำนวณตำแหน่งแกน X (Spacing)
        eff_width = b - 2*cv - 2*stirrup_d - d_bar
        
        if n == 1:
            x_positions = [b/2]
        else:
            # ถ้ามีหลายเส้น ให้กระจายเท่าๆ กัน
            if eff_width <= 0: return # พื้นที่ไม่พอ
            gap = eff_width / (n - 1)
            start_x = cv + stirrup_d + d_bar/2
            x_positions = [start_x + i*gap for i in range(int(n))]
            
        for x in x_positions:
            circle = patches.Circle((x, y_pos), radius=d_bar/2, color='#D32F2F', zorder=10)
            ax.add_patch(circle)
            
    # วาดเหล็กบนและล่าง
    draw_bars(n_top, is_top=True)
    draw_bars(n_bot, is_top=False)
    
    # จัดหน้ากระดาษ
    ax.set_xlim(-0.05, b+0.05)
    ax.set_ylim(-0.05, h+0.05)
    ax.set_aspect('equal')
    ax.axis('off') # ปิดแกนเลข
    
    # ใส่ Text บอกรายละเอียด
    plt.text(b/2, h + 0.03, f"Top: {int(n_top)}-DB{int(db_mm)}", ha='center', fontsize=10, color='#D32F2F', fontweight='bold')
    plt.text(b/2, -0.05, f"Bot: {int(n_bot)}-DB{int(db_mm)}", ha='center', fontsize=10, color='#D32F2F', fontweight='bold')
    plt.text(b/2, h/2, f"Stirrup: {stirrup_info.split('(')[0]}", ha='center', fontsize=8, color='blue', rotation=90, alpha=0.7)
    
    plt.tight_layout()
    return fig
