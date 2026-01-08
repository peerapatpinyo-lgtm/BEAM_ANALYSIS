def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    Cross Section - ปรับมาตราส่วน เส้น และตัวหนังสือให้เท่ากับ Long Section เป๊ะ
    """
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds, db = 6, db_main_mm
    
    # ใช้ SCALE_FACTOR เดียวกันเพื่อให้ขนาดวัตถุในจอเท่ากัน
    # เพิ่มพื้นที่รอบข้าง (Padding) ให้เท่ากับ Long Section
    width_inches = (b + 1000) / SCALE_FACTOR 
    height_inches = (h + 1200) / SCALE_FACTOR
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=100)
    
    # 1. Concrete Outline (ความหนาเส้น 2 เท่ากับ Long Section)
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='#1a1a1a', facecolor='#ffffff'))
    
    # 2. Stirrup (เส้นบางกว่าเล็กน้อย)
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=1.5, edgecolor='#2c3e50', facecolor='none'))
    
    # 3. Bars (ความหนาและขนาดต้องดูเท่ากัน)
    def draw_bars(n, y_pos, color):
        if n < 1: return
        spacing = (b - 2*cover - 2*ds - db) / (max(1, n - 1))
        for i in range(int(n)):
            x = (b/2) if n == 1 else (cover + ds + db/2 + i*spacing)
            # ขนาด DB ปรับตามสเกลจริง
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=10))

    y_bot, y_top = cover + ds + db/2, h - cover - ds - db/2
    draw_bars(n_bottom, y_bot, '#8e1c19')
    draw_bars(n_top, y_top, '#1a5276')
    
    # 4. Labels (Font size 12 เท่ากับ Long Section และย้ายมาไว้ข้างๆ)
    # ใช้การวาง text แบบไม่มีลูกศรตามที่คุณชอบ
    ax.text(b + 80, y_top, f"{int(n_top)}-DB{int(db)}", va='center', color='#1a5276', fontweight='bold', fontsize=12)
    ax.text(b + 80, y_bot, f"{int(n_bottom)}-DB{int(db)}", va='center', color='#8e1c19', fontweight='bold', fontsize=12)
    ax.text(b/2, h + 150, f"{stirrup_name}", ha='center', color='#1d8348', fontsize=12, fontweight='bold')

    # 5. Engineering Ticks Dimension (สไตล์เดียวกันเป๊ะ)
    def draw_tick_dim(p1, p2, text, vert=False):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', lw=1.2) # เส้นหลัก
        tick = 30 # ขนาดขีดเฉียง
        for p in [p1, p2]:
            ax.plot([p[0]-tick, p[0]+tick], [p[1]-tick, p[1]+tick], color='black', lw=2)
        if vert:
            ax.text(p1[0]-80, (p1[1]+p2[1])/2, text, va='center', ha='right', rotation=90, fontsize=12)
        else:
            ax.text((p1[0]+p2[0])/2, p1[1]-80, text, ha='center', va='top', fontsize=12)

    # บอกขนาด B และ H
    draw_tick_dim([0, -150], [b, -150], f"{int(b)}")
    draw_tick_dim([-150, 0], [-150, h], f"{int(h)}", vert=True)
    
    # ปรับแนวระนาบให้สมดุล (Center alignment)
    ax.set_xlim(-500, b + 500)
    ax.set_ylim(-600, h + 600)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig
