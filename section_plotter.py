import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 📐 มาตรฐานสีและฟอนต์ (Engineering Style) ---
C_BEAM = '#000000'   # เส้นขอบคาน (ดำ)
C_TOP  = '#d63031'   # เหล็กบน (แดง)
C_BOT  = '#27ae60'   # เหล็กล่าง (เขียว)
C_STIR = '#636e72'   # เหล็กปลอก (เทา)

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, title="CROSS SECTION"):
    """ วาดหน้าตัดขวาง ให้เหล็กอยู่ข้างในและสมส่วน """
    b, h = b_m * 1000, h_m * 1000
    fig, ax = plt.subplots(figsize=(4, 5))
    
    # 1. วาดคอนกรีต (Outline)
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=2, ec=C_BEAM, fc='none', zorder=1))
    
    # 2. ระยะห่างเหล็ก (Cover + 12mm เพื่อไม่ให้ทับเส้นขอบ)
    gap = cover_mm + 12
    y_top = h - gap
    y_bot = gap
    
    def draw_bars(n, y_pos, db, color):
        if n <= 0: return
        # กระจายเหล็กตามหน้ากว้าง b
        xs = np.linspace(gap, b - gap, int(n)) if n > 1 else [b/2]
        for x in xs:
            ax.add_patch(patches.Circle((x, y_pos), db/2, fc=color, ec='black', lw=0.5, zorder=5))

    draw_bars(n_top, y_top, db_top_mm, C_TOP)
    draw_bars(n_bot, y_bot, db_bot_mm, C_BOT)
    
    ax.set_title(title, fontweight='bold', pad=15)
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-50, b + 50)
    ax.set_ylim(-50, h + 50)
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """ วาดรูปตัดตามยาว ให้คานผอมยาวและเหล็กไม่ทับขอบ """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    
    # --- 🛠️ แก้ปัญหาคานหนา: กำหนด figsize ให้กว้างมากแต่เตี้ย ---
    # เช่น คานยาว 5 เมตร รูปจะกว้าง 12 นิ้ว สูง 2.5 นิ้ว
    fig, ax = plt.subplots(figsize=(14, 2.5), dpi=120)
    
    # 1. วาดขอบคาน (Beam Body)
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=1.5, ec=C_BEAM, fc='none', zorder=2))
    
    # 2. วาด Support (ใต้ท้องคานเท่านั้น)
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            sx = row['x'] * 1000
            # วาดเสาจาก y=0 ลงไปถึง y=-300 (ค่าลบคือใต้คาน)
            ax.add_patch(patches.Rectangle((sx-75, -300), 150, 300, fc='#dfe6e9', ec='black', lw=1, zorder=1))
            ax.text(sx, -450, f"S{row.get('id','')}", ha='center', fontweight='bold', fontsize=9)

    # 3. วาดเหล็กเสริม (Offset เข้ามาข้างใน 15mm เพื่อไม่ให้ทับเส้นขอบ)
    y_top = h_mm - (cover_mm + 15)
    y_bot = cover_mm + 15
    
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # เหล็กบน (Main Top)
        ax.plot([x_curr, x_curr + span_L], [y_top, y_top], color=C_TOP, lw=2, zorder=5)
        # เหล็กล่าง (Main Bot)
        ax.plot([x_curr + 50, x_curr + span_L - 50], [y_bot, y_bot], color=C_BOT, lw=2, zorder=5)
        
        # Labels (อยู่นอกตัวคานเพื่อความชัดเจน)
        ax.text(mid, h_mm + 80, f"{res['neg']['n']}-DB{int(res['top_db'])}", color=C_TOP, ha='center', fontsize=8)
        ax.text(mid, -150, f"RB{int(res['stir_db'])}@{int(res['shear']['s'])}", color=C_STIR, ha='center', fontsize=8)
        
        x_curr += span_L

    # --- 🛠️ แก้ปัญหาคานหนา: ใช้ aspect='auto' เพื่อให้สเกล x และ y แยกกันได้ ---
    ax.set_aspect('auto') 
    ax.axis('off')
    
    # ขยายขอบเขตการมองเห็น
    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-600, h_mm + 400)
    
    plt.tight_layout()
    return fig
