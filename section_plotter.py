import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 📐 Professional Configuration ---
C_CONC = '#000000'  # Concrete Outline
C_REBAR_T = '#C0392B' # Top Rebar (Standard Red)
C_REBAR_B = '#27AE60' # Bottom Rebar (Standard Green)
C_STIRRUP = '#BDC3C7' # Stirrups (Light Gray)

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    วาดรูปตัดตามยาวแบบมืออาชีพ (Shop Drawing Style)
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    
    # กำหนดสัดส่วนรูปภาพให้ "ผอมยาว" พิเศษ (กว้าง 18 นิ้ว สูง 3 นิ้ว)
    fig, ax = plt.subplots(figsize=(18, 3), dpi=130)
    
    # 1. วาด Concrete Outline (zorder=2)
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=1.5, ec=C_CONC, fc='none', zorder=2))
    
    # 2. คำนวณระยะเหล็กเสริม (Clearance Offset)
    # บังคับให้เหล็กห่างจากขอบบน/ล่าง = cover + 20mm เพื่อความสวยงามและชัดเจน
    y_top = h_mm - (cover_mm + 20)
    y_bot = cover_mm + 20
    
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # --- วาดเหล็กปลอก (Stirrups) แบบจางๆ ---
        s_val = res['shear']['s']
        num_s = int(span_L / s_val)
        for sx in np.linspace(x_curr + 50, x_curr + span_L - 50, num_s):
            ax.plot([sx, sx], [cover_mm, h_mm-cover_mm], color=C_STIRRUP, lw=0.5, alpha=0.4, zorder=1)

        # --- วาดเหล็กเมน (Main Rebars) ---
        # เหล็กบน (Top) - ยาวตลอดช่วง
        ax.plot([x_curr, x_curr + span_L], [y_top, y_top], color=C_REBAR_T, lw=2.5, zorder=5, solid_capstyle='round')
        # เหล็กล่าง (Bottom) - เว้นระยะจาก Support เล็กน้อยตามมาตรฐาน
        ax.plot([x_curr + 50, x_curr + span_L - 50], [y_bot, y_bot], color=C_REBAR_B, lw=2.5, zorder=5, solid_capstyle='round')
        
        # --- Annotations (Labels) ---
        # วาดข้อความเหนือและใต้คาน ไม่ให้ทับเนื้อคอนกรีต
        ax.text(mid, h_mm + 120, f"{int(res['neg']['n'])}-DB{int(res['top_db'])}", 
                color=C_REBAR_T, ha='center', fontweight='bold', fontsize=9)
        ax.text(mid, y_bot + 40, f"{int(res['pos']['n'])}-DB{int(res['bot_db'])}", 
                color=C_REBAR_B, ha='center', fontsize=8)
        ax.text(mid, -180, f"RB{int(res['stir_db'])}@{int(s_val)}", 
                color='#546E7A', ha='center', fontsize=8, style='italic')

        x_curr += span_L

    # 3. วาด Support (เสารองรับใต้คาน)
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            sx = row['x'] * 1000
            # วาดหน้าตัดเสาเริ่มจาก y=0 ลงไปด้านล่าง
            ax.add_patch(patches.Rectangle((sx-100, -400), 200, 400, fc='#F8F9FA', ec='black', lw=1, zorder=0))
            ax.text(sx, -550, f"S{row.get('id','')}", ha='center', fontweight='bold', fontsize=9)

    # 4. Final Adjustment
    ax.set_aspect('auto') # บังคับสเกลให้ยืดแนวราบ
    ax.axis('off')
    
    # ปรับขีดจำกัดมุมมอง (X, Y)
    ax.set_xlim(-600, total_L + 600)
    ax.set_ylim(-700, h_mm + 600)
    
    plt.tight_layout()
    return fig

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, title="SECTION A-A"):
    """
    วาดหน้าตัดขวาง (Cross Section) ให้สมส่วนและถูกต้องตามหลักวิศวกรรม
    """
    b, h = b_m * 1000, h_m * 1000
    fig, ax = plt.subplots(figsize=(4, 5))
    
    # 1. Concrete Frame
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=2.5, ec=C_CONC, fc='none', zorder=1))
    
    # 2. Reinforcement Offset (เหล็กต้องอยู่ในระยะหุ้ม)
    gap = cover_mm + 15
    y_t = h - gap
    y_b = gap
    
    def draw_rebar_group(n, y, db, color):
        if n <= 0: return
        xs = np.linspace(gap, b - gap, int(n)) if n > 1 else [b/2]
        for x in xs:
            ax.add_patch(patches.Circle((x, y), db/2, fc=color, ec='black', lw=0.7, zorder=5))

    draw_rebar_group(n_top, y_t, db_top_mm, C_REBAR_T)
    draw_rebar_group(n_bot, y_b, db_bot_mm, C_REBAR_B)
    
    ax.set_title(title, fontweight='bold', pad=15)
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-100, b + 100)
    ax.set_ylim(-100, h + 100)
    return fig
