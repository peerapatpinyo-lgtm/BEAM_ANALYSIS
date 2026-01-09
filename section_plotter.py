import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 🏗️ Professional Design Constants ---
C_CONC_OUTLINE = '#000000' # เส้นขอบคานสีดำเข้ม
C_REBAR_TOP    = '#d63031' # เหล็กบนสีแดง
C_REBAR_BOT    = '#27ae60' # เหล็กล่างสีเขียว
C_STIRRUP      = '#b2bec3' # เหล็กปลอกสีเทาจาง

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    วาดรูปตัดตามยาวฉบับสมบูรณ์แบบ (Professional Structural Detailing)
    แก้ไขปัญหาคานหนา เหล็กทับเส้น และ Support ไม่แสดงผล
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    
    # --- 1. การจัดการสเกล (Vertical Exaggeration) ---
    # บังคับความสูงในรูปให้เป็น 1000 หน่วยเสมอ เพื่อให้คานผอมยาว ไม่ว่า h_m จะเป็นเท่าไหร่
    v_h = 1000 
    
    # คำนวณความกว้างรูปภาพ (Figsize) ตามสัดส่วนความยาวจริง
    fig_w = max(16, total_L / 500)
    fig, ax = plt.subplots(figsize=(fig_w, 3), dpi=140)
    
    # --- 2. วาดตัวคาน (Beam Body) ---
    # ใช้ zorder=2 เพื่อให้เป็นกรอบหลัก
    ax.add_patch(patches.Rectangle((0, 0), total_L, v_h, lw=2, ec=C_CONC_OUTLINE, fc='white', zorder=2))
    
    # --- 3. วาด Support (ใต้ท้องคาน) ---
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            sx = row['x'] * 1000
            # วาดเสารองรับจากท้องคาน (y=0) ลงไปด้านล่าง (y=-500)
            column = patches.Rectangle((sx-100, -500), 200, 500, fc='#dfe6e9', ec='black', lw=1.2, zorder=1)
            ax.add_patch(column)
            # ใส่ชื่อ Support ใต้เสา
            ax.text(sx, -700, f"S{row.get('id','')}", ha='center', va='top', fontweight='bold', fontsize=10)

    # --- 4. วาดเหล็กเสริม (Reinforcement) ---
    # วางเหล็กที่ระยะ 20% จากขอบบนและล่าง เพื่อให้ไม่ทับเส้นขอบแน่นอน (Perfect Offset)
    y_top_pos = v_h * 0.8
    y_bot_pos = v_h * 0.2
    
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid_span = x_curr + span_L/2
        
        # วาดเหล็กปลอก (Stirrups) แบบเส้นจางหลังเหล็กเมน
        s_spacing = res['shear']['s']
        num_stirrups = int(span_L / s_spacing)
        for stir_x in np.linspace(x_curr + 50, x_curr + span_L - 50, num_stirrups):
            ax.plot([stir_x, stir_x], [100, v_h - 100], color=C_STIRRUP, lw=0.6, alpha=0.5, zorder=3)

        # วาดเหล็กเมนบน (Top Rebar)
        ax.plot([x_curr, x_curr + span_L], [y_top_pos, y_top_pos], 
                color=C_REBAR_TOP, lw=3, zorder=5, solid_capstyle='round')
        
        # วาดเหล็กล่าง (Bottom Rebar)
        ax.plot([x_curr + 40, x_curr + span_L - 40], [y_bot_pos, y_bot_pos], 
                color=C_REBAR_BOT, lw=3, zorder=5, solid_capstyle='round')
        
        # --- รายละเอียดเหล็ก (Annotations) ---
        # วางไว้นอกคานเพื่อความชัดเจนสูงสุด
        ax.text(mid_span, v_h + 150, f"{int(res['neg']['n'])}-DB{int(res['top_db'])}", 
                color=C_REBAR_TOP, ha='center', fontweight='bold', fontsize=9)
        
        ax.text(mid_span, -150, f"RB{int(res['stir_db'])}@{int(s_spacing)}", 
                color='#636e72', ha='center', fontsize=8, style='italic')

        x_curr += span_L

    # --- 5. การตั้งค่ามุมมอง ---
    ax.set_aspect('auto')
    ax.axis('off')
    
    # ปรับขอบเขตให้ครอบคลุม Support และ Label ทั้งหมด
    ax.set_xlim(-600, total_L + 600)
    ax.set_ylim(-900, v_h + 500)
    
    plt.tight_layout()
    return fig

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, title="SECTION A-A"):
    """
    วาดหน้าตัดขวาง (Cross Section) ให้เคลียร์และสวยงาม
    """
    b, h = b_m * 1000, h_m * 1000
    fig, ax = plt.subplots(figsize=(4, 5))
    
    # วาดคอนกรีต
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=2.5, ec='black', fc='none', zorder=1))
    
    # ระยะห่างเหล็ก (Clear Gap)
    gap = cover_mm + 15
    y_t, y_b = h - gap, gap
    
    def draw_bars(n, y, db, color):
        if n <= 0: return
        xs = np.linspace(gap, b - gap, int(n)) if n > 1 else [b/2]
        for x in xs:
            ax.add_patch(patches.Circle((x, y), db/2, fc=color, ec='black', lw=0.8, zorder=5))

    draw_bars(n_top, y_t, db_top_mm, C_REBAR_TOP)
    draw_bars(n_bot, y_b, db_bot_mm, C_REBAR_BOT)
    
    ax.set_title(title, fontweight='bold', pad=20)
    ax.axis('equal')
    ax.axis('off')
    return fig
