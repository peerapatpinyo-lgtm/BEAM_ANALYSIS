import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 🏗️ Engineering Standard Config ---
COLOR_CONCRETE = '#FFFFFF'
COLOR_DIM      = '#000000'
COLOR_STIRRUP  = '#bdc3c7' # ใช้สีจางลงเพื่อให้เหล็กเมนเด่น
COLOR_TOP      = '#c0392b'
COLOR_BOT      = '#27ae60'
FONT_MAIN      = 10
FONT_DIM       = 8

def _setup_figure(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    return fig, ax

def _draw_dim_line(ax, p1, p2, text, offset=0, is_vert=False):
    """วาดเส้นบอกขนาดมาตรฐานวิศวกรรม"""
    if is_vert:
        x_pos = p1[0] - offset
        ax.annotate("", xy=(x_pos, p1[1]), xytext=(x_pos, p2[1]),
                    arrowprops=dict(arrowstyle='<|-|>', color=COLOR_DIM, lw=0.8))
        ax.text(x_pos - 10, (p1[1]+p2[1])/2, text, ha='right', va='center', rotation=90, fontsize=FONT_DIM)
    else:
        y_pos = p1[1] + offset
        ax.annotate("", xy=(p1[0], y_pos), xytext=(p2[0], y_pos),
                    arrowprops=dict(arrowstyle='<|-|>', color=COLOR_DIM, lw=0.8))
        ax.text((p1[0]+p2[0])/2, y_pos + 10, text, ha='center', va='bottom', fontsize=FONT_DIM)

def _draw_support_symbol(ax, x, y, sup_type, sup_id):
    """วาด Support ใต้ท้องคาน (อิงตามสัญลักษณ์ใน Textbook)"""
    size = 200
    # วาดตัวเสาหรือฐานรองรับ
    ax.add_patch(patches.Rectangle((x-75, y-350), 150, 350, fc='#ecf0f1', ec='black', lw=1, zorder=2))
    ax.text(x, y - 500, f"S{sup_id}", ha='center', fontweight='bold', fontsize=9)

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, stir_text, fc, fy, title="SECTION A-A"):
    """รูปตัดขวาง (Cross Section) - สัดส่วนสมจริง"""
    b, h = b_m * 1000.0, h_m * 1000.0
    fig, ax = _setup_figure((5, 6))
    
    # Concrete & Stirrup
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=2, ec='black', fc='#fdfdfd', zorder=1))
    st_off = cover_mm
    ax.add_patch(patches.Rectangle((st_off, st_off), b-2*st_off, h-2*st_off, lw=1.2, ec='#7f8c8d', fill=False, zorder=2))
    
    def draw_bars(n, y, db, color):
        if n <= 0: return b/2
        xs = np.linspace(st_off+15, b-st_off-15, int(n)) if n > 1 else [b/2]
        for x in xs:
            ax.add_patch(patches.Circle((x, y), db/2, fc=color, ec='black', lw=0.5, zorder=5))
        return xs[-1]

    y_t = h - st_off - 10 - db_top_mm/2
    y_b = st_off + 10 + db_bot_mm/2
    draw_bars(n_top, y_t, db_top_mm, COLOR_TOP)
    draw_bars(n_bot, y_b, db_bot_mm, COLOR_BOT)
    
    # Text Labels
    ax.text(b/2, h+50, title, ha='center', fontweight='bold', fontsize=11)
    ax.text(b+40, y_t, f"{int(n_top)}-DB{int(db_top_mm)}", color=COLOR_TOP, va='center', fontweight='bold')
    ax.text(b+40, y_b, f"{int(n_bot)}-DB{int(db_bot_mm)}", color=COLOR_BOT, va='center', fontweight='bold')

    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-100, b + 250)
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """รูปตัดตามยาว (Longitudinal Section) - แก้ไขสเกลให้คานดูบางและยาว"""
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    
    # ปรับสัดส่วน Figure ให้ยาวมากเมื่อเทียบกับความสูง (เช่น 15:3)
    fig, ax = _setup_figure((15, 3))
    
    # 1. วาดโครงสร้างคาน
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=1.5, ec='black', fc='none', zorder=10))
    
    # 2. วาด Support
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            _draw_support_symbol(ax, row['x']*1000, 0, row.get('type','Pin'), row.get('id',''))

    # 3. วาดเหล็กเสริม
    x_curr = 0
    y_t = h_mm - cover_mm - 15
    y_b = cover_mm + 15

    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # วาดเหล็กปลอกเป็นเส้นแนวตั้งจางๆ
        s = res['shear']['s']
        num_s = int(span_L / s)
        for sx in np.linspace(x_curr + 50, x_curr + span_L - 50, num_s):
            ax.plot([sx, sx], [cover_mm, h_mm-cover_mm], color=COLOR_STIRRUP, lw=0.4, alpha=0.5)

        # เหล็กบน (Top Rebar)
        ax.plot([x_curr, x_curr + span_L], [y_t, y_t], color=COLOR_TOP, lw=2.5, zorder=11)
        # เหล็กล่าง (Bottom Rebar)
        ax.plot([x_curr + 50, x_curr + span_L - 50], [y_b, y_b], color=COLOR_BOT, lw=2.5, zorder=11)

        # Labels แยกตำแหน่งไม่ให้ซ้อน
        ax.text(mid, h_mm + 100, f"{res['neg']['n']}-DB{int(res['top_db'])}", color=COLOR_TOP, ha='center', fontsize=8)
        ax.text(mid, y_b + 40, f"{res['pos']['n']}-DB{int(res['bot_db'])}", color=COLOR_BOT, ha='center', fontsize=8)
        ax.text(mid, -150, f"RB{int(res['stir_db'])}@{int(s)}", color='#7f8c8d', ha='center', fontsize=8)

        x_curr += span_L

    # Dimensions
    _draw_dim_line(ax, (0, h_mm), (total_L, h_mm), f"L = {total_L/1000:.2f} m", offset=300)

    ax.set_aspect('auto') # บังคับให้สัดส่วนยืดตามความยาวจริง ไม่โดนบีบเป็นสี่เหลี่ยมจัตุรัส
    ax.axis('off')
    ax.set_xlim(-400, total_L + 400)
    ax.set_ylim(-600, h_mm + 500)
    return fig
