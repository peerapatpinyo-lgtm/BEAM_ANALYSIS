import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 🏗️ Engineering Config ---
COLOR_CONCRETE = '#FFFFFF'
COLOR_DIM      = '#000000'
COLOR_STIRRUP  = '#576574'
COLOR_TOP      = '#c0392b'
COLOR_BOT      = '#27ae60'
FONT_MAIN      = 10
FONT_DIM       = 9

def _setup_figure(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    return fig, ax

def _draw_dim_line(ax, p1, p2, text, offset=0, is_vert=False):
    """วาดเส้นบอกขนาด (Dimension Line) ให้ดูสะอาดตา"""
    if is_vert:
        x_pos = p1[0] - offset
        ax.annotate("", xy=(x_pos, p1[1]), xytext=(x_pos, p2[1]),
                    arrowprops=dict(arrowstyle='<|-|>', color=COLOR_DIM, lw=0.8))
        ax.text(x_pos - 15, (p1[1]+p2[1])/2, text, ha='right', va='center', rotation=90, fontsize=FONT_DIM)
    else:
        y_pos = p1[1] + offset
        ax.annotate("", xy=(p1[0], y_pos), xytext=(p2[0], y_pos),
                    arrowprops=dict(arrowstyle='<|-|>', color=COLOR_DIM, lw=0.8))
        ax.text((p1[0]+p2[0])/2, y_pos + 15, text, ha='center', va='bottom', fontsize=FONT_DIM)

def _draw_support_symbol(ax, x, y, sup_type, sup_id):
    """วาด Support ใต้ท้องคาน (ไม่ทับเหล็ก)"""
    size = 200
    if sup_type == 'Fixed':
        ax.add_patch(patches.Rectangle((x-60, y-400), 120, 400, fc='#bdc3c7', ec='black', hatch='///', zorder=2))
    else: # Pin/Roller style
        tri = patches.Polygon([[x, y], [x-100, y-size], [x+100, y-size]], closed=True, fc='#bdc3c7', ec='black', zorder=2)
        ax.add_patch(tri)
        if sup_type == 'Roller':
            ax.plot([x-120, x+120], [y-size-40, y-size-40], color='black', lw=2)
        else:
            ax.plot([x-120, x+120], [y-size, y-size], color='black', lw=2)
    ax.text(x, y - size - 150, f"S{sup_id}", ha='center', fontsize=9, fontweight='bold')

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, stir_text, fc, fy, title="SECTION A-A"):
    """รูปตัดขวาง (Cross Section) - สเกลสมส่วนและจัดเหล็กสวยงาม"""
    b, h = b_m * 1000.0, h_m * 1000.0
    fig, ax = _setup_figure((5, 6))
    
    # Concrete & Stirrup
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=2, ec='black', fc='#FDFDFD', zorder=1))
    st_off = cover_mm
    ax.add_patch(patches.Rectangle((st_off, st_off), b-2*st_off, h-2*st_off, lw=1.2, ec=COLOR_STIRRUP, fill=False, zorder=2))
    
    # วางเหล็กเสริม (กระจายตามหน้ากว้าง b)
    def draw_bars(n, y, db, color):
        if n <= 0: return b/2
        xs = np.linspace(st_off+15+db/2, b-st_off-15-db/2, int(n)) if n > 1 else [b/2]
        for x in xs:
            ax.add_patch(patches.Circle((x, y), db/2, fc=color, ec='black', lw=0.5, zorder=5))
        return xs[-1]

    y_top = h - st_off - 10 - db_top_mm/2
    y_bot = st_off + 10 + db_bot_mm/2
    
    lx_t = draw_bars(n_top, y_top, db_top_mm, COLOR_TOP)
    lx_b = draw_bars(n_bot, y_bot, db_bot_mm, COLOR_BOT)
    
    # Labels
    ax.text(b+50, h-30, f"{int(n_top)}-DB{int(db_top_mm)} (Top)", color=COLOR_TOP, fontweight='bold', va='top')
    ax.text(b+50, 30, f"{int(n_bot)}-DB{int(db_bot_mm)} (Bot)", color=COLOR_BOT, fontweight='bold', va='bottom')
    ax.text(b/2, h+100, title, ha='center', fontweight='bold', fontsize=12)

    _draw_dim_line(ax, (0, 0), (b, 0), f"{int(b)}", offset=-80)
    _draw_dim_line(ax, (0, 0), (0, h), f"{int(h)}", offset=80, is_vert=True)

    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-150, b + 350)
    ax.set_ylim(-150, h + 200)
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """รูปตัดตามยาว (Longitudinal) - สเกลสมส่วนแนวนอน ไม่เป็นกำแพง"""
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    
    # ปรับ figsize ให้ยาวตามสัดส่วนคานจริง (กว้าง 14 นิ้ว)
    fig, ax = _setup_figure((14, 4))
    
    # 1. วาดคอนกรีตคาน
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=2, ec='black', fc='white', zorder=1))
    
    # 2. วาด Support
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            _draw_support_symbol(ax, row['x']*1000, 0, row.get('type','Pin'), row.get('id',''))

    # 3. วาดเหล็กเสริม
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # เหล็กบน (Main Top)
        ax.plot([x_curr, x_curr + span_L], [h_mm-cover_mm-15, h_mm-cover_mm-15], color=COLOR_TOP, lw=2.5, zorder=10)
        # เหล็กล่าง (Main Bot)
        ax.plot([x_curr+50, x_curr+span_L-50], [cover_mm+15, cover_mm+15], color=COLOR_BOT, lw=2.5, zorder=10)

        # Labels (จัดตำแหน่งไม่ให้ซ้อนกัน)
        ax.text(mid, h_mm + 150, f"{res['neg']['n']}-DB{int(res['top_db'])}", color=COLOR_TOP, ha='center', fontweight='bold')
        ax.text(mid, cover_mm + 60, f"{res['pos']['n']}-DB{int(res['bot_db'])}", color=COLOR_BOT, ha='center')
        ax.text(mid, -250, f"RB{int(res['stir_db'])}@{int(res['shear']['s'])}", color=COLOR_STIRRUP, ha='center', fontsize=8)

        x_curr += span_L

    _draw_dim_line(ax, (0, h_mm), (total_L, h_mm), f"Total L = {total_L/1000:.2f} m", offset=400)

    ax.set_aspect('auto') # บังคับสัดส่วนให้ยืดตามแนวนอน
    ax.axis('off')
    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-700, h_mm + 700)
    return fig
