import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_mm, cover_mm):
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 350 
    
    fig, ax = plt.subplots(figsize=(max(16, total_L/350), 4.5))
    
    # 1. วาดคอนกรีต
    ax.add_patch(patches.Rectangle((0, 0), total_L, v_h, lw=2, ec='black', fc='white', zorder=2))
    
    curr_x = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        s_spacing = res['shear']['s']
        
        # --- เพิ่มเหล็กปลอก (Stirrups) เส้นบางสีอ่อน ---
        num_stirrups = int(span_L / s_spacing)
        for j in range(num_stirrups + 1):
            sx = curr_x + (j * s_spacing)
            if sx <= curr_x + span_L:
                ax.plot([sx, sx], [cover_mm, v_h - cover_mm], 
                        color='#bdc3c7', lw=0.8, ls='-', zorder=3)
        
        # วาดเหล็กเมน (บน/ล่าง)
        y_t, y_b = v_h - cover_mm - 10, cover_mm + 10
        ax.plot([curr_x, curr_x + span_L], [y_t, y_t], color='#d30000', lw=3.5, zorder=10)
        ax.plot([curr_x + 40, curr_x + span_L - 40], [y_b, y_b], color='#008c00', lw=3.5, zorder=10)
        
        # ป้ายบอกเหล็ก
        mid = curr_x + span_L/2
        label_opt = dict(ha='center', fontweight='bold', fontsize=10, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        ax.text(mid, v_h + 60, f"{int(res['neg']['n'])}-DB{int(res['top_db'])}", color='#d30000', **label_opt)
        ax.text(mid, y_b - 40, f"{int(res['pos']['n'])}-DB{int(res['bot_db'])}", color='#008c00', va='top', **label_opt)
        
        # ระบุเหล็กปลอกด้านล่าง
        ax.text(mid, -120, f"RB{int(res['stir_db'])}@{int(s_spacing)}", color='#7f8c8d', fontsize=9, ha='center', style='italic')
        
        curr_x += span_L

    # วาด Grid Lines
    gx = 0
    for i, s_mm in enumerate(spans_mm + [0]):
        ax.plot([gx, gx], [-400, v_h + 300], color='#7f8c8d', ls='-.', lw=1, zorder=1)
        ax.text(gx, v_h + 350, chr(65+i), ha='center', fontweight='bold', fontsize=14)
        if i < len(spans_mm): gx += s_mm

    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-500, v_h + 600)
    
    # แปลงเป็น SVG สำหรับแสดงผล
    f_svg = io.StringIO()
    fig.savefig(f_svg, format="svg", bbox_inches='tight')
    svg_data = f_svg.getvalue()
    
    # แปลงเป็น PNG สำหรับดาวน์โหลด (High DPI)
    f_png = io.BytesIO()
    fig.savefig(f_png, format="png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return svg_data, f_png.getvalue()
