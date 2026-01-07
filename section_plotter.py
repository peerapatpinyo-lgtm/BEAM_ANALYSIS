import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b, h, cover_mm, db_mm, n_top, n_bot, stirrup_info, fc, fy):
    fig, ax = plt.subplots(figsize=(3.5, 4))
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=1.5, ec='#2C3E50', fc='#E5E7E9'))
    
    c, db = cover_mm/1000, db_mm/1000
    if b-2*c > 0: ax.add_patch(patches.FancyBboxPatch((c, c), b-2*c, h-2*c, boxstyle="round,pad=0,rounding_size=0.01", lw=1.2, ec='#C0392B', fc='none', ls='--'))
    
    def draw_bars(n, y, color):
        if n <= 0: return
        xs = [b/2] if n==1 else np.linspace(c+db/2, b-c-db/2, n)
        for x in xs: ax.add_patch(patches.Circle((x, y), db/2, ec='black', fc=color, zorder=10))
            
    draw_bars(int(n_top), h-c-db/2, '#E74C3C')
    draw_bars(int(n_bot), c+db/2, '#2980B9')
    
    ax.text(b/2, -0.05, f"{n_bot}-DB{db_mm}", ha='center', color='#2980B9', weight='bold')
    ax.text(b/2, h+0.05, f"{n_top}-DB{db_mm}", ha='center', color='#E74C3C', weight='bold')
    ax.axis('off'); ax.set_aspect('equal')
    plt.tight_layout()
    return fig

def plot_longitudinal_section(spans, sup_df, design_data, h, cover_mm):
    cum_spans = [0] + list(np.cumsum(spans))
    total_len = cum_spans[-1]
    fig, ax = plt.subplots(figsize=(10, 3))
    
    ax.add_patch(patches.Rectangle((0, 0), total_len, h, lw=1.5, ec='black', fc='#FDFFE6'))
    c = cover_mm/1000.0
    
    for i, span in enumerate(spans):
        sx, ex = cum_spans[i], cum_spans[i+1]
        d = design_data[i]
        
        # Bot Bars with Hooks
        hook = 0.15
        ax.plot([sx+0.05, sx+0.05, ex-0.05, ex-0.05], [c+hook, c, c, c+hook], color='#2980B9', lw=2)
        ax.text((sx+ex)/2, c+0.1, f"{d['pos']['n']}-DB16", color='#2980B9', ha='center', size=8, weight='bold')
        
        # Top Bars
        Le = span/3
        if i==0: ax.plot([sx, sx+Le], [h-c, h-c], color='#E74C3C', lw=2)
        else: 
            ax.plot([sx-Le, sx+Le], [h-c, h-c], color='#E74C3C', lw=2)
            ax.text(sx, h-c-0.1, f"{d['neg']['n']}-DB16", color='#E74C3C', ha='center', size=8)

    # Supports
    for _, s in sup_df.iterrows():
        if s['type'] != 'None':
            x = cum_spans[int(s['id'])]
            ax.add_patch(patches.Polygon([[x,0], [x-0.1,-0.15], [x+0.1,-0.15]], closed=True, fc='gray'))

    ax.set_xlim(-0.5, total_len+0.5); ax.set_ylim(-0.3, h+0.2)
    ax.axis('off'); ax.set_aspect('equal')
    plt.tight_layout()
    return fig
