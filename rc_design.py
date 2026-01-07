# ... (Code เดิมที่มี design_span_expert อยู่ด้านบน) ...

def get_steel_weight(diameter_mm):
    """Calculate steel weight (kg/m) based on diameter"""
    # Formula: D^2 / 162
    return (diameter_mm ** 2) / 162.0

def generate_bbs(design_res, spans, b, h, cover_mm):
    """
    Generate Bar Bending Schedule (BBS) Data
    """
    bbs_data = []
    
    # 1. Longitudinal Bars (Main Reinforcement)
    for i, res in enumerate(design_res):
        span_len = spans[i]
        
        # Bottom Bars (Positive Moment)
        n_bot = res['pos']['n']
        db_bot = res['db'] # Assumed DB16 from main code
        if n_bot > 0:
            # Length approx = Span + Anchorage (Simplified)
            # In real detail: Span - Cover + Hooks
            len_bar = span_len + 0.3 + 0.3 # +30cm hooks both sides
            w_unit = get_steel_weight(db_bot)
            total_w = n_bot * len_bar * w_unit
            
            bbs_data.append({
                "Span": f"Span {i+1}",
                "Position": "Bottom (Main)",
                "Bar": f"DB{db_bot}",
                "Shape": "U-Hook",
                "No. of Bars": n_bot,
                "Length (m)": round(len_bar, 2),
                "Total Wt (kg)": round(total_w, 2)
            })
            
        # Top Bars (Negative Moment)
        n_top = res['neg']['n']
        db_top = res['db']
        if n_top > 0:
            # Top bars usually cover L/3 of adjacent spans
            len_bar = (span_len / 3.0) * 2 # Simplified
            w_unit = get_steel_weight(db_top)
            total_w = n_top * len_bar * w_unit
            
            bbs_data.append({
                "Span": f"Span {i+1}",
                "Position": "Top (Support)",
                "Bar": f"DB{db_top}",
                "Shape": "Straight",
                "No. of Bars": n_top,
                "Length (m)": round(len_bar, 2),
                "Total Wt (kg)": round(total_w, 2)
            })

    # 2. Stirrups (Shear Reinforcement)
    cover = cover_mm / 1000.0
    stirrup_w = b - 2*cover
    stirrup_h = h - 2*cover
    # Stirrup Length = Perimeter + Hooks (approx 0.1m)
    len_stir = (2 * stirrup_w) + (2 * stirrup_h) + 0.15 
    
    for i, res in enumerate(design_res):
        span_len = spans[i]
        s_info = res['shear_stirrups'] # e.g., "RB9 @ 0.15 m"
        
        try:
            # Parse spacing
            parts = s_info.split('@')
            if len(parts) > 1:
                db_stir = 9 # RB9 (Fixed for now or parse string)
                spacing = float(parts[1].replace('m', '').strip())
                
                # Number of stirrups = (Span / spacing) + 1
                n_stir = int(span_len / spacing) + 1
                w_unit = get_steel_weight(db_stir)
                total_w = n_stir * len_stir * w_unit
                
                bbs_data.append({
                    "Span": f"Span {i+1}",
                    "Position": "Stirrup",
                    "Bar": f"RB{db_stir}",
                    "Shape": "Rect-Ring",
                    "No. of Bars": n_stir,
                    "Length (m)": round(len_stir, 2),
                    "Total Wt (kg)": round(total_w, 2)
                })
        except:
            pass # Skip if format error
            
    return bbs_data

def get_boq(spans, b, h, bbs_data):
    """
    Generate Bill of Quantities (Concrete Vol & Steel Weight)
    """
    # Concrete Volume
    total_len = sum(spans)
    vol_conc = total_len * b * h
    
    # Steel Weight
    total_steel = sum([item['Total Wt (kg)'] for item in bbs_data])
    
    return vol_conc, total_steel
