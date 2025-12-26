def analyze_structure(spans_data, supports_data, loads_data):
    """
    วิเคราะห์คานโดยใช้ anaStruct (2D FEM) - Fixed Node ID Issue
    """
    ss = SystemElements(EA=15000, EI=5000) 
    
    # 1. สร้าง Elements (คานแต่ละช่วง)
    start_x = 0
    for length in spans_data:
        end_x = start_x + length
        ss.add_element(location=[[start_x, 0], [end_x, 0]])
        start_x = end_x
    
    # 2. ใส่ Supports (จุดรองรับ)
    # --- แก้ไขตรงนี้: Node ของ anastruct เริ่มที่ 1 ไม่ใช่ 0 ---
    for i, supp_type in enumerate(supports_data):
        node_id = i + 1  # <--- เปลี่ยนจาก i เป็น i + 1
        
        if supp_type == 'Fix':
            ss.add_support_fixed(node_id=node_id)
        elif supp_type == 'Pin':
            ss.add_support_hinged(node_id=node_id)
        elif supp_type == 'Roller':
            ss.add_support_roll(node_id=node_id, direction=1) 

    # 3. ใส่ Loads (Apply Load Combination)
    for load in loads_data:
        # Load Combination
        mag_dead = load['dl'] + load['sdl']
        mag_live = load['ll']
        wu_total = (FACTOR_DL * mag_dead) + (FACTOR_LL * mag_live)
        
        span_idx = load['span_idx']
        element_id = span_idx + 1 # Element ก็เริ่มที่ 1 เช่นกัน (บรรทัดนี้ถูกแล้ว)
        
        if load['type'] == 'Uniform Load':
            ss.q_load(q=wu_total, element_id=element_id)
            
        elif load['type'] == 'Point Load':
            ss.point_load(node_id=None, element_id=element_id, position=load['pos'], Fy=-wu_total)
    
    # 4. Analyze
    ss.solve()
    
    return ss
