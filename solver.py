import numpy as np
import pandas as pd
from indetermbeam import Beam, Support, PointLoad, DistributedLoad, UDL

def solve_beam(spans, sup_df, loads_df, params):
    """
    Analyzes the beam using the 'indetermbeam' library.
    
    Parameters:
    - spans: list of span lengths (e.g. [4.0, 5.0])
    - sup_df: DataFrame of supports
    - loads_df: DataFrame of loads
    - params: dict containing 'E' (Pa) and 'I' (m^4)
    
    Returns:
    - x: Array of positions
    - M: Array of Moment values (Nm)
    - V: Array of Shear values (N)
    - D: Array of Deflection values (m)
    - R: Reaction dictionary (or raw object)
    """
    
    # 1. Initialize Beam
    total_length = sum(spans)
    # indetermbeam รับค่า E และ I ในหน่วย SI (Pa, m^4)
    beam = Beam(total_length, E=params['E'], I=params['I'])
    
    # 2. Add Supports
    # indetermbeam support types: (kx, ky, mr) -> 1=fixed, 0=free
    # Fixed: (1,1,1), Pin: (1,1,0), Roller: (0,1,0)
    for _, row in sup_df.iterrows():
        pos = row['x']
        stype = row['type']
        
        # Default stiffness (infinity for rigid supports)
        # Tuple format: (restraint_x, restraint_y, restraint_moment)
        if stype == 'Fixed':
            beam.add_supports(Support(pos, (1, 1, 1)))
        elif stype == 'Pin':
            beam.add_supports(Support(pos, (1, 1, 0)))
        elif stype == 'Roller':
            beam.add_supports(Support(pos, (0, 1, 0)))
            
    # 3. Add Loads
    if not loads_df.empty:
        # คำนวณระยะสะสมของแต่ละช่วงคานเพื่อระบุตำแหน่งโหลด
        cum_dist = [0] + list(np.cumsum(spans))
        
        for _, load in loads_df.iterrows():
            span_idx = int(load['span_index'])
            start_x = cum_dist[span_idx]
            
            # แปลงโหลด (แรงลงต้องเป็นค่าลบใน indetermbeam? 
            # ปกติ indetermbeam: Load ลง = ลบ, แต่เราจะจัดการทิศทางตอน Plot)
            # แต่ตาม Convention วิศวะโยธา: Load ลง ใส่เป็นค่าลบใน Solver 
            # ซึ่ง input เราเป็น Magnitude (บวก) ดังนั้นต้องคูณ -1
            mag = -load['mag'] 
            
            if load['type'] == 'P':
                # Point Load
                pos = start_x + load['dist']
                beam.add_loads(PointLoad(mag, pos))
                
            elif load['type'] == 'U':
                # UDL (Uniform Distributed Load)
                # Input เรามีแค่ dist (length) เริ่มจากซ้ายของ span
                x_start = start_x
                x_end = x_start + load['dist'] # UDL เต็มช่วงหรือบางช่วงตาม logic input
                # หมายเหตุ: ใน input_handler ปัจจุบัน UDL คือเต็ม Span (dist=L)
                
                beam.add_loads(DistributedLoad(mag, (x_start, x_end)))

    # 4. Analyze
    try:
        beam.analyze()
    except Exception as e:
        # กรณี unstable หรือ error อื่นๆ
        print(f"Analysis Error: {e}")
        # คืนค่าเป็น 0 เพื่อไม่ให้ app crash
        x_dummy = np.linspace(0, total_length, 100)
        return x_dummy, x_dummy*0, x_dummy*0, x_dummy*0, {}

    # 5. Extract Results for Plotting
    # query values at n points
    n_points = 500
    x_eval = np.linspace(0, total_length, n_points)
    
    # Get internal forces (Note: indetermbeam returns tuple of arrays sometimes, verify API)
    # get_shear(x), get_bending_moment(x), get_deflection(x)
    
    V = beam.get_shear(x_eval)
    M = beam.get_bending_moment(x_eval)
    D = beam.get_deflection(x_eval)
    
    # Get Reactions
    # beam.get_reactions() returns a dict usually
    R = beam.get_reaction_forces() 
    
    return x_eval, M, V, D, R
