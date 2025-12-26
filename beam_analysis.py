import numpy as np
import pandas as pd

def run_beam_analysis(spans, supports, loads):
    """
    Direct Stiffness Method with COMPREHENSIVE DATA NORMALIZATION.
    Fixes 'string indices must be integers' by converting all DataFrames to Dicts first.
    """
    
    # ==========================================
    # 1. DATA NORMALIZATION (THE FIX)
    # ==========================================
    
    # --- Clean LOADS ---
    # แปลง loads ให้เป็น List of Dicts เสมอ ไม่ว่าจะมาเป็น DataFrame หรือ List
    clean_loads = []
    if isinstance(loads, pd.DataFrame):
        clean_loads = loads.to_dict('records')
    elif isinstance(loads, list):
        clean_loads = loads
    
    # --- Clean SUPPORTS ---
    # แปลง supports ให้เป็น List of Dicts ที่มี key 'type'
    clean_supports = []
    n_nodes = len(spans) + 1
    
    for i in range(n_nodes):
        # ดึงค่าดิบ
        raw = None
        if isinstance(supports, list):
            raw = supports[i] if i < len(supports) else None
        elif isinstance(supports, (pd.DataFrame, pd.Series)):
             # ถ้าเป็น DF ให้ใช้ iloc
            try:
                raw = supports.iloc[i]
            except:
                raw = None
        
        # แปลงเป็น String identifier
        sup_type = "None"
        if isinstance(raw, str):
            sup_type = raw
        elif isinstance(raw, dict):
            sup_type = raw.get('type', 'None')
        elif hasattr(raw, 'get'): # Series/Object
            sup_type = raw.get('type', 'None')
        
        # เก็บใน format มาตรฐาน
        clean_supports.append({'type': sup_type})

    # ==========================================
    # 2. SYSTEM SETUP
    # ==========================================
    n_spans = len(spans)
    dof_per_node = 2
    total_dof = n_nodes * dof_per_node
    
    K_global = np.zeros((total_dof, total_dof))
    F_global = np.zeros(total_dof)
    E = 2e6  
    I = 0.001 

    # ==========================================
    # 3. MATRIX ASSEMBLY & FEM
    # ==========================================
    for i in range(n_spans):
        L = spans[i]
        L2 = L * L; L3 = L2 * L
        
        # Element Stiffness
        k_local = (E * I / L3) * np.array([
            [12,      6*L,    -12,     6*L],
            [6*L,     4*L2,   -6*L,    2*L2],
            [-12,    -6*L,     12,    -6*L],
            [6*L,     2*L2,   -6*L,    4*L2]
        ])

        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        for r in range(4):
            for c in range(4):
                K_global[idx[r], idx[c]] += k_local[r, c]

        # FEM Calculation (Using clean_loads)
        fem_vec = np.zeros(4) 
        
        # Filter loads for this span
        # ต้องใช้ .get() เผื่อ user ไม่กรอกบางช่อง และแปลงเป็น float เพื่อความชัวร์
        current_span_loads = [
            l for l in clean_loads 
            if int(l.get('span_idx', -1)) == i
        ]
        
        for load in current_span_loads:
            l_type = load.get('type', 'None')
            
            # Safe parsing of numbers
            try: w = -float(load.get('w', 0.0))
            except: w = 0.0
            
            try: P = -float(load.get('P', 0.0))
            except: P = 0.0
            
            try: x_loc = float(load.get('x', 0.0))
            except: x_loc = 0.0

            if l_type == 'U' and abs(w) > 0:
                fem_vec[0] += w * L / 2       
                fem_vec[1] += w * L2 / 12     
                fem_vec[2] += w * L / 2       
                fem_vec[3] += -w * L2 / 12    

            elif l_type == 'P' and abs(P) > 0:
                a = x_loc
                b = L - a
                if 0 <= a <= L: # Allow load at ends
                    fem_vec[0] += P * (b**2 * (3*a + b)) / L3  
                    fem_vec[1] += P * a * (b**2) / L2          
                    fem_vec[2] += P * (a**2 * (a + 3*b)) / L3  
                    fem_vec[3] += -P * (a**2) * b / L2         

        for r in range(4):
            F_global[idx[r]] -= fem_vec[r]

    # ==========================================
    # 4. BOUNDARY CONDITIONS
    # ==========================================
    free_dof = np.ones(total_dof, dtype=bool)
    vis_supports_data = [] 

    for i in range(n_nodes):
        stype = clean_supports[i]['type'] # Safe now
        vis_supports_data.append({'x': sum(spans[:i]), 'type': stype})

        dof_y = 2*i
        dof_m = 2*i + 1
        
        if stype == "Pin":
            free_dof[dof_y] = False 
        elif stype == "Roller":
            free_dof[dof_y] = False 
        elif stype == "Fixed":
            free_dof[dof_y] = False 
            free_dof[dof_m] = False 

    # ==========================================
    # 5. SOLVER
    # ==========================================
    K_reduced = K_global[np.ix_(free_dof, free_dof)]
    F_reduced = F_global[free_dof]
    
    try:
        D_reduced = np.linalg.solve(K_reduced, F_reduced)
    except np.linalg.LinAlgError:
        D_reduced = np.zeros_like(F_reduced) 

    D_full = np.zeros(total_dof)
    D_full[free_dof] = D_reduced

    # ==========================================
    # 6. POST-PROCESS (Internal Forces)
    # ==========================================
    x_coords = []
    shear_vals = []
    moment_vals = []
    span_ids = []
    global_x = 0.0
    
    for i in range(n_spans):
        L = spans[i]
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        u = D_full[idx] 
        
        # Recalculate FEM for plotting logic
        L2 = L*L; L3 = L2*L
        k_local = (E * I / L3) * np.array([
            [12,      6*L,    -12,     6*L],
            [6*L,     4*L2,   -6*L,    2*L2],
            [-12,    -6*L,     12,    -6*L],
            [6*L,     2*L2,   -6*L,    4*L2]
        ])
        
        fem_vec = np.zeros(4)
        current_span_loads = [l for l in clean_loads if int(l.get('span_idx', -1)) == i]
        
        for load in current_span_loads:
            l_type = load.get('type', 'None')
            try: w = -float(load.get('w', 0.0))
            except: w = 0.0
            try: P = -float(load.get('P', 0.0))
            except: P = 0.0
            try: x_loc = float(load.get('x', 0.0))
            except: x_loc = 0.0

            if l_type == 'U':
                fem_vec[0] += w * L / 2
                fem_vec[1] += w * L2 / 12
                fem_vec[2] += w * L / 2
                fem_vec[3] += -w * L2 / 12
            elif l_type == 'P':
                a = x_loc
                b = L - a
                fem_vec[0] += P * (b**2 * (3*a + b)) / L3
                fem_vec[1] += P * a * (b**2) / L2
                fem_vec[2] += P * (a**2 * (a + 3*b)) / L3
                fem_vec[3] += -P * (a**2) * b / L2

        f_end = np.dot(k_local, u) + fem_vec
        V_start = f_end[0]
        M_start = f_end[1] 
        
        # Plotting Points
        num_pts = 50
        x_span = np.linspace(0, L, num_pts)
        for load in current_span_loads:
            if load.get('type') == 'P':
                lx = float(load.get('x', 0.0))
                # Add points around load for sharp jump
                x_span = np.append(x_span, [lx - 1e-5, lx + 1e-5])
        
        x_span = np.sort(np.unique(x_span))
        x_span = x_span[(x_span >= 0) & (x_span <= L)]

        for x in x_span:
            V_x = V_start
            M_x = M_start + V_start * x
            
            for load in current_span_loads:
                l_type = load.get('type')
                try: w_mag = float(load.get('w', 0.0))
                except: w_mag = 0.0
                try: P_mag = float(load.get('P', 0.0))
                except: P_mag = 0.0
                try: px = float(load.get('x', 0.0))
                except: px = 0.0

                if l_type == 'U':
                    V_x -= w_mag * x
                    M_x -= w_mag * x**2 / 2
                elif l_type == 'P':
                    if x > px:
                        V_x -= P_mag
                        M_x -= P_mag * (x - px)
            
            x_coords.append(global_x + x)
            shear_vals.append(V_x)
            moment_vals.append(M_x) 
            span_ids.append(i)
            
        global_x += L

    res_df = pd.DataFrame({
        'x': x_coords,
        'shear': shear_vals,
        'moment': moment_vals,
        'span_id': span_ids
    })
    
    return res_df, pd.DataFrame(vis_supports_data)
