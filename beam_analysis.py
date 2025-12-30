import numpy as np
import pandas as pd
from indetermbeam import Beam, Support, PointLoadV, DistributedLoadV

class BeamAnalysisEngine:
    def __init__(self, spans, supports, loads):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.total_len = sum(spans)
        self.beam = Beam(self.total_len)
        self._setup_structure()

    def _setup_structure(self):
        cum_len = [0] + list(np.cumsum(self.spans))
        
        # 1. Supports
        for _, row in self.supports.iterrows():
            pos = cum_len[int(row['id'])]
            stype = row['type']
            # IndetermBeam convention: Support(coord, (kx, ky, km)) 1=Fixed, 0=Free
            if stype == "Pin":
                self.beam.add_supports(Support(pos, (1,1,0)))
            elif stype == "Roller":
                self.beam.add_supports(Support(pos, (0,1,0)))
            elif stype == "Fixed":
                self.beam.add_supports(Support(pos, (1,1,1)))

        # 2. Loads
        for l in self.loads:
            span_start = cum_len[l['span_idx']]
            if l['type'] == 'U':
                start = span_start
                end = cum_len[l['span_idx']+1]
                self.beam.add_loads(DistributedLoadV(-l['w'], (start, end)))
            elif l['type'] == 'P':
                pos = span_start + l['x']
                self.beam.add_loads(PointLoadV(-l['P'], pos))

    def solve(self):
        try:
            self.beam.analyze()
            
            # --- PRECISE SAMPLING STRATEGY ---
            # เพื่อแก้ปัญหา Max ไม่ลงที่ 0.00 หรือ Support
            # เราต้องสร้างจุด x ที่รวมตำแหน่งสำคัญ (Critical Points) ทั้งหมด
            
            points = set()
            
            # 1. Boundary Points (Start & End) -> บังคับใส่ 0.00
            points.add(0.0)
            points.add(self.total_len)
            
            # 2. Support Locations (All Nodes)
            cum_len = [0] + list(np.cumsum(self.spans))
            for x in cum_len:
                points.add(x)
                # Add epsilon points for Shear discontinuity (Left/Right)
                if x > 0: points.add(x - 1e-6)
                if x < self.total_len: points.add(x + 1e-6)

            # 3. Point Load Locations
            for l in self.loads:
                span_start = cum_len[l['span_idx']]
                if l['type'] == 'P':
                    px = span_start + l['x']
                    points.add(px)
                    points.add(px - 1e-6)
                    points.add(px + 1e-6)

            # 4. Dense Sampling (fill gaps)
            # เพิ่มความละเอียดเป็น 500 จุด เพื่อความเนียนของกราฟโค้ง (Parabola)
            dense_x = np.linspace(0, self.total_len, 501)
            for x in dense_x:
                points.add(x)

            # 5. Clean up
            # Sort and round to remove duplicates like 0.000000001
            sorted_points = sorted(list(points))
            x_arr = np.unique(np.round(sorted_points, 6)) # Round to 6 decimals
            
            # Filter valid range only
            x_arr = x_arr[(x_arr >= 0) & (x_arr <= self.total_len)]

            # --- CALCULATE ---
            shear = self.beam.get_shear(x_arr)
            moment = self.beam.get_bending_moment(x_arr)
            
            df = pd.DataFrame({'x': x_arr, 'shear': shear, 'moment': moment})
            reactions = self.beam.get_reaction_forces()
            
            return df, reactions

        except Exception as e:
            # st.error(f"Engine Error: {e}") # Debug only
            return None, None
