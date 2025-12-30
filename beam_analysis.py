import numpy as np
import pandas as pd
from indetermbeam import Beam, Support, PointLoadV, DistributedLoadV

class BeamAnalysisEngine:
    def __init__(self, spans, supports, loads):
        self.spans = spans
        self.supports = supports
        self.loads = loads
        self.beam = Beam(sum(spans))
        self._setup_structure()

    def _setup_structure(self):
        # 1. Supports
        cum_len = [0] + list(np.cumsum(self.spans))
        for _, row in self.supports.iterrows():
            pos = cum_len[int(row['id'])]
            stype = row['type']
            
            # Map UI Types to IndetermBeam Types
            if stype == "Pin":
                self.beam.add_supports(Support(pos, (1,1,0))) # Fix x,y
            elif stype == "Roller":
                self.beam.add_supports(Support(pos, (0,1,0))) # Fix y
            elif stype == "Fixed":
                self.beam.add_supports(Support(pos, (1,1,1))) # Fix x,y,m

        # 2. Loads
        for l in self.loads:
            span_start = cum_len[l['span_idx']]
            if l['type'] == 'U':
                # Uniform Load (start, end, val)
                # Note: Indetermbeam uses negative for downward if consistent
                # But here we assume input positive = gravity downward, handle sign later
                start = span_start
                end = cum_len[l['span_idx']+1]
                self.beam.add_loads(DistributedLoadV(-l['w'], (start, end)))
            elif l['type'] == 'P':
                pos = span_start + l['x']
                self.beam.add_loads(PointLoadV(-l['P'], pos))

    def solve(self):
        try:
            self.beam.analyze()
            
            # --- CRITICAL FIX: SMART SAMPLING ---
            # สร้างจุด x ที่ละเอียดเป็นพิเศษตรง Support และ Load เพื่อจับ Peak
            
            x_points = set()
            cum_len = [0] + list(np.cumsum(self.spans))
            
            # 1. Add Span ends (Supports) with tiny offsets for Shear jumps
            for x in cum_len:
                x_points.add(x)
                x_points.add(x - 1e-5) # Just left
                x_points.add(x + 1e-5) # Just right

            # 2. Add Point Load locations
            for l in self.loads:
                span_start = cum_len[l['span_idx']]
                if l['type'] == 'P':
                    px = span_start + l['x']
                    x_points.add(px)
                    x_points.add(px - 1e-5)
                    x_points.add(px + 1e-5)

            # 3. Add regular intervals (High resolution)
            n_segments = 200 # increase resolution
            total_len = sum(self.spans)
            regular = np.linspace(0, total_len, n_segments)
            x_points.update(regular)

            # Sort and filter valid
            x_final = sorted([x for x in x_points if 0 <= x <= total_len])
            x_arr = np.array(x_final)

            # Calculate Forces
            shear = self.beam.get_shear(x_arr)
            moment = self.beam.get_bending_moment(x_arr)
            
            # DataFrame for plotting
            df = pd.DataFrame({
                'x': x_arr,
                'shear': shear,
                'moment': moment
            })
            
            # Get Reactions
            reactions = self.beam.get_reaction_forces()
            
            return df, reactions

        except Exception as e:
            return None, None
