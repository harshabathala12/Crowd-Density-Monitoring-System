import numpy as np

class CrowdAnalytics:
    def __init__(self):
        # 1. TUNING
        self.DENSITY_WEIGHT = 0.5
        self.MOTION_WEIGHT = 0.5 
        
        # 2. CALIBRATION (Adjust this based on your scene)
        self.MAX_CAPACITY = 100.0 
        
        # 3. THRESHOLDS (0-100 Scale)
        self.THRESH_WARNING = 40.0   # > 30 = Orange
        self.THRESH_CRITICAL = 50.0  # > 40 = Red + Email + Freeze
        
        self.sectors = {
            "North-West": (0, 0, 0.5, 0.5),
            "North-East": (0.5, 0, 1.0, 0.5),
            "South-West": (0, 0.5, 0.5, 1.0),
            "South-East": (0.5, 0.5, 1.0, 1.0)
        }

    def analyze_frame(self, detections, flow_magnitude):
        count = len(detections)
        
        # 1. Density (0.0 - 1.0)
        density_score = min(count / self.MAX_CAPACITY, 1.0)
        
        # 2. Motion (0.0 - 1.0)
        motion_score = min(flow_magnitude / 10.0, 1.0) # Assuming 10.0 is high motion
        
        # 3. Calculate Index (0.0 - 1.0)
        raw_index = (density_score * self.DENSITY_WEIGHT) + \
                    (motion_score * self.MOTION_WEIGHT)
        
        # 4. CONVERT TO 0-100 SCALE
        final_index = round(raw_index * 100, 1)
        
        # 5. Determine Status (Using 0-100 Integers)
        status = "NORMAL"
        if final_index >= self.THRESH_CRITICAL:
            status = "CRITICAL"
        elif final_index >= self.THRESH_WARNING:
            status = "WARNING"

        hotspot_name = self._get_active_sector(detections)

        return {
            "count": count,
            "congestion_index": final_index, # Already 0-100
            "status": status,
            "hotspot": hotspot_name
        }

    def _get_active_sector(self, detections):
        sector_counts = {k: 0 for k in self.sectors}
        for box in detections:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            if cx < 0.5 and cy < 0.5: sector_counts["North-West"] += 1
            elif cx >= 0.5 and cy < 0.5: sector_counts["North-East"] += 1
            elif cx < 0.5 and cy >= 0.5: sector_counts["South-West"] += 1
            else: sector_counts["South-East"] += 1
        return max(sector_counts, key=sector_counts.get)