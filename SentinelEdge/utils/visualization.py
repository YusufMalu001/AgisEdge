import cv2
import numpy as np

def draw_detections(img, detections, classes=["vehicle", "human", "drone"]):
    """
    Draws bounding boxes and labels onto the image frame.
    Detections format: [x1, y1, x2, y2, confidence, class_id]
    """
    annotated = img.copy()
    
    # Consistent color map for professional look (DRDO/Defense UAV HUD feel)
    # class 0 (vehicle): Orange-yellow, class 1 (human): Red, class 2 (drone): Bright Blue
    colors = {
        0: (0, 215, 255),   # gold/yellow
        1: (50, 50, 240),   # light red
        2: (240, 150, 50)   # cyan/light blue
    }
    
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        color = colors.get(cls_id, (0, 255, 0))
        label = f"{classes[cls_id] if cls_id < len(classes) else 'unknown'}: {conf:.2f}"
        
        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # HUD corner style lines for high quality defense feel
        length = min(15, int((x2 - x1) * 0.25))
        # Top-Left corner
        cv2.line(annotated, (x1, y1), (x1 + length, y1), color, 4)
        cv2.line(annotated, (x1, y1), (x1, y1 + length), color, 4)
        # Top-Right corner
        cv2.line(annotated, (x2, y1), (x2 - length, y1), color, 4)
        cv2.line(annotated, (x2, y1), (x2, y1 + length), color, 4)
        # Bottom-Left corner
        cv2.line(annotated, (x1, y2), (x1 + length, y2), color, 4)
        cv2.line(annotated, (x1, y2), (x1, y2 - length), color, 4)
        # Bottom-Right corner
        cv2.line(annotated, (x2, y2), (x2 - length, y2), color, 4)
        cv2.line(annotated, (x2, y2), (x2, y2 - length), color, 4)

        # Label background
        (w_lbl, h_lbl), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, y1 - h_lbl - 5), (x1 + w_lbl + 5, y1), color, -1)
        
        # Label text (black text over solid block)
        cv2.putText(
            annotated, label, (x1 + 3, y1 - 3), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
        )
        
    return annotated

def draw_hud_overlay(img, fps, latency, backend_name):
    """
    Draws a telemetry HUD overlay on top of the drone video frame.
    """
    annotated = img.copy()
    h, w = img.shape[:2]
    
    # Overlay background panel (semi-transparent gray)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (5, 5), (280, 85), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
    
    # HUD text details
    hud_color = (0, 255, 0)  # Green tactical overlay
    cv2.putText(annotated, f"SYS_BACKEND: {backend_name.upper()}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1, cv2.LINE_AA)
    cv2.putText(annotated, f"FPS_TARGET: 30.0 | ACTUAL: {fps:.1f}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1, cv2.LINE_AA)
    cv2.putText(annotated, f"INF_LATENCY: {latency * 1000:.2f} ms", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1, cv2.LINE_AA)
    
    # Corner brackets representing drone viewfinder
    bracket_len = 30
    bracket_color = (0, 255, 0)
    # TL
    cv2.line(annotated, (10, 10), (10 + bracket_len, 10), bracket_color, 2)
    cv2.line(annotated, (10, 10), (10, 10 + bracket_len), bracket_color, 2)
    # TR
    cv2.line(annotated, (w - 10, 10), (w - 10 - bracket_len, 10), bracket_color, 2)
    cv2.line(annotated, (w - 10, 10), (w - 10, 10 + bracket_len), bracket_color, 2)
    # BL
    cv2.line(annotated, (10, h - 10), (10 + bracket_len, h - 10), bracket_color, 2)
    cv2.line(annotated, (10, h - 10), (10, h - 10 - bracket_len), bracket_color, 2)
    # BR
    cv2.line(annotated, (w - 10, h - 10), (w - 10 - bracket_len, h - 10), bracket_color, 2)
    cv2.line(annotated, (w - 10, h - 10), (w - 10, h - 10 - bracket_len), bracket_color, 2)
    
    return annotated
