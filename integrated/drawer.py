import cv2

class Drawer:
    def __init__(self, fence_polygon):
        self.fence_polygon = fence_polygon

    def draw(self, frame, yolo_detections, fence_results, anpr_boxes, last_ocr, fps):
        # 1. Draw Fence Polygon
        cv2.polylines(frame, [self.fence_polygon], isClosed=True, color=(0, 0, 255), thickness=3)
        fx, fy = self.fence_polygon[0]
        cv2.putText(frame, "RESTRICTED ZONE", (int(fx), max(25, int(fy) - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # 2. Draw Tracked Objects (Persons, Vehicles, etc.)
        intruder_count = fence_results.get("outside_count", 0)
        safe_count = fence_results.get("inside_count", 0)
        
        for obj in fence_results.get("person_results", []):
            x1, y1, x2, y2 = obj["bbox"]
            is_intruder = obj["status"] == "inside"
            color = (0, 0, 255) if is_intruder else (0, 255, 0)
            status_text = "INTRUDER" if is_intruder else "SAFE"
            obj_desc = obj.get("display_label", f"{obj.get('class_name', 'OBJ').upper()} ID:{obj['track_id']}")
            label = f"{status_text} {obj_desc} {obj['confidence']:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, obj["foot_point"], 5, (0, 255, 255), -1)
            cv2.putText(frame, label, (x1, max(25, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # 4. Draw ANPR Boxes & Text
        for box_info in anpr_boxes:
            x1, y1, x2, y2 = box_info["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            # If text is available from this frame's OCR, display it.
            # Alternatively, if not running OCR this frame, we could try to map last_ocr based on overlap.
            text_to_draw = box_info["text"]
            confidence = box_info["confidence"]
            
            if not text_to_draw and last_ocr:
                # Simple fallback: if there's only 1 plate or we blindly show the first last_ocr text
                # A robust approach calculates Intersection over Union (IoU), but for simplicity:
                text_to_draw = last_ocr[0].get("text") if len(last_ocr) > 0 else None
                confidence = last_ocr[0].get("confidence", 0.0) if len(last_ocr) > 0 else 0.0
                
            if text_to_draw:
                label = f"{text_to_draw} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # 5. Draw HUD
        hud_color = (0, 0, 255) if intruder_count > 0 else (0, 255, 0)
        alert_text = "ALERT: ZONE BREACH!" if intruder_count > 0 else "STATUS: SECURE"
        
        cv2.putText(frame, alert_text, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.75, hud_color, 2)
        cv2.putText(frame, f"Safe: {safe_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, f"Intruders: {intruder_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(frame, f"Pipeline FPS: {fps:.1f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        
        return frame
