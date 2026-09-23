import cv2
import numpy as np

def select_zone(frame):
    points = []
    window_name = "Draw Restricted Zone (Click points, Press ENTER to finish, 'r' to reset, 'c' to cancel)"

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display_frame = frame.copy()
        
        # Draw the points and lines
        if len(points) > 0:
            for i, p in enumerate(points):
                cv2.circle(display_frame, p, 5, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(display_frame, points[i - 1], points[i], (0, 0, 255), 2)
            
            # Close the polygon visually if there are at least 3 points
            if len(points) >= 3:
                cv2.line(display_frame, points[-1], points[0], (0, 0, 255), 2)
                
                # Fill the polygon with a semi-transparent overlay
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [np.array(points)], (0, 0, 255))
                cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)

        cv2.putText(display_frame, "Click to draw polygon. Press ENTER to confirm, 'R' to reset.", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER key
            if len(points) >= 3:
                break
            else:
                print("[WARNING] Please draw at least 3 points before pressing ENTER.")
        elif key == ord('r'):
            points = []
        elif key == ord('c'):
            # Return a default polygon if cancelled
            points = [(200, 150), (700, 150), (750, 500), (150, 500)]
            break

    cv2.destroyWindow(window_name)
    
    return np.array(points, dtype=np.int32)
