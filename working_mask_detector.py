import cv2
import numpy as np
import os

class WorkingMaskDetector:
    def __init__(self):
        # Initialize face detection using multiple cascades for better accuracy
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        
        # Initialize camera without setting specific properties
        self.cap = cv2.VideoCapture(0)
        
        # Colors
        self.MASK_COLOR = (0, 0, 255)  # Red for mask
        self.NO_MASK_COLOR = (0, 255, 0)  # Green for no mask
        self.TEXT_COLOR = (255, 255, 255)  # White text
        
        # Detection parameters
        self.mask_threshold = 0.3  # Threshold for mask detection
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Camera opened successfully!")
        
    def detect_faces(self, gray):
        """Detect faces using multiple cascades for better accuracy"""
        faces1 = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        faces2 = self.face_cascade_alt.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Combine detections
        all_faces = list(faces1) + list(faces2)
        
        # Remove duplicates and merge overlapping detections
        return self.merge_overlapping_faces(all_faces)
    
    def merge_overlapping_faces(self, faces):
        """Merge overlapping face detections"""
        if len(faces) == 0:
            return []
        
        # Convert to list of rectangles
        rects = [list(face) for face in faces]
        
        # Sort by area (largest first)
        rects.sort(key=lambda x: x[2] * x[3], reverse=True)
        
        merged = []
        for rect in rects:
            x, y, w, h = rect
            
            # Check if this rectangle overlaps significantly with any existing one
            should_add = True
            for existing in merged:
                ex, ey, ew, eh = existing
                
                # Calculate overlap
                overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = overlap_x * overlap_y
                
                # If overlap is more than 50% of smaller rectangle, don't add
                smaller_area = min(w * h, ew * eh)
                if overlap_area > 0.5 * smaller_area:
                    should_add = False
                    break
            
            if should_add:
                merged.append(rect)
        
        return merged
    
    def simple_mask_detection(self, face_img):
        """Simple rule-based mask detection using color and texture analysis"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Analyze lower face region (where masks typically are)
            height, width = face_img.shape[:2]
            lower_face = face_img[height//2:, :]
            lower_face_hsv = hsv[height//2:, :]
            
            # Check for common mask colors (white, blue, black)
            # White mask detection
            white_lower = np.array([0, 0, 200])
            white_upper = np.array([180, 30, 255])
            white_mask = cv2.inRange(lower_face_hsv, white_lower, white_upper)
            
            # Blue mask detection
            blue_lower = np.array([100, 50, 50])
            blue_upper = np.array([130, 255, 255])
            blue_mask = cv2.inRange(lower_face_hsv, blue_lower, blue_upper)
            
            # Black mask detection
            black_lower = np.array([0, 0, 0])
            black_upper = np.array([180, 255, 50])
            black_mask = cv2.inRange(lower_face_hsv, black_lower, black_upper)
            
            # Combine all mask detections
            combined_mask = white_mask | blue_mask | black_mask
            
            # Calculate mask coverage percentage
            mask_pixels = cv2.countNonZero(combined_mask)
            total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
            mask_coverage = mask_pixels / total_pixels if total_pixels > 0 else 0
            
            # Additional texture analysis
            # Calculate standard deviation of gray values (masks tend to have lower variance)
            gray_lower = gray[height//2:, :]
            texture_variance = np.std(gray_lower)
            
            # Decision logic
            if mask_coverage > self.mask_threshold or texture_variance < 30:
                return "Mask", mask_coverage
            else:
                return "No Mask", 1 - mask_coverage
                
        except Exception as e:
            print(f"Error in mask detection: {e}")
            return "Unknown", 0.0
    
    def draw_triangle(self, frame, x, y, w, h, color):
        """Draw a red triangle on the face when mask is detected"""
        # Calculate triangle points
        center_x = x + w // 2
        top_y = y - 10
        left_x = center_x - 20
        right_x = center_x + 20
        bottom_y = y + 30
        
        # Draw filled triangle
        triangle_points = np.array([[center_x, top_y], [left_x, bottom_y], [right_x, bottom_y]], np.int32)
        cv2.fillPoly(frame, [triangle_points], color)
        
        # Draw triangle outline
        cv2.polylines(frame, [triangle_points], True, (0, 0, 0), 2)
    
    def draw_face_annotation(self, frame, x, y, w, h, prediction, confidence):
        """Draw face detection box and annotations"""
        # Determine color
        if prediction == "Mask":
            color = self.MASK_COLOR
            label = f"Mask Detected ({confidence:.2f})"
        else:
            color = self.NO_MASK_COLOR
            label = f"No Mask ({confidence:.2f})"
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw triangle for mask detection
        if prediction == "Mask":
            self.draw_triangle(frame, x, y, w, h, self.MASK_COLOR)
        
        # Draw label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y - 30), (x + label_size[0], y), color, -1)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 2)
    
    def add_info_panel(self, frame, face_count, mask_count):
        """Add information panel to the frame"""
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text information
        cv2.putText(frame, "Face Mask Detection", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces Detected: {face_count}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Masks Detected: {mask_count}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run_detection(self):
        """Main detection loop"""
        print("Starting Face Mask Detection...")
        print("This version works with default camera settings")
        print("Press 'q' to quit")
        
        frame_count = 0
        
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print(f"Failed to capture frame (attempt {frame_count})")
                frame_count += 1
                if frame_count > 10:
                    print("Too many failed attempts. Stopping.")
                    break
                continue
            
            frame_count = 0  # Reset counter on successful capture
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detect_faces(gray)
            
            face_count = len(faces)
            mask_count = 0
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y + h, x:x + w]
                
                # Detect mask
                prediction, confidence = self.simple_mask_detection(face_roi)
                
                # Count masks
                if prediction == "Mask":
                    mask_count += 1
                
                # Draw annotations
                self.draw_face_annotation(frame, x, y, w, h, prediction, confidence)
            
            # Add information panel
            self.add_info_panel(frame, face_count, mask_count)
            
            # Display frame
            cv2.imshow('Face Mask Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function"""
    try:
        detector = WorkingMaskDetector()
        detector.run_detection()
    except Exception as e:
        print(f"Error running detection: {e}")

if __name__ == "__main__":
    main()

