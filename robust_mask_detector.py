import cv2
import numpy as np
import os
import time

class RobustMaskDetector:
    def __init__(self):
        # Initialize face detection using multiple cascades for better accuracy
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        
        # Initialize camera with index 1 (which was found to be working)
        self.cap = cv2.VideoCapture(1)
        
        # Colors
        self.MASK_COLOR = (0, 0, 255)  # Red for mask
        self.NO_MASK_COLOR = (0, 255, 0)  # Green for no mask
        self.TEXT_COLOR = (255, 255, 255)  # White text
        
        # Enhanced detection parameters
        self.mask_threshold = 0.25  # Lowered threshold for better sensitivity
        self.texture_threshold = 35  # Adjusted texture threshold
        self.confidence_threshold = 0.6  # Minimum confidence for mask detection
        
        # Camera retry parameters
        self.max_retries = 5
        self.retry_delay = 0.1
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            print("Error: Could not open camera at index 1")
            return
        
        print("Robust Face Mask Detection Started!")
        print("Camera opened successfully at index 1!")
        
    def detect_faces(self, gray):
        """Detect faces using multiple cascades for better accuracy"""
        faces1 = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # More sensitive
            minNeighbors=4,    # Reduced for better detection
            minSize=(40, 40)   # Larger minimum size
        )
        
        faces2 = self.face_cascade_alt.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(40, 40)
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
    
    def enhanced_mask_detection(self, face_img):
        """Enhanced mask detection using multiple analysis methods"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Get face dimensions
            height, width = face_img.shape[:2]
            
            # Focus on lower 60% of face (where masks typically are)
            lower_face_start = int(height * 0.4)
            lower_face = face_img[lower_face_start:, :]
            lower_face_hsv = hsv[lower_face_start:, :]
            lower_face_gray = gray[lower_face_start:, :]
            
            # Method 1: Enhanced Color Analysis
            color_score = self.analyze_mask_colors(lower_face_hsv)
            
            # Method 2: Texture Analysis
            texture_score = self.analyze_texture(lower_face_gray)
            
            # Method 3: Edge Analysis
            edge_score = self.analyze_edges(lower_face_gray)
            
            # Method 4: Contour Analysis
            contour_score = self.analyze_contours(lower_face_gray)
            
            # Method 5: Skin Tone Analysis (masks cover skin)
            skin_score = self.analyze_skin_coverage(lower_face_hsv)
            
            # Combine all scores with weights
            final_score = (
                color_score * 0.35 +
                texture_score * 0.25 +
                edge_score * 0.20 +
                contour_score * 0.15 +
                skin_score * 0.05
            )
            
            # Decision logic with confidence
            if final_score > self.confidence_threshold:
                return "Mask", final_score
            elif final_score > 0.4:
                return "Unknown", final_score
            else:
                return "No Mask", 1 - final_score
                
        except Exception as e:
            print(f"Error in enhanced mask detection: {e}")
            return "Unknown", 0.0
    
    def analyze_mask_colors(self, hsv_img):
        """Enhanced color analysis for mask detection"""
        # White mask detection (more precise)
        white_lower = np.array([0, 0, 180])
        white_upper = np.array([180, 25, 255])
        white_mask = cv2.inRange(hsv_img, white_lower, white_upper)
        
        # Blue mask detection (medical masks)
        blue_lower = np.array([100, 40, 40])
        blue_upper = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv_img, blue_lower, blue_upper)
        
        # Black mask detection
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 60])
        black_mask = cv2.inRange(hsv_img, black_lower, black_upper)
        
        # Green mask detection (some medical masks)
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv_img, green_lower, green_upper)
        
        # Combine all mask detections
        combined_mask = white_mask | blue_mask | black_mask | green_mask
        
        # Calculate coverage
        mask_pixels = cv2.countNonZero(combined_mask)
        total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
        coverage = mask_pixels / total_pixels if total_pixels > 0 else 0
        
        return min(coverage * 2, 1.0)  # Boost the score
    
    def analyze_texture(self, gray_img):
        """Analyze texture patterns (masks have different texture than skin)"""
        # Calculate local standard deviation (texture measure)
        kernel_size = 5
        mean_img = cv2.blur(gray_img, (kernel_size, kernel_size))
        diff_img = cv2.absdiff(gray_img, mean_img)
        texture_variance = np.std(diff_img)
        
        # Normalize texture score (lower variance = more likely mask)
        normalized_variance = texture_variance / 255.0
        texture_score = max(0, 1 - normalized_variance * 2)
        
        return texture_score
    
    def analyze_edges(self, gray_img):
        """Analyze edge patterns (masks have different edge characteristics)"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        
        # Detect edges
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate edge density
        edge_pixels = cv2.countNonZero(edges)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_density = edge_pixels / total_pixels if total_pixels > 0 else 0
        
        # Masks typically have fewer edges than skin
        edge_score = max(0, 1 - edge_density * 10)
        
        return edge_score
    
    def analyze_contours(self, gray_img):
        """Analyze contour patterns"""
        # Apply threshold to create binary image
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return 0.5
        
        # Analyze contour characteristics
        total_area = gray_img.shape[0] * gray_img.shape[1]
        contour_areas = [cv2.contourArea(c) for c in contours]
        max_contour_area = max(contour_areas) if contour_areas else 0
        
        # Large, simple contours suggest mask
        contour_score = min(max_contour_area / total_area * 3, 1.0)
        
        return contour_score
    
    def analyze_skin_coverage(self, hsv_img):
        """Analyze skin tone coverage (masks cover skin)"""
        # Define skin color range in HSV
        skin_lower = np.array([0, 20, 70])
        skin_upper = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv_img, skin_lower, skin_upper)
        
        # Calculate skin coverage
        skin_pixels = cv2.countNonZero(skin_mask)
        total_pixels = skin_mask.shape[0] * skin_mask.shape[1]
        skin_coverage = skin_pixels / total_pixels if total_pixels > 0 else 0
        
        # Lower skin coverage suggests mask
        skin_score = max(0, 1 - skin_coverage)
        
        return skin_score
    
    def draw_triangle(self, frame, x, y, w, h, color):
        """Draw a red triangle on the face when mask is detected"""
        # Calculate triangle points
        center_x = x + w // 2
        top_y = y - 15
        left_x = center_x - 25
        right_x = center_x + 25
        bottom_y = y + 35
        
        # Draw filled triangle
        triangle_points = np.array([[center_x, top_y], [left_x, bottom_y], [right_x, bottom_y]], np.int32)
        cv2.fillPoly(frame, [triangle_points], color)
        
        # Draw triangle outline
        cv2.polylines(frame, [triangle_points], True, (0, 0, 0), 2)
    
    def draw_face_annotation(self, frame, x, y, w, h, prediction, confidence):
        """Draw face detection box and annotations"""
        # Determine color based on prediction and confidence
        if prediction == "Mask":
            color = self.MASK_COLOR
            label = f"Mask Detected ({confidence:.2f})"
        elif prediction == "No Mask":
            color = self.NO_MASK_COLOR
            label = f"No Mask ({confidence:.2f})"
        else:
            color = (255, 165, 0)  # Orange for unknown
            label = f"Unknown ({confidence:.2f})"
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw triangle for mask detection
        if prediction == "Mask":
            self.draw_triangle(frame, x, y, w, h, self.MASK_COLOR)
        
        # Draw label with background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y - 35), (x + label_size[0], y), color, -1)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 2)
    
    def add_info_panel(self, frame, face_count, mask_count, status="Running"):
        """Add information panel to the frame"""
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text information
        cv2.putText(frame, "Robust Face Mask Detection", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces Detected: {face_count}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Masks Detected: {mask_count}", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (20, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def capture_frame_with_retry(self):
        """Capture frame with retry logic"""
        for attempt in range(self.max_retries):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return True, frame
            else:
                print(f"Frame capture attempt {attempt + 1} failed, retrying...")
                time.sleep(self.retry_delay)
        
        return False, None
    
    def run_detection(self):
        """Main detection loop with robust error handling"""
        print("Starting Robust Face Mask Detection...")
        print("Using camera index 1 (working camera)")
        print("Enhanced accuracy with multiple detection methods")
        print("Robust error handling for camera issues")
        print("Press 'q' to quit")
        
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while True:
            # Capture frame with retry logic
            success, frame = self.capture_frame_with_retry()
            
            if not success:
                consecutive_failures += 1
                print(f"Failed to capture frame (consecutive failures: {consecutive_failures})")
                
                if consecutive_failures >= max_consecutive_failures:
                    print("Too many consecutive failures. Stopping.")
                    break
                
                # Show error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "Camera Error - Retrying...", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Robust Face Mask Detection', error_frame)
                cv2.waitKey(1000)  # Wait 1 second
                continue
            
            consecutive_failures = 0  # Reset on successful capture
            
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
                
                # Detect mask using enhanced method
                prediction, confidence = self.enhanced_mask_detection(face_roi)
                
                # Count masks
                if prediction == "Mask":
                    mask_count += 1
                
                # Draw annotations
                self.draw_face_annotation(frame, x, y, w, h, prediction, confidence)
            
            # Add information panel
            self.add_info_panel(frame, face_count, mask_count, "Running")
            
            # Display frame
            cv2.imshow('Robust Face Mask Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function"""
    try:
        detector = RobustMaskDetector()
        detector.run_detection()
    except Exception as e:
        print(f"Error running detection: {e}")

if __name__ == "__main__":
    main()

