import cv2
import numpy as np
import time

def detect_cameras():
    """Detect available cameras including Bluetooth cameras"""
    available_cameras = []
    
    print("🔍 Scanning for available cameras...")
    
    # Test camera indices 0-10
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            # Get camera properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Try to capture a test frame
            ret, frame = cap.read()
            if ret and frame is not None:
                camera_info = {
                    'index': i,
                    'width': int(width),
                    'height': int(height),
                    'fps': fps,
                    'working': True
                }
                available_cameras.append(camera_info)
                print(f"✅ Camera {i}: {width}x{height} @ {fps}fps")
            else:
                print(f"⚠️  Camera {i}: Opened but no frame capture")
            
            cap.release()
        else:
            print(f"❌ Camera {i}: Not available")
    
    # Also try Media Foundation backend for Bluetooth cameras
    print("\n🔍 Scanning for Bluetooth cameras (Media Foundation)...")
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                # Check if this is a different camera than DirectShow
                existing = any(cam['index'] == i for cam in available_cameras)
                if not existing:
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    camera_info = {
                        'index': i,
                        'width': int(width),
                        'height': int(height),
                        'fps': fps,
                        'working': True,
                        'backend': 'Media Foundation',
                        'bluetooth': True  # Likely Bluetooth camera
                    }
                    available_cameras.append(camera_info)
                    print(f"📱 Bluetooth Camera {i}: {width}x{height} @ {fps}fps (Media Foundation)")
                cap.release()
    
    return available_cameras

def select_camera(available_cameras):
    """Let user select a camera"""
    if not available_cameras:
        print("❌ No cameras found!")
        return None, None
    
    print(f"\n📹 Found {len(available_cameras)} camera(s):")
    for i, cam in enumerate(available_cameras):
        backend_info = f" ({cam.get('backend', 'DirectShow')})" if 'backend' in cam else ""
        bluetooth_info = " [Bluetooth]" if cam.get('bluetooth', False) else ""
        print(f"  {i+1}. Camera {cam['index']}: {cam['width']}x{cam['height']}{backend_info}{bluetooth_info}")
    
    while True:
        try:
            choice = input(f"\nSelect camera (1-{len(available_cameras)}) or press Enter for default (Camera 2): ").strip()
            
            if choice == "":
                # Default to camera 2 (which we know works)
                selected_cam = next((cam for cam in available_cameras if cam['index'] == 2), available_cameras[0])
                backend = cv2.CAP_DSHOW
                print(f"✅ Using default camera: Camera {selected_cam['index']}")
                return selected_cam, backend
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_cameras):
                selected_cam = available_cameras[choice_idx]
                backend = cv2.CAP_MSMF if selected_cam.get('bluetooth', False) else cv2.CAP_DSHOW
                print(f"✅ Selected: Camera {selected_cam['index']}")
                return selected_cam, backend
            else:
                print("❌ Invalid choice. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number.")

def main():
    """Working Face Mask Detection System with Bluetooth Camera Support"""
    print("=== WORKING FACE MASK DETECTION WITH BLUETOOTH SUPPORT ===")
    
    # Detect available cameras
    available_cameras = detect_cameras()
    
    if not available_cameras:
        print("❌ No working cameras found. Please check your camera connections.")
        return
    
    # Let user select camera
    selected_camera, backend = select_camera(available_cameras)
    
    if selected_camera is None:
        return
    
    # Initialize camera with selected settings
    print(f"\n📹 Opening camera {selected_camera['index']} with {backend} backend...")
    cap = cv2.VideoCapture(selected_camera['index'], backend)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera {selected_camera['index']}")
        return
    
    print("✅ Camera opened successfully!")
    
    # Initialize face detection
    print("Loading face detection model...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("❌ Failed to load face detection model")
        return
    
    print("✅ Face detection model loaded!")
    
    # Wait for camera to initialize
    print("Initializing camera...")
    time.sleep(2)
    
    print("Starting Face Mask Detection...")
    print("Press 'q' to quit, 'c' to change camera")
    
    frame_count = 0
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        frame_count += 1
        
        if not ret or frame is None:
            print(f"Failed to capture frame {frame_count}")
            continue
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y + h, x:x + w]
            
            # Simple mask detection based on color analysis
            try:
                # Convert to HSV for color analysis
                hsv_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
                
                # Focus on lower part of face (where masks typically are)
                lower_face_start = int(h * 0.6)
                lower_face = hsv_roi[lower_face_start:, :]
                
                # Check for common mask colors
                # White mask detection
                white_mask = cv2.inRange(lower_face, np.array([0, 0, 180]), np.array([180, 30, 255]))
                
                # Blue mask detection (medical masks)
                blue_mask = cv2.inRange(lower_face, np.array([100, 50, 50]), np.array([130, 255, 255]))
                
                # Black mask detection
                black_mask = cv2.inRange(lower_face, np.array([0, 0, 0]), np.array([180, 255, 60]))
                
                # Calculate mask coverage
                total_pixels = lower_face.shape[0] * lower_face.shape[1]
                white_pixels = cv2.countNonZero(white_mask)
                blue_pixels = cv2.countNonZero(blue_mask)
                black_pixels = cv2.countNonZero(black_mask)
                
                mask_coverage = (white_pixels + blue_pixels + black_pixels) / total_pixels if total_pixels > 0 else 0
                
                # Determine if mask is detected
                if mask_coverage > 0.25:  # Threshold for mask detection
                    # Draw red rectangle and triangle for mask
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    
                    # Draw red triangle above face
                    center_x = x + w // 2
                    triangle_points = np.array([[center_x, y - 25], [center_x - 25, y + 25], [center_x + 25, y + 25]], np.int32)
                    cv2.fillPoly(frame, [triangle_points], (0, 0, 255))
                    cv2.putText(frame, "MASK DETECTED", (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Add confidence score
                    cv2.putText(frame, f"Confidence: {mask_coverage:.2f}", (x, y + h + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    # Draw green rectangle for no mask
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "NO MASK", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Add confidence score
                    cv2.putText(frame, f"Confidence: {1-mask_coverage:.2f}", (x, y + h + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
            except Exception as e:
                # If there's an error in mask detection, just draw a neutral box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 2)
                cv2.putText(frame, "ERROR", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # Add information panel
        cv2.rectangle(frame, (10, 10), (500, 140), (0, 0, 0), -1)
        cv2.putText(frame, "Face Mask Detection System", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Camera: {selected_camera['index']} ({backend})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces Detected: {len(faces)}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit, 'c' to change camera", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Face Mask Detection', frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("\n🔄 Changing camera...")
            cap.release()
            cv2.destroyAllWindows()
            
            # Re-select camera
            selected_camera, backend = select_camera(available_cameras)
            if selected_camera is None:
                return
            
            cap = cv2.VideoCapture(selected_camera['index'], backend)
            if not cap.isOpened():
                print(f"❌ Failed to open camera {selected_camera['index']}")
                return
            
            print(f"✅ Switched to camera {selected_camera['index']}")
            frame_count = 0
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"Detection completed. Total frames processed: {frame_count}")

if __name__ == "__main__":
    main()

