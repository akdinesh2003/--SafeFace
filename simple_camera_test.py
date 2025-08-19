import cv2
import numpy as np
import time

def test_camera():
    print("=== Simple Camera Test ===")
    print("Testing camera index 2 with DirectShow backend...")
    
    # Try to open camera
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return
    
    print("✅ Camera opened successfully")
    
    # Try to get camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera properties:")
    print(f"  Width: {width}")
    print(f"  Height: {height}")
    print(f"  FPS: {fps}")
    
    # Try to capture frames
    print("\nAttempting to capture frames...")
    frame_count = 0
    max_frames = 30  # Try for 30 frames
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        
        if ret and frame is not None:
            frame_count += 1
            print(f"✅ Frame {frame_count}: {frame.shape}")
            
            # Show the frame
            cv2.imshow('Camera Test', frame)
            
            # Wait for key press or 100ms
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        else:
            print(f"❌ Failed to capture frame {frame_count + 1}")
            frame_count += 1
            time.sleep(0.1)
    
    print(f"\nTest completed. Successfully captured {frame_count} frames.")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()

