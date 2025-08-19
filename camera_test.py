import cv2
import numpy as np

def test_cameras():
    """Test different camera indices to find a working one"""
    print("Testing camera access...")
    
    # Try different camera indices
    for camera_index in range(5):  # Try indices 0-4
        print(f"\nTrying camera index {camera_index}...")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Camera {camera_index} is not available")
            cap.release()
            continue
        
        # Try to read a frame
        ret, frame = cap.read()
        
        if ret:
            print(f"✓ Camera {camera_index} is working!")
            print(f"Frame size: {frame.shape}")
            
            # Show the frame briefly
            cv2.imshow(f'Camera {camera_index} Test', frame)
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyAllWindows()
            
            cap.release()
            return camera_index
        else:
            print(f"✗ Camera {camera_index} failed to capture frame")
            cap.release()
    
    print("\nNo working camera found!")
    return None

def test_camera_with_settings(camera_index):
    """Test camera with different settings"""
    print(f"\nTesting camera {camera_index} with different settings...")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("Camera not accessible")
        return False
    
    # Try different resolutions
    resolutions = [
        (640, 480),
        (320, 240),
        (1280, 720),
        (800, 600)
    ]
    
    for width, height in resolutions:
        print(f"Trying resolution {width}x{height}...")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        ret, frame = cap.read()
        
        if ret:
            print(f"✓ Resolution {width}x{height} works!")
            print(f"Actual frame size: {frame.shape}")
            
            # Show the frame
            cv2.imshow(f'Camera {camera_index} - {width}x{height}', frame)
            cv2.waitKey(3000)  # Show for 3 seconds
            cv2.destroyAllWindows()
            
            cap.release()
            return True
        else:
            print(f"✗ Resolution {width}x{height} failed")
    
    cap.release()
    return False

def main():
    """Main function to test cameras"""
    print("=" * 50)
    print("CAMERA TEST UTILITY")
    print("=" * 50)
    
    # Test for available cameras
    working_camera = test_cameras()
    
    if working_camera is not None:
        print(f"\nFound working camera at index {working_camera}")
        
        # Test with different settings
        if test_camera_with_settings(working_camera):
            print(f"\n✓ Camera {working_camera} is fully functional!")
            print(f"Use camera index {working_camera} in your face mask detection scripts")
        else:
            print(f"\n✗ Camera {working_camera} has issues with settings")
    else:
        print("\nNo working camera found. Possible issues:")
        print("1. Camera is not connected")
        print("2. Camera is being used by another application")
        print("3. Camera drivers are not installed")
        print("4. Camera permissions are not granted")
        print("\nTry:")
        print("- Closing other applications that might use the camera")
        print("- Restarting your computer")
        print("- Checking camera permissions in Windows Settings")

if __name__ == "__main__":
    main()
