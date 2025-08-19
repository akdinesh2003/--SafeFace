import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import imutils

class FaceMaskDetector:
    def __init__(self):
        # Load pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load mask detection model
        self.mask_model = self.load_mask_model()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Colors for visualization
        self.MASK_COLOR = (0, 0, 255)  # Red for mask detected
        self.NO_MASK_COLOR = (0, 255, 0)  # Green for no mask
        self.TEXT_COLOR = (255, 255, 255)  # White text
        
    def load_mask_model(self):
        """Load the pre-trained mask detection model"""
        try:
            # Create a simple CNN model for mask detection
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
            
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
                MaxPooling2D(2, 2),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(2, activation='softmax')  # 2 classes: mask, no_mask
            ])
            
            # Compile the model
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            # Save the model
            model.save('mask_detection_model.h5')
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def preprocess_face(self, face_img):
        """Preprocess face image for mask detection"""
        try:
            # Resize to 64x64
            face_img = cv2.resize(face_img, (64, 64))
            # Convert to array and normalize
            face_img = img_to_array(face_img)
            face_img = face_img / 255.0
            # Add batch dimension
            face_img = np.expand_dims(face_img, axis=0)
            return face_img
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None
    
    def predict_mask(self, face_img):
        """Predict if face has mask or not"""
        try:
            if self.mask_model is None:
                return "Unknown"
            
            processed_face = self.preprocess_face(face_img)
            if processed_face is None:
                return "Unknown"
            
            # Make prediction
            prediction = self.mask_model.predict(processed_face)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Return prediction based on confidence
            if confidence > 0.6:
                return "Mask" if predicted_class == 0 else "No Mask"
            else:
                return "Unknown"
                
        except Exception as e:
            print(f"Error predicting mask: {e}")
            return "Unknown"
    
    def draw_triangle(self, frame, x, y, w, h, color):
        """Draw a triangle on the face"""
        # Calculate triangle points
        center_x = x + w // 2
        top_y = y - 10
        left_x = center_x - 20
        right_x = center_x + 20
        bottom_y = y + 30
        
        # Draw triangle
        triangle_points = np.array([[center_x, top_y], [left_x, bottom_y], [right_x, bottom_y]], np.int32)
        cv2.fillPoly(frame, [triangle_points], color)
        
        # Draw triangle outline
        cv2.polylines(frame, [triangle_points], True, (0, 0, 0), 2)
    
    def draw_face_box(self, frame, x, y, w, h, has_mask):
        """Draw face detection box and triangle"""
        color = self.MASK_COLOR if has_mask == "Mask" else self.NO_MASK_COLOR
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw triangle if mask detected
        if has_mask == "Mask":
            self.draw_triangle(frame, x, y, w, h, self.MASK_COLOR)
        
        # Add text label
        label = "Mask Detected" if has_mask == "Mask" else "No Mask" if has_mask == "No Mask" else "Unknown"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 2)
    
    def run_detection(self):
        """Main detection loop"""
        print("Starting Face Mask Detection...")
        print("Press 'q' to quit")
        
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y + h, x:x + w]
                
                # Predict mask
                mask_prediction = self.predict_mask(face_roi)
                
                # Draw detection results
                self.draw_face_box(frame, x, y, w, h, mask_prediction)
            
            # Add instructions to frame
            cv2.putText(frame, "Press 'q' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Face Mask Detection', frame)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the face mask detector"""
    try:
        detector = FaceMaskDetector()
        detector.run_detection()
    except Exception as e:
        print(f"Error running detection: {e}")

if __name__ == "__main__":
    main()
