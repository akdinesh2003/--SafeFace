import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class AdvancedFaceMaskDetector:
    def __init__(self):
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Colors
        self.MASK_COLOR = (0, 0, 255)  # Red
        self.NO_MASK_COLOR = (0, 255, 0)  # Green
        self.UNKNOWN_COLOR = (255, 165, 0)  # Orange
        self.TEXT_COLOR = (255, 255, 255)  # White
        
        # Load or create model
        self.model = self.load_or_create_model()
        
    def create_advanced_model(self):
        """Create an advanced CNN model for mask detection"""
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Flatten and Dense Layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(2, activation='softmax')  # 2 classes: mask, no_mask
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_or_create_model(self):
        """Load existing model or create new one"""
        model_path = 'advanced_mask_model.h5'
        
        if os.path.exists(model_path):
            try:
                print("Loading existing model...")
                return tf.keras.models.load_model(model_path)
            except:
                print("Error loading model, creating new one...")
        
        print("Creating new advanced model...")
        model = self.create_advanced_model()
        model.save(model_path)
        return model
    
    def preprocess_face(self, face_img):
        """Preprocess face image for prediction"""
        try:
            # Resize to 128x128 for better accuracy
            face_img = cv2.resize(face_img, (128, 128))
            
            # Convert BGR to RGB
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Normalize
            face_img = face_img.astype('float32') / 255.0
            
            # Add batch dimension
            face_img = np.expand_dims(face_img, axis=0)
            
            return face_img
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None
    
    def predict_mask(self, face_img):
        """Predict mask status with confidence"""
        try:
            processed_face = self.preprocess_face(face_img)
            if processed_face is None:
                return "Unknown", 0.0
            
            # Make prediction
            prediction = self.model.predict(processed_face, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Determine result based on confidence threshold
            if confidence > 0.7:
                result = "Mask" if predicted_class == 0 else "No Mask"
            elif confidence > 0.5:
                result = "Unknown"
            else:
                result = "Unknown"
            
            return result, confidence
            
        except Exception as e:
            print(f"Error predicting mask: {e}")
            return "Unknown", 0.0
    
    def draw_advanced_triangle(self, frame, x, y, w, h, color, confidence):
        """Draw an advanced triangle with confidence indicator"""
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
        
        # Draw confidence bar
        bar_width = 50
        bar_height = 5
        bar_x = center_x - bar_width // 2
        bar_y = bottom_y + 10
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # Confidence level bar
        confidence_width = int(bar_width * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), (255, 255, 255), -1)
    
    def draw_face_annotation(self, frame, x, y, w, h, prediction, confidence):
        """Draw comprehensive face annotation"""
        # Determine color based on prediction
        if prediction == "Mask":
            color = self.MASK_COLOR
            label = f"Mask Detected ({confidence:.2f})"
        elif prediction == "No Mask":
            color = self.NO_MASK_COLOR
            label = f"No Mask ({confidence:.2f})"
        else:
            color = self.UNKNOWN_COLOR
            label = f"Unknown ({confidence:.2f})"
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw triangle for mask detection
        if prediction == "Mask":
            self.draw_advanced_triangle(frame, x, y, w, h, color, confidence)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y - 30), (x + label_size[0], y), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 2)
    
    def add_status_info(self, frame, face_count, mask_count):
        """Add status information to frame"""
        # Status background
        cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0, 128), -1)
        
        # Status text
        cv2.putText(frame, f"Faces Detected: {face_count}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Masks Detected: {mask_count}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run_detection(self):
        """Main detection loop with enhanced features"""
        print("Starting Advanced Face Mask Detection...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with improved parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=6,
                minSize=(40, 40),
                maxSize=(300, 300)
            )
            
            face_count = len(faces)
            mask_count = 0
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y + h, x:x + w]
                
                # Predict mask
                prediction, confidence = self.predict_mask(face_roi)
                
                # Count masks
                if prediction == "Mask":
                    mask_count += 1
                
                # Draw annotations
                self.draw_face_annotation(frame, x, y, w, h, prediction, confidence)
            
            # Add status information
            self.add_status_info(frame, face_count, mask_count)
            
            # Display frame
            cv2.imshow('Advanced Face Mask Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function"""
    try:
        detector = AdvancedFaceMaskDetector()
        detector.run_detection()
    except Exception as e:
        print(f"Error running detection: {e}")

if __name__ == "__main__":
    main()
