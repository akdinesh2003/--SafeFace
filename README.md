# Face Mask Detection Project

This project implements real-time face mask detection using computer vision and machine learning techniques. The system can detect multiple people in a camera feed and identify whether they are wearing masks, displaying a red triangle on masked faces with "Mask Detected" labels.

## Features

- **Real-time Detection**: Live camera feed processing
- **Multi-face Detection**: Can detect and track multiple people simultaneously
- **Mask Classification**: Identifies whether a person is wearing a mask
- **Visual Indicators**: Red triangles on masked faces with confidence scores
- **Multiple Detection Methods**: Three different approaches for various use cases

## Project Structure

```
FD/
├── requirements.txt              # Python dependencies
├── face_mask_detection.py        # Basic face mask detection
├── advanced_mask_detector.py     # Advanced CNN-based detection
├── demo_mask_detector.py         # Demo version with simple detection
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- Webcam
- Windows 10/11 (tested on Windows 10)

### Setup

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Camera Access**:
   Make sure your webcam is connected and accessible.

## Usage

### Option 1: Demo Version (Recommended for Testing)
The demo version uses simple color and texture analysis for immediate testing:

```bash
python demo_mask_detector.py
```

### Option 2: Basic Detection
Basic face mask detection with simple CNN model:

```bash
python face_mask_detection.py
```

### Option 3: Advanced Detection
Advanced detection with sophisticated CNN architecture:

```bash
python advanced_mask_detector.py
```

## How It Works

### Face Detection
- Uses OpenCV's Haar Cascade classifiers
- Combines multiple cascade files for better accuracy
- Merges overlapping detections to avoid duplicates

### Mask Detection Methods

#### Demo Version (demo_mask_detector.py)
- **Color Analysis**: Detects common mask colors (white, blue, black)
- **Texture Analysis**: Analyzes pixel variance in lower face region
- **Rule-based**: Uses thresholds to classify mask vs no-mask

#### Basic Version (face_mask_detection.py)
- **Simple CNN**: 3-layer convolutional neural network
- **64x64 Input**: Resizes face images for processing
- **Binary Classification**: Mask vs No Mask

#### Advanced Version (advanced_mask_detector.py)
- **Deep CNN**: 6-layer architecture with batch normalization
- **128x128 Input**: Higher resolution for better accuracy
- **Confidence Scoring**: Provides confidence levels for predictions

### Visual Output

- **Green Rectangle**: No mask detected
- **Red Rectangle + Triangle**: Mask detected
- **Confidence Scores**: Displayed with each detection
- **Status Panel**: Shows total faces and masks detected

## Controls

- **'q'**: Quit the application
- **Camera**: Automatically uses default webcam (index 0)

## Technical Details

### Dependencies
- **OpenCV**: Computer vision and image processing
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computing
- **imutils**: Image processing utilities

### Model Architecture (Advanced Version)
```
Input (128x128x3)
├── Conv2D(32) + BatchNorm + ReLU
├── Conv2D(32) + BatchNorm + ReLU
├── MaxPool2D + Dropout(0.25)
├── Conv2D(64) + BatchNorm + ReLU
├── Conv2D(64) + BatchNorm + ReLU
├── MaxPool2D + Dropout(0.25)
├── Conv2D(128) + BatchNorm + ReLU
├── Conv2D(128) + BatchNorm + ReLU
├── MaxPool2D + Dropout(0.25)
├── Flatten
├── Dense(512) + BatchNorm + ReLU + Dropout(0.5)
├── Dense(256) + BatchNorm + ReLU + Dropout(0.5)
└── Dense(2, softmax)  # Output: Mask/No Mask
```

## Performance Notes

- **Demo Version**: Fastest, works immediately without training
- **Basic Version**: Moderate speed, requires model creation
- **Advanced Version**: Best accuracy, slower processing

## Troubleshooting

### Common Issues

1. **Camera Not Found**:
   - Check if webcam is connected
   - Try changing camera index in code (0, 1, 2...)

2. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

3. **Model Loading Errors**:
   - Delete existing model files to recreate them
   - Check TensorFlow version compatibility

4. **Performance Issues**:
   - Reduce camera resolution in code
   - Use demo version for better performance

### System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Multi-core processor recommended
- **GPU**: Optional, TensorFlow will use CPU by default

## Customization

### Adjusting Detection Sensitivity
In `demo_mask_detector.py`, modify:
```python
self.mask_threshold = 0.3  # Increase for stricter detection
```

### Changing Colors
Modify color constants in any script:
```python
self.MASK_COLOR = (0, 0, 255)      # Red for mask
self.NO_MASK_COLOR = (0, 255, 0)   # Green for no mask
```

### Camera Settings
Adjust camera properties:
```python
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

## Future Enhancements

- [ ] Training on custom dataset
- [ ] Support for different mask types
- [ ] Face recognition integration
- [ ] Alert system for no-mask detection
- [ ] Mobile app version
- [ ] Cloud-based processing

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the project.

---

**Note**: This is a demonstration project. For production use, consider training the models on a larger, more diverse dataset for better accuracy across different lighting conditions, mask types, and face orientations.

## Author

AK DINESH   https://github.com/akdinesh2003
