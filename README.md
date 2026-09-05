# BreedAI - AI-Powered Cattle Breed Identification System

## Overview
BreedAI is a web-based application that uses deep learning to identify cattle and buffalo breeds from images. The system employs a MobileNetV2 architecture fine-tuned on Indian cattle breeds, enabling accurate breed classification through a user-friendly web interface. This tool is designed to assist field workers, veterinarians, and livestock professionals in accurate breed identification for national livestock programs.

## Table of Contents
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Performance Metrics](#performance-metrics)
- [File Structure](#file-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Features
- Real-time breed identification from uploaded images
- Support for 50 Indian cattle and buffalo breeds
- Top-3 breed predictions with confidence scores
- Visual confidence indicators with progress bars
- Camera capture and gallery upload support
- Mobile-responsive dark theme interface
- Client-side inference using ONNX Runtime Web
- Privacy-focused (images processed locally, not stored)

## Technical Architecture

### Model Training Pipeline
- **Dataset**: 50 Indian cattle breeds with 8,449 labeled images
- **Architecture**: MobileNetV2 (3.5M parameters)
- **Transfer Learning**: ImageNet pretrained weights
- **Training Setup**:
  - Optimizer: AdamW (discriminative learning rates)
  - Loss: CrossEntropyLoss
  - Scheduler: ReduceLROnPlateau
  - Batch Size: 32
  - Epochs: 15

### Deployment Flow
```text
PyTorch Model (.pth) → ONNX Export → ONNX Runtime Web → Web Application
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- ONNX Runtime
- Web browser with WebAssembly support

### Model Training Environment
```bash
# Clone the repository
git clone (https://github.com/jassu-dev/BreedAI.git)
cd breedai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scriptsctivate

# Install dependencies
pip install torch torchvision
pip install onnx onnxruntime
pip install numpy matplotlib tqdm scikit-learn
```

### Web Application Setup
```bash
# Install web server (optional)
npm install -g http-server

# Or use Python's built-in server
cd website
python -m http.server 8000
```

## Usage

### Running the Web Application
1. Place the ONNX model and labels file in the `assets` directory:
```text
website/
├── index.html
├── assets/
│   ├── model.onnx
│   └── labels.txt
```
2. Start the web server:
```bash
cd website
python -m http.server 8000
```
3. Open your browser and navigate to `http://localhost:8000`
4. Upload an image by clicking "Take Photo" or "Choose from Gallery"
5. Click "Analyze Breed" to get breed predictions with confidence scores

### Using the Model Programmatically
```python
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn

# Load model
checkpoint = torch.load('best_model.pth', map_location='cpu')
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
num_classes = len(checkpoint['class_names'])

model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, num_classes)
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('sample_image.jpg')
tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    outputs = model(tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    top3_probs, top3_indices = torch.topk(probabilities, 3)
```

## Model Training

### Dataset
- **Source**: Indian Cattle Breed Dataset (Kaggle)
- **Classes**: 50 Indian cattle and buffalo breeds
- **Split**: 70% training, 15% validation, 15% test
- **Images**: 8,449 total (5,891 train, 1,266 val, 1,292 test)

### Training Configuration
```python
# Model Architecture
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, num_classes)
)

# Optimizer
optimizer = optim.AdamW([
    {'params': feature_params, 'lr': 1e-4},
    {'params': classifier_params, 'lr': 1e-3}
], weight_decay=1e-4)

# Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)
```

## Performance Results

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | 87.93% |
| Test Accuracy | 83.78% |
| Training Accuracy | 90.07% |
| Number of Parameters | 3.5M |

### Class Distribution
The model is trained to identify the following 50 breeds:
1. Amritmahal, 2. Ayrshire, 3. Bargur, 4. Dangi, 5. Deoni, 6. Gir, 7. Hallikar, 8. Hariana, 9. Himachali Pahari, 10. Kangayam, 11. Kankrej, 12. Kenkatha, 13. Khariar, 14. Khillari, 15. Konkan Kapila, 16. Kosali, 17. Krishna Valley, 18. Ladakhi, 19. Lakhimi, 20. Malnad Gidda, 21. Mewati, 22. Nari, 23. Nimari, 24. Ongole, 25. Poda Thirupu, 26. Pulikulam, 27. Punganur, 28. Purnea, 29. Rathi, 30. Red Kandhari, 31. Red Sindhi, 32. Sahiwal, 33. Shweta Kapila, 34. Tharparkar, 35. Umblachery, 36. Vechur, 37. Bachaur, 38. Badri, 39. Bhelai, 40. Dagri, 41. Gangatari, 42. Gaolao, 43. Ghumsari, 44. Kherigarh, 45. Malvi, 46. Motu, 47. Nagori, 48. Ponwar, 49. Siri, 50. Thutho

## Web Application

### User Interface
The web interface features:
- Clean dark theme design
- Full-screen farm background image with opacity
- Green color scheme for intuitive interaction
- Mobile-responsive layout
- Two upload options: camera capture and gallery selection
- Real-time breed identification results
- Confidence score visualization with progress bars

### ONNX Runtime Integration
The application uses ONNX Runtime Web for client-side inference:
- WASM backend for browser compatibility
- No server-side processing required
- Local image processing ensures data privacy
- Fast inference with WebGL acceleration support

## File Structure
```text
breedai/
├── model.ipynb                 # Training notebook
├── best_model.pth              # Trained PyTorch model
├── model.onnx                  # Exported ONNX model
├── labels.txt                  # Breed names list
├── website/
│   ├── index.html              # Single HTML application
│   └── assets/
│       ├── model.onnx          # ONNX model for web
│       └── labels.txt          # Labels for web
└── processed_data/             # Dataset directory
    ├── train/                  # Training images (70%)
    ├── val/                    # Validation images (15%)
    └── test/                   # Test images (15%)
```

## Technologies Used

### Model Development
- PyTorch: Deep learning framework
- MobileNetV2: Lightweight CNN architecture
- Transfer Learning: ImageNet pretrained weights
- ONNX: Model interoperability format

### Web Application
- HTML5: Structure
- CSS3: Styling with custom properties
- JavaScript: Application logic
- ONNX Runtime Web: Client-side inference
- Google Fonts: Typography

## Performance Optimization

### Model Optimization
- ONNX format reduces model size
- WebAssembly execution for near-native speed
- Efficient preprocessing pipeline

### Web Optimization
- Single-file architecture
- Lazy loading of ONNX Runtime
- Minimal external dependencies
- Progressive loading indicator

## Contributing
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Test thoroughly
5. Submit a pull request

## License
This project is licensed under the MIT License.

## Acknowledgments
- Indian Cattle Dataset contributors
- ONNX Runtime team
- PyTorch community
- Kaggle platform for dataset hosting

## Contact
For questions or support, please contact the development team.
