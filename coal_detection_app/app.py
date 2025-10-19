from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import cv2
import base64
from io import BytesIO
import json

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define image transformations
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model architectures (must match training)
def create_quality_model(num_classes=2):
    """Create coal quality model architecture"""
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

def create_defect_model(num_classes=2):
    """Create defect classification model architecture"""
    model = models.resnet50(pretrained=False)
    
    # Freeze early layers, fine-tune later layers
    for name, param in model.named_parameters():
        if 'layer3' in name or 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model

# Load models
class CoalDetectionSystem:
    def __init__(self):
        self.quality_model = None
        self.defect_model = None
        self.quality_class_names = ['Defect_coal', 'Good_quality_coal']
        self.defect_class_names = ['Cracks_and_fractures', 'Surface_deformation']
        self.load_models()
    
    def load_models(self):
        """Load both trained models"""
        try:
            # Load quality model
            quality_model_path = 'models/coal_quality_model.pth'
            if os.path.exists(quality_model_path):
                self.quality_model = create_quality_model(len(self.quality_class_names))
                self.quality_model.load_state_dict(torch.load(quality_model_path, map_location=device))
                self.quality_model.to(device)
                self.quality_model.eval()
                print("✅ Coal quality model loaded successfully")
            else:
                print(f"❌ Quality model not found at {quality_model_path}")
            
            # Load defect model
            defect_model_path = 'models/coal_defect_classification_best.pth'
            if os.path.exists(defect_model_path):
                self.defect_model = create_defect_model(len(self.defect_class_names))
                self.defect_model.load_state_dict(torch.load(defect_model_path, map_location=device))
                self.defect_model.to(device)
                self.defect_model.eval()
                print("✅ Defect classification model loaded successfully")
            else:
                print(f"❌ Defect model not found at {defect_model_path}")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def predict_quality(self, image):
        """Predict coal quality (good vs defective)"""
        if self.quality_model is None:
            return None, None, None
        
        # Preprocess image
        image_tensor = data_transforms(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.quality_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.quality_class_names[predicted.item()]
        confidence_score = confidence.item()
        all_probabilities = probabilities.cpu().numpy()[0]
        
        return predicted_class, confidence_score, all_probabilities
    
    def predict_defect(self, image):
        """Predict defect type"""
        if self.defect_model is None:
            return None, None, None
        
        # Preprocess image
        image_tensor = data_transforms(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.defect_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.defect_class_names[predicted.item()]
        confidence_score = confidence.item()
        all_probabilities = probabilities.cpu().numpy()[0]
        
        return predicted_class, confidence_score, all_probabilities
    
    def analyze_coal(self, image):
        """
        Complete coal analysis pipeline:
        1. First check if coal is good or defective
        2. If defective, classify the defect type
        """
        try:
            # Step 1: Quality detection
            quality_class, quality_confidence, quality_probs = self.predict_quality(image)
            
            if quality_class is None:
                return {
                    'error': 'Quality model not available'
                }
            
            result = {
                'quality_prediction': {
                    'class': quality_class,
                    'confidence': float(quality_confidence),
                    'probabilities': {
                        self.quality_class_names[i]: float(quality_probs[i]) 
                        for i in range(len(self.quality_class_names))
                    }
                },
                'defect_prediction': None
            }
            
            # Step 2: If defective, classify defect type
            if quality_class == 'Defect_coal' and self.defect_model is not None:
                defect_class, defect_confidence, defect_probs = self.predict_defect(image)
                
                if defect_class is not None:
                    result['defect_prediction'] = {
                        'class': defect_class,
                        'confidence': float(defect_confidence),
                        'probabilities': {
                            self.defect_class_names[i]: float(defect_probs[i]) 
                            for i in range(len(self.defect_class_names))
                        }
                    }
            
            return result
            
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}'
            }

# Initialize the coal detection system
coal_system = CoalDetectionSystem()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image):
    """Convert PIL image to base64 string for display in HTML"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

# Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for coal analysis"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Read and process image
            image = Image.open(file.stream).convert('RGB')
            
            # Analyze coal
            result = coal_system.analyze_coal(image)
            
            # Convert image to base64 for display
            image_base64 = image_to_base64(image)
            result['image'] = image_base64
            
            return jsonify(result)
        else:
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    quality_loaded = coal_system.quality_model is not None
    defect_loaded = coal_system.defect_model is not None
    
    status = 'healthy' if quality_loaded else 'models not fully loaded'
    
    return jsonify({
        'status': status,
        'quality_model_loaded': quality_loaded,
        'defect_model_loaded': defect_loaded,
        'device': str(device)
    })

@app.route('/model_status')
def model_status():
    """Model status endpoint"""
    quality_loaded = coal_system.quality_model is not None
    defect_loaded = coal_system.defect_model is not None
    
    if quality_loaded and defect_loaded:
        message = "جميع النماذج جاهزة للاستخدام"
        status = "success"
    elif quality_loaded and not defect_loaded:
        message = "نموذج الجودة جاهز، لكن نموذج العيوب غير متوفر"
        status = "warning"
    else:
        message = "النماذج غير متوفرة، يرجى التحقق من الملفات"
        status = "error"
    
    return jsonify({
        'message': message,
        'status': status,
        'quality_loaded': quality_loaded,
        'defect_loaded': defect_loaded
    })

if __name__ == '__main__':
    print("🚀 Starting Coal Detection API...")
    print("📊 Available endpoints:")
    print("   • GET  / - Web interface")
    print("   • POST /predict - API endpoint for predictions")
    print("   • GET  /health - Health check")
    print("   • GET  /model_status - Model status")
    
    app.run(debug=True, host='0.0.0.0', port=5000)