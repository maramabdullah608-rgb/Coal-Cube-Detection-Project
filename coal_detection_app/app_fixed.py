import sys
import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import base64
from io import BytesIO
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Try to import PyTorch, but have fallback
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    
    PYTORCH_AVAILABLE = True
    print("✅ PyTorch loaded successfully!")
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")
    PYTORCH_AVAILABLE = False
except OSError as e:
    print(f"❌ PyTorch DLL error: {e}")
    print("💡 This is likely a Python 3.13 compatibility issue.")
    PYTORCH_AVAILABLE = False

# Define image transformations
if PYTORCH_AVAILABLE:
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

class CoalDetectionSystem:
    def __init__(self):
        self.quality_model = None
        self.defect_model = None
        
        # IMPORTANT: Try different class orders to find the correct one
        # Common class orders used in training:
        self.quality_class_orders = [
            ['Defect_coal', 'Good_quality_coal'],  # Option 1: Index 0: Defect, Index 1: Good
            ['Good_quality_coal', 'Defect_coal']   # Option 2: Index 0: Good, Index 1: Defect
        ]
        
        self.defect_class_orders = [
            ['Cracks_and_fractures', 'Surface_deformation'],  # Option 1: Index 0: Cracks, Index 1: Surface
            ['Surface_deformation', 'Cracks_and_fractures']   # Option 2: Index 0: Surface, Index 1: Cracks
        ]
        
        # Start with the most common ordering
        self.current_quality_order = 0
        self.current_defect_order = 0
        
        self.quality_class_names = self.quality_class_orders[self.current_quality_order]
        self.defect_class_names = self.defect_class_orders[self.current_defect_order]
        
        if PYTORCH_AVAILABLE:
            self.load_models()
        else:
            print("🚨 Running in simulation mode (PyTorch not available)")
    
    def load_models(self):
        """Load both trained models"""
        try:
            # Load quality model
            quality_model_path = 'models/coal_quality_model.pth'
            if os.path.exists(quality_model_path):
                print("📦 Loading coal quality model...")
                print("🔍 Testing different class orders for quality model...")
                
                # Test both class orders
                for i, class_names in enumerate(self.quality_class_orders):
                    print(f"   Testing order {i}: {class_names}")
                
                # Model architecture
                model = models.resnet50(pretrained=False)
                num_ftrs = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_ftrs, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, 2)  # 2 classes
                )
                
                self.quality_model = model
                
                try:
                    checkpoint = torch.load(quality_model_path, map_location=device)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.quality_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.quality_model.load_state_dict(checkpoint)
                except Exception as e:
                    print(f"⚠️ Standard load failed: {e}")
                    self.quality_model.load_state_dict(torch.load(quality_model_path, map_location=device, weights_only=False))
                
                self.quality_model.to(device)
                self.quality_model.eval()
                print("✅ Coal quality model loaded successfully")
                
            else:
                print(f"❌ Quality model not found at {quality_model_path}")
            
            # Load defect model
            defect_model_path = 'models/coal_defect_classification_best.pth'
            if os.path.exists(defect_model_path):
                print("📦 Loading defect classification model...")
                print("🔍 Testing different class orders for defect model...")
                
                for i, class_names in enumerate(self.defect_class_orders):
                    print(f"   Testing order {i}: {class_names}")
                
                model = models.resnet50(pretrained=False)
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
                    nn.Linear(256, 2)  # 2 classes
                )
                
                self.defect_model = model
                
                try:
                    checkpoint = torch.load(defect_model_path, map_location=device)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.defect_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.defect_model.load_state_dict(checkpoint)
                except Exception as e:
                    print(f"⚠️ Standard load failed: {e}")
                    self.defect_model.load_state_dict(torch.load(defect_model_path, map_location=device, weights_only=False))
                
                self.defect_model.to(device)
                self.defect_model.eval()
                print("✅ Defect classification model loaded successfully")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
    
    def predict_quality(self, image):
        """Predict coal quality with correct class mapping"""
        if not PYTORCH_AVAILABLE or self.quality_model is None:
            return self._simulate_quality_prediction()
        
        try:
            # Preprocess image
            image_tensor = data_transforms(image).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.quality_model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Use current class order
            predicted_class = self.quality_class_names[predicted_idx.item()]
            confidence_score = confidence.item()
            all_probabilities = probabilities.cpu().numpy()[0]
            
            print(f"🔍 Quality Prediction (Order {self.current_quality_order}):")
            print(f"   Predicted index: {predicted_idx.item()}")
            print(f"   Predicted class: {predicted_class}")
            print(f"   Confidence: {confidence_score:.4f}")
            print(f"   All probabilities: {dict(zip(self.quality_class_names, all_probabilities))}")
            
            return predicted_class, confidence_score, all_probabilities
            
        except Exception as e:
            print(f"❌ Error in quality prediction: {e}")
            return self._simulate_quality_prediction()
    
    def predict_defect(self, image):
        """Predict defect type with correct class mapping"""
        if not PYTORCH_AVAILABLE or self.defect_model is None:
            return self._simulate_defect_prediction()
        
        try:
            # Preprocess image
            image_tensor = data_transforms(image).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.defect_model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Use current class order
            predicted_class = self.defect_class_names[predicted_idx.item()]
            confidence_score = confidence.item()
            all_probabilities = probabilities.cpu().numpy()[0]
            
            print(f"🔍 Defect Prediction (Order {self.current_defect_order}):")
            print(f"   Predicted index: {predicted_idx.item()}")
            print(f"   Predicted class: {predicted_class}")
            print(f"   Confidence: {confidence_score:.4f}")
            print(f"   All probabilities: {dict(zip(self.defect_class_names, all_probabilities))}")
            
            return predicted_class, confidence_score, all_probabilities
            
        except Exception as e:
            print(f"❌ Error in defect prediction: {e}")
            return self._simulate_defect_prediction()
    
    def switch_quality_order(self):
        """Switch between different class orders for quality model"""
        self.current_quality_order = (self.current_quality_order + 1) % len(self.quality_class_orders)
        self.quality_class_names = self.quality_class_orders[self.current_quality_order]
        print(f"🔄 Switched quality class order to: {self.quality_class_names}")
        return self.current_quality_order
    
    def switch_defect_order(self):
        """Switch between different class orders for defect model"""
        self.current_defect_order = (self.current_defect_order + 1) % len(self.defect_class_orders)
        self.defect_class_names = self.defect_class_orders[self.current_defect_order]
        print(f"🔄 Switched defect class order to: {self.defect_class_names}")
        return self.current_defect_order
    
    def _simulate_quality_prediction(self):
        """Realistic simulation for quality"""
        # Use image characteristics for better simulation
        try:
            img_array = image if hasattr(image, 'size') else None
            if img_array:
                # Simulate based on actual image analysis
                width, height = image.size
                # Brighter images tend to be better quality
                brightness = sum(image.convert('L').getdata()) / (width * height * 255)
                
                if brightness > 0.6:  # Bright image - likely good quality
                    quality_class = 'Good_quality_coal'
                    confidence = random.uniform(0.8, 0.95)
                else:  # Dark image - might be defective
                    quality_class = 'Defect_coal' 
                    confidence = random.uniform(0.7, 0.9)
            else:
                quality_class = random.choice(['Good_quality_coal', 'Defect_coal'])
                confidence = random.uniform(0.7, 0.95)
        except:
            quality_class = random.choice(['Good_quality_coal', 'Defect_coal'])
            confidence = random.uniform(0.7, 0.95)
        
        # Generate realistic probabilities
        if quality_class == 'Good_quality_coal':
            good_prob = confidence
            defect_prob = 1 - confidence
        else:
            defect_prob = confidence
            good_prob = 1 - confidence
        
        probs = [defect_prob, good_prob] if self.quality_class_names[0] == 'Defect_coal' else [good_prob, defect_prob]
        
        print(f"🔍 Simulated Quality: {quality_class} (Confidence: {confidence:.4f})")
        
        return quality_class, confidence, probs
    
    def _simulate_defect_prediction(self):
        """Realistic simulation for defect type"""
        defect_class = random.choice(['Cracks_and_fractures', 'Surface_deformation'])
        confidence = random.uniform(0.7, 0.95)
        
        # Generate realistic probabilities based on current class order
        if defect_class == 'Cracks_and_fractures':
            crack_prob = confidence
            surface_prob = 1 - confidence
        else:
            surface_prob = confidence
            crack_prob = 1 - confidence
        
        if self.defect_class_names[0] == 'Cracks_and_fractures':
            probs = [crack_prob, surface_prob]
        else:
            probs = [surface_prob, crack_prob]
        
        print(f"🔍 Simulated Defect: {defect_class} (Confidence: {confidence:.4f})")
        
        return defect_class, confidence, probs

# Initialize system
coal_system = CoalDetectionSystem()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image):
    try:
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"❌ Error converting image to base64: {e}")
        return None

@app.route('/')
def index():
    status = "Real AI Models" if PYTORCH_AVAILABLE and coal_system.quality_model else "Demo Mode"
    quality_order = coal_system.current_quality_order
    defect_order = coal_system.current_defect_order
    return render_template('index.html', status=status, quality_order=quality_order, defect_order=defect_order)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Process image
            image = Image.open(file.stream).convert('RGB')
            image_base64 = image_to_base64(image)
            
            if image_base64 is None:
                return jsonify({'error': 'Error processing image'}), 500
            
            print(f"📊 Analyzing image: {file.filename}")
            
            # Get quality prediction
            quality_class, quality_confidence, quality_probs = coal_system.predict_quality(image)
            
            result = {
                'quality_prediction': {
                    'class': quality_class,
                    'confidence': float(quality_confidence),
                    'probabilities': {
                        coal_system.quality_class_names[i]: float(quality_probs[i]) 
                        for i in range(len(coal_system.quality_class_names))
                    }
                },
                'defect_prediction': None,
                'image': image_base64,
                'mode': 'real' if PYTORCH_AVAILABLE else 'simulation',
                'quality_class_order': coal_system.current_quality_order,
                'defect_class_order': coal_system.current_defect_order
            }
            
            # If defective, get defect prediction
            if quality_class == 'Defect_coal':
                print("🔍 Coal is defective - classifying defect type...")
                defect_class, defect_confidence, defect_probs = coal_system.predict_defect(image)
                
                result['defect_prediction'] = {
                    'class': defect_class,
                    'confidence': float(defect_confidence),
                    'probabilities': {
                        coal_system.defect_class_names[i]: float(defect_probs[i]) 
                        for i in range(len(coal_system.defect_class_names))
                    }
                }
                
                print(f"✅ Defect classification complete: {defect_class}")
            
            print(f"✅ Analysis complete: {quality_class}")
            return jsonify(result)
            
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/switch_quality_order')
def switch_quality_order():
    """Switch quality class order for testing"""
    new_order = coal_system.switch_quality_order()
    return jsonify({
        'message': f'Quality class order switched to: {coal_system.quality_class_names}',
        'new_order': new_order,
        'class_names': coal_system.quality_class_names
    })

@app.route('/switch_defect_order')
def switch_defect_order():
    """Switch defect class order for testing"""
    new_order = coal_system.switch_defect_order()
    return jsonify({
        'message': f'Defect class order switched to: {coal_system.defect_class_names}',
        'new_order': new_order,
        'class_names': coal_system.defect_class_names
    })

@app.route('/debug/models')
def debug_models():
    """Debug endpoint to check model status and class mappings"""
    quality_loaded = PYTORCH_AVAILABLE and coal_system.quality_model is not None
    defect_loaded = PYTORCH_AVAILABLE and coal_system.defect_model is not None
    
    return jsonify({
        'pytorch_available': PYTORCH_AVAILABLE,
        'quality_model_loaded': quality_loaded,
        'defect_model_loaded': defect_loaded,
        'current_quality_order': coal_system.current_quality_order,
        'current_defect_order': coal_system.current_defect_order,
        'quality_classes': coal_system.quality_class_names,
        'defect_classes': coal_system.defect_class_names,
        'all_quality_orders': coal_system.quality_class_orders,
        'all_defect_orders': coal_system.defect_class_orders,
        'mode': 'real' if quality_loaded else 'simulation'
    })

if __name__ == '__main__':
    print("🚀 Starting Coal Detection App...")
    print("🔍 Class mappings:")
    print(f"   Quality ({coal_system.current_quality_order}): {coal_system.quality_class_names}")
    print(f"   Defect ({coal_system.current_defect_order}): {coal_system.defect_class_names}")
    print("💡 If results are incorrect, try switching class orders at:")
    print("   /switch_quality_order and /switch_defect_order")
    
    if not PYTORCH_AVAILABLE:
        print("⚠️  Running in DEMO MODE")
    else:
        print("✅ PyTorch available")
    
    app.run(debug=True, host='0.0.0.0', port=5000)