# 🌾 AgriGuardAI - Smart Agriculture Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0.3-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8.0-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AgriGuardAI** is an AI-powered agricultural platform that helps farmers protect their crops using advanced machine learning and deep learning technologies. The platform provides crop disease detection, fertilizer recommendations, and crop suggestions to maximize agricultural productivity.

## 🎯 Features

### 🔬 Crop Disease Detection
- **AI-Powered Analysis**: Uses CNN models for accurate disease identification
- **GPT-2 Enhanced Reports**: Comprehensive disease analysis with treatment recommendations
- **Multi-Crop Support**: Supports 9+ crop types including tomato, potato, corn, apple, grape, etc.
- **30+ Disease Types**: Detects various diseases like Early Blight, Late Blight, Bacterial Spot, etc.
- **95%+ Accuracy**: High precision disease detection with confidence scoring
- **Detailed Treatment Plans**: Organic and chemical treatment options with cost estimates

### 🌱 Fertilizer Recommendation
- **Smart NPK Analysis**: Recommends optimal fertilizer based on soil and crop conditions
- **Soil Type Consideration**: Supports Black, Clayey, Loamy, Red, and Sandy soils
- **Crop-Specific Formulations**: Tailored recommendations for different crop types
- **Cost-Effective Solutions**: Budget-friendly fertilizer suggestions

### 🌾 Crop Recommendation
- **Data-Driven Suggestions**: ML-based crop recommendations using soil parameters
- **Environmental Factors**: Considers temperature, humidity, pH, and rainfall
- **Nutrient Analysis**: NPK (Nitrogen, Phosphorus, Potassium) level assessment
- **22+ Crop Options**: Supports major crops like rice, wheat, maize, cotton, etc.

## 🚀 Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Keras, Scikit-learn
- **AI Enhancement**: GPT-2 (Transformers)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap
- **Image Processing**: PIL, OpenCV
- **Data Processing**: NumPy, Pandas

## 📁 Project Structure

```
AgriGuardAI/
├── AgriGo/
│   ├── AgriGo/
│   │   ├── dataset/                 # Training datasets
│   │   │   ├── Crop_recommendation.csv
│   │   │   └── Fertilizer Prediction.csv
│   │   ├── models/                  # Trained ML/DL models
│   │   │   ├── DL_models/          # Deep learning models for disease detection
│   │   │   │   ├── apple_model.h5
│   │   │   │   ├── tomato_model.h5
│   │   │   │   └── ...
│   │   │   └── ML_models/          # Machine learning models
│   │   │       ├── crop_model.pkl
│   │   │       ├── fertilizer_model.pkl
│   │   │       └── scalers
│   │   ├── static/                 # Static assets
│   │   │   ├── css/
│   │   │   ├── js/
│   │   │   └── images/
│   │   ├── templates/              # HTML templates
│   │   ├── uploads/                # Uploaded images
│   │   ├── app.py                  # Main Flask application
│   │   └── functions.py            # Core ML/AI functions
│   ├── requirements.txt            # Python dependencies
│   └── Dockerfile                  # Docker configuration
└── env/                           # Virtual environment
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/annarao/AgriGuardAI.git
cd AgriGuardAI
```

### Step 2: Create Virtual Environment
```bash
python -m venv env
# On Windows
env\Scripts\activate
# On macOS/Linux
source env/bin/activate
```

### Step 3: Install Dependencies
```bash
cd AgriGo
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
cd AgriGo
python app.py
```

### Step 5: Access the Platform
Open your web browser and navigate to:
```
http://localhost:5000
```

## 📊 Supported Crops & Diseases

### Disease Detection Crops:
- **Tomato**: 10 conditions (9 diseases + healthy)
- **Potato**: 3 conditions (Early Blight, Late Blight + healthy)
- **Corn**: 4 conditions (3 diseases + healthy)
- **Apple**: 4 conditions (3 diseases + healthy)
- **Grape**: 4 conditions (3 diseases + healthy)
- **Cherry**: 2 conditions (Powdery Mildew + healthy)
- **Peach**: 2 conditions (Bacterial Spot + healthy)
- **Pepper**: 2 conditions (Bacterial Spot + healthy)
- **Strawberry**: 2 conditions (Leaf Scorch + healthy)

### Common Diseases Detected:
- Early Blight
- Late Blight
- Bacterial Spot
- Powdery Mildew
- Apple Scab
- Black Rot
- Leaf Scorch
- Common Rust
- Northern Leaf Blight
- And more...

### Crop Recommendation Options:
Apple, Banana, Blackgram, Chickpea, Coconut, Coffee, Cotton, Grapes, Jute, Kidney Beans, Lentil, Maize, Mango, Moth Beans, Mung Bean, Muskmelon, Orange, Papaya, Pigeon Peas, Pomegranate, Rice, Watermelon

## 🎨 User Interface

The platform features a modern, responsive design with:
- **Intuitive Navigation**: Easy-to-use interface for all user levels
- **Gradient Themes**: Professional color schemes with visual appeal
- **Mobile Responsive**: Works seamlessly on desktop, tablet, and mobile devices
- **Interactive Forms**: User-friendly input forms with validation
- **Comprehensive Reports**: Detailed analysis with visual indicators

## 📈 Model Performance

- **Disease Detection Accuracy**: 95%+ across all supported crops
- **Confidence Scoring**: Real-time confidence levels for predictions
- **Processing Speed**: 2-3 seconds for disease detection
- **GPT-2 Analysis**: 3-5 seconds for comprehensive reports
- **Model Size**: Optimized for production deployment

## 🔧 API Endpoints

### Main Routes:
- `GET /` - Home page
- `GET,POST /crop-disease` - Disease detection interface
- `GET,POST /crop-recommendation` - Crop recommendation system
- `GET,POST /fertilizer-recommendation` - Fertilizer suggestion tool
- `GET /about` - Platform information
- `GET /developer` - Developer details
- `GET /contact` - Contact information

## 🧪 Usage Examples

### Disease Detection:
1. Navigate to "Crop Disease Detection"
2. Select crop type from dropdown
3. Upload clear image of affected plant
4. Get comprehensive analysis with treatment recommendations

### Crop Recommendation:
1. Go to "Crop Recommendation"
2. Enter soil parameters (N, P, K, pH, temperature, humidity, rainfall)
3. Receive optimal crop suggestions

### Fertilizer Recommendation:
1. Access "Fertilizer Recommendation"
2. Input soil nutrients and crop type
3. Get specific fertilizer formulation recommendations

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Developer Information

**Developer**: Annarao  
**USN**: 3VY24MC012  
**University**: Visvesvaraya Technological University, Kalaburagi  
**Project Title**: AgriGuard.AI – Plant Disease Detection & Fertilizer Recommendation System  
**Mentor**: Ms. Shobha Biradar, Assistant Professor, VTU CPGS Kalaburagi  
**Email**: annarao@vtu.ac.in  
**GitHub**: [@annarao](https://github.com/annarao)

## 🙏 Acknowledgments

- Visvesvaraya Technological University for academic support
- Ms. Shobha Biradar for project mentorship
- Open-source community for tools and libraries
- Agricultural research institutions for datasets

## 📞 Support

For support, email annarao@vtu.ac.in or create an issue in the GitHub repository.

## 🔮 Future Enhancements

- [ ] Mobile application development
- [ ] Multi-language support
- [ ] Weather integration
- [ ] Historical data tracking
- [ ] PDF report generation
- [ ] Real-time monitoring dashboard
- [ ] Community feedback system
- [ ] Advanced analytics and insights

## ⚠️ Disclaimer

This AI-powered platform provides informational analysis and recommendations. Results may vary based on environmental conditions and other factors. Always consult with local agricultural experts and extension services for professional advice tailored to your specific situation.

---

**Made with ❤️ for farmers and agriculture enthusiasts**

*Empowering agriculture through artificial intelligence*