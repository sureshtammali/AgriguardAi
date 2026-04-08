import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Load GPT-2 model
gpt2_tokenizer = None
gpt2_model = None

def load_gpt2():
    global gpt2_tokenizer, gpt2_model
    if gpt2_tokenizer is None:
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    return gpt2_tokenizer, gpt2_model

def generate_disease_info(crop, disease_name, confidence):
    tokenizer, model = load_gpt2()
    
    if disease_name.lower() == 'healthy':
        return {
            'confidence': confidence,
            'severity': 'None',
            'description': 'Excellent news! Your crop appears healthy with no visible signs of disease. The plant shows normal growth patterns and leaf coloration.',
            'symptoms': 'No disease symptoms detected. Leaves are green and vibrant with no discoloration, spots, or wilting.',
            'causes': 'Not applicable - crop is healthy.',
            'remedy': 'Continue current maintenance practices including regular watering, proper spacing, and monitoring for early signs of stress or disease.',
            'prevention': 'Maintain good agricultural practices: proper irrigation, crop rotation, adequate spacing, and regular field inspection.',
            'fertilizer': 'Use balanced NPK fertilizer (10-10-10 or 14-14-14) for optimal growth. Apply organic compost every 2-3 weeks.',
            'organic_treatment': 'Apply neem oil spray monthly as preventive measure. Use compost tea for soil health.',
            'chemical_treatment': 'No chemical treatment needed. Continue preventive fungicide if in high-risk area.',
            'timeline': 'Continue monitoring weekly. Maintain current care routine.',
            'cost_estimate': 'Maintenance cost: $20-40 per acre per month for preventive care.'
        }
    
    # Determine severity
    severity = 'High' if confidence > 85 else 'Medium' if confidence > 70 else 'Low'
    
    # Get detailed disease information
    disease_details = get_disease_details(crop, disease_name)
    
    # Generate AI-enhanced description
    prompt = f"Agricultural disease {disease_name} in {crop} plants causes"
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=50)
    outputs = model.generate(inputs['input_ids'], max_length=120, num_return_sequences=1, 
                            temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id,
                            top_p=0.9, repetition_penalty=1.2)
    ai_description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        'confidence': confidence,
        'severity': severity,
        'description': disease_details['description'],
        'symptoms': disease_details['symptoms'],
        'causes': disease_details['causes'],
        'remedy': disease_details['remedy'],
        'prevention': disease_details['prevention'],
        'fertilizer': get_disease_fertilizer(disease_name),
        'organic_treatment': disease_details['organic'],
        'chemical_treatment': disease_details['chemical'],
        'timeline': disease_details['timeline'],
        'cost_estimate': disease_details['cost']
    }

def get_disease_details(crop, disease_name):
    disease_db = {
        'Early_blight': {
            'description': 'Early blight is a common fungal disease caused by Alternaria solani. It affects leaves, stems, and fruits, causing dark brown spots with concentric rings (target-like pattern). The disease thrives in warm, humid conditions and can significantly reduce yield if left untreated.',
            'symptoms': 'Dark brown to black spots with concentric rings on older leaves, yellowing around spots, leaf drop, stem lesions, and fruit rot near the stem end. Symptoms typically start on lower, older leaves.',
            'causes': 'Caused by Alternaria solani fungus. Spreads through water splash, wind, contaminated tools, and infected plant debris. Favored by warm temperatures (24-29°C) and high humidity.',
            'remedy': 'Remove and destroy infected leaves immediately. Apply copper-based fungicides or chlorothalonil every 7-10 days. Improve air circulation by proper spacing and pruning.',
            'prevention': 'Use disease-resistant varieties, practice crop rotation (3-4 years), mulch to prevent soil splash, avoid overhead irrigation, and remove plant debris after harvest.',
            'organic': 'Spray with Bacillus subtilis or copper sulfate. Apply neem oil (2-3 ml/L) weekly. Use baking soda solution (1 tbsp/gallon water). Compost tea application.',
            'chemical': 'Mancozeb (2g/L), Chlorothalonil (2ml/L), or Azoxystrobin (1ml/L). Alternate fungicides to prevent resistance. Apply every 7-14 days.',
            'timeline': 'Treatment duration: 3-4 weeks. Visible improvement in 7-10 days. Continue preventive sprays throughout season.',
            'cost': 'Treatment cost: $50-100 per acre. Organic treatment: $30-60 per acre. Preventive measures: $20-40 per acre.'
        },
        'Late_blight': {
            'description': 'Late blight is a devastating disease caused by Phytophthora infestans, the same pathogen that caused the Irish Potato Famine. It can destroy entire crops within days under favorable conditions. Affects leaves, stems, and fruits.',
            'symptoms': 'Water-soaked spots on leaves that turn brown/black, white fuzzy growth on leaf undersides, rapid plant collapse, brown lesions on stems, and firm brown rot on fruits. Disease spreads rapidly in cool, wet weather.',
            'causes': 'Phytophthora infestans (water mold). Spreads through wind-blown spores, infected seed tubers, and water splash. Thrives in cool (15-20°C), wet conditions with high humidity.',
            'remedy': 'Act immediately! Remove and burn infected plants. Apply systemic fungicides containing metalaxyl or dimethomorph. Spray every 5-7 days during outbreak.',
            'prevention': 'Plant certified disease-free seeds, use resistant varieties, ensure good drainage, avoid overhead watering, and apply preventive fungicides in high-risk periods.',
            'organic': 'Copper hydroxide spray (3g/L) every 5 days. Bordeaux mixture application. Remove infected tissue immediately. Limited organic options - prevention is key.',
            'chemical': 'Metalaxyl + Mancozeb (2.5g/L), Dimethomorph (1ml/L), or Cymoxanil + Famoxadone. Spray every 5-7 days. Rotate fungicide groups.',
            'timeline': 'Critical: Treat within 24-48 hours of detection. Continue treatment for 4-6 weeks. Monitor daily during wet weather.',
            'cost': 'Emergency treatment: $100-200 per acre. Preventive program: $80-150 per acre per season. Crop loss without treatment: 50-100%.'
        },
        'Bacterial_spot': {
            'description': 'Bacterial spot is caused by Xanthomonas species. It affects leaves, stems, and fruits, causing significant yield loss and fruit quality reduction. The disease is highly contagious and difficult to control once established.',
            'symptoms': 'Small, dark brown to black spots with yellow halos on leaves, raised brown spots on fruits, leaf yellowing and drop, and stem lesions. Spots may merge causing large dead areas.',
            'causes': 'Xanthomonas bacteria spread through water splash, contaminated seeds, infected transplants, and handling wet plants. Favored by warm (25-30°C), humid conditions.',
            'remedy': 'Remove infected plants immediately. Apply copper-based bactericides. Avoid working with wet plants. Improve air circulation and reduce leaf wetness.',
            'prevention': 'Use disease-free seeds and transplants, practice 3-year crop rotation, avoid overhead irrigation, disinfect tools, and apply preventive copper sprays.',
            'organic': 'Copper sulfate (2-3g/L) weekly. Bacillus subtilis spray. Hydrogen peroxide solution (1:10). Remove infected tissue promptly.',
            'chemical': 'Copper hydroxide (3g/L) + Mancozeb (2g/L). Streptomycin sulfate where legal. Acibenzolar-S-methyl for resistance. Apply every 7-10 days.',
            'timeline': 'Treatment: 4-6 weeks. Improvement visible in 10-14 days. Preventive sprays throughout growing season recommended.',
            'cost': 'Treatment: $60-120 per acre. Preventive program: $40-80 per acre. Resistant varieties may reduce costs by 30-50%.'
        },
        'Leaf_scorch': {
            'description': 'Leaf scorch is a physiological disorder often caused by environmental stress, though can be associated with fungal pathogens. Results in browning and drying of leaf margins and tips, reducing photosynthesis and plant vigor.',
            'symptoms': 'Brown, dried leaf edges and tips, yellowing between veins, leaf curling, premature leaf drop, and reduced fruit size. Symptoms worsen during hot, dry periods.',
            'causes': 'Water stress, salt accumulation, nutrient imbalance (especially potassium deficiency), root damage, or fungal infection. Exacerbated by high temperatures and low humidity.',
            'remedy': 'Ensure consistent watering, improve soil drainage, apply potassium-rich fertilizer, mulch to retain moisture, and provide shade during extreme heat.',
            'prevention': 'Maintain consistent soil moisture, use drip irrigation, apply organic mulch, ensure proper drainage, and monitor soil nutrient levels regularly.',
            'organic': 'Apply compost tea, seaweed extract (foliar spray), and potassium sulfate. Mulch with organic matter. Ensure adequate watering schedule.',
            'chemical': 'Potassium nitrate (5g/L foliar spray), balanced fertilizer application. If fungal: apply appropriate fungicide based on pathogen identification.',
            'timeline': 'Recovery: 2-4 weeks with proper care. New growth should appear healthy. Continue monitoring and adjust watering/fertilization.',
            'cost': 'Treatment: $30-60 per acre. Irrigation system improvement: $200-500 per acre (one-time). Ongoing maintenance: $20-40 per acre.'
        },
        'Powdery_mildew': {
            'description': 'Powdery mildew is a fungal disease causing white, powdery growth on leaves and stems. While rarely fatal, it weakens plants, reduces yield, and affects fruit quality. Common in many crops and spreads rapidly.',
            'symptoms': 'White powdery coating on leaves (upper surface), leaf curling and distortion, yellowing leaves, stunted growth, and reduced fruit quality. Severe cases cause leaf drop.',
            'causes': 'Various fungal species (Erysiphales order). Spreads by wind-borne spores. Thrives in moderate temperatures (20-25°C) with high humidity but doesn\'t require water on leaves.',
            'remedy': 'Remove infected leaves, improve air circulation, reduce humidity, apply sulfur-based fungicides or potassium bicarbonate. Spray early morning.',
            'prevention': 'Plant resistant varieties, ensure proper spacing, prune for airflow, avoid excess nitrogen, and apply preventive sulfur or biological fungicides.',
            'organic': 'Sulfur dust or spray (3g/L), baking soda solution (1 tbsp + 1 tsp oil per gallon), neem oil (2ml/L), or milk spray (1:9 ratio). Apply weekly.',
            'chemical': 'Myclobutanil (0.5ml/L), Trifloxystrobin (0.5ml/L), or Potassium bicarbonate (5g/L). Rotate fungicide groups. Spray every 7-14 days.',
            'timeline': 'Treatment: 3-4 weeks. Visible improvement in 5-7 days. Continue preventive applications throughout season, especially during favorable conditions.',
            'cost': 'Organic treatment: $25-50 per acre. Chemical treatment: $40-80 per acre. Preventive program: $30-60 per acre per season.'
        }
    }
    
    # Default template for diseases not in database
    default = {
        'description': f'{disease_name.replace("_", " ")} is a plant disease affecting {crop}. It can cause significant damage if not treated promptly. Early detection and proper management are crucial for crop health.',
        'symptoms': 'Visible signs include leaf discoloration, spots, wilting, stunted growth, and reduced vigor. Symptoms may vary based on disease stage and environmental conditions.',
        'causes': 'Caused by pathogenic organisms or environmental stress. Spreads through water, wind, contaminated tools, or infected plant material. Favorable conditions include high humidity and moderate temperatures.',
        'remedy': 'Remove infected plant parts immediately. Apply appropriate fungicides or bactericides. Improve cultural practices including spacing, watering, and sanitation.',
        'prevention': 'Use disease-resistant varieties, practice crop rotation, maintain field hygiene, ensure proper drainage, and monitor plants regularly for early detection.',
        'organic': 'Apply neem oil, copper-based products, or biological control agents. Use compost tea and maintain soil health. Remove infected tissue promptly.',
        'chemical': 'Apply appropriate fungicides or bactericides based on pathogen type. Follow label instructions. Rotate chemical groups to prevent resistance.',
        'timeline': 'Treatment duration: 3-6 weeks. Monitor progress weekly. Continue preventive measures throughout growing season.',
        'cost': 'Treatment cost: $50-100 per acre. Preventive measures: $30-60 per acre. Early intervention reduces overall costs significantly.'
    }
    
    # Find matching disease
    for key in disease_db:
        if key.lower() in disease_name.lower().replace('_', ' ').replace('(', '').replace(')', ''):
            return disease_db[key]
    
    return default

def get_disease_fertilizer(disease_name):
    disease_fertilizer_map = {
        'Early_blight': 'Use balanced NPK (20-20-20) with added calcium. Apply fungicide with copper compounds.',
        'Late_blight': 'Apply high potassium fertilizer (10-20-30) and phosphorus-rich compounds.',
        'Bacterial_spot': 'Use calcium-based fertilizers and reduce nitrogen. Apply copper fungicides.',
        'Leaf_scorch': 'Apply potassium-rich fertilizer (10-10-30) and ensure adequate watering.',
        'Powdery_mildew': 'Use balanced fertilizer (14-14-14) with sulfur-based fungicides.',
        'Black_rot': 'Apply phosphorus-rich fertilizer (10-26-26) and copper-based treatments.',
        'Apple_scab': 'Use balanced NPK (17-17-17) with preventive fungicide sprays.',
        'Cedar_apple_rust': 'Apply iron and zinc supplements with fungicide treatment.',
        'Common_rust': 'Use nitrogen-controlled fertilizer (10-20-20) with fungicide application.',
        'Northern_Leaf_Blight': 'Apply balanced fertilizer (20-20-20) and improve drainage.',
        'Cercospora_leaf_spot': 'Use potassium-rich fertilizer (15-15-30) with fungicide.',
        'Leaf_Mold': 'Apply calcium nitrate and reduce humidity. Use fungicides.',
        'Septoria_leaf_spot': 'Use balanced NPK (15-15-15) with copper fungicides.',
        'Spider_mites': 'Apply micronutrient fertilizer with neem oil or miticides.',
        'Target_Spot': 'Use potassium-rich fertilizer (10-15-30) with fungicide rotation.',
        'Tomato_Yellow_Leaf_Curl_Virus': 'Apply micronutrient mix and control whiteflies. No cure available.',
        'Tomato_mosaic_virus': 'Use balanced fertilizer (14-14-14) and remove infected plants.',
        'Esca': 'Apply potassium and magnesium supplements. Prune infected areas.',
        'Leaf_blight': 'Use phosphorus-rich fertilizer (10-26-26) with fungicide treatment.'
    }
    
    for key in disease_fertilizer_map:
        if key.lower() in disease_name.lower().replace('_', ' ').replace('(', '').replace(')', ''):
            return disease_fertilizer_map[key]
    
    return 'Apply balanced NPK fertilizer (17-17-17) and consult agricultural expert for specific treatment.'

def get_model(path):
    model = load_model(path, compile=False)
    return model

def img_predict(path, crop):
    data = load_img(path, target_size=(224, 224, 3))
    data = np.asarray(data).reshape((-1, 224, 224, 3))
    data = data * 1.0 / 255
    model = get_model(os.path.join(BASE_DIR, 'models', 'DL_models', f'{crop}_model.h5'))
    
    predictions = model.predict(data)[0]
    
    if len(crop_diseases_classes[crop]) > 2:
        predicted = np.argmax(predictions)
        confidence = float(predictions[predicted]) * 100
    else:
        p = predictions[0]
        predicted = int(np.round(p))
        confidence = float(p if predicted == 1 else (1 - p)) * 100
    
    return predicted, confidence

def get_diseases_classes(crop, prediction):
    crop_classes = crop_diseases_classes[crop]
    return crop_classes[prediction][1].replace("_", " ")

def get_crop_recommendation(item):
    scaler_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'crop_scaler.pkl')
    model_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'crop_model.pkl')

    with open(scaler_path, 'rb') as f:
        crop_scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        crop_model = pickle.load(f)

    scaled_item = crop_scaler.transform(np.array(item).reshape(-1, len(item)))
    prediction = crop_model.predict(scaled_item)[0]
    return crops[prediction]

def get_fertilizer_recommendation(num_features, cat_features):
    scaler_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'fertilizer_scaler.pkl')
    model_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'fertilizer_model.pkl')
    
    with open(scaler_path, 'rb') as f:
        fertilizer_scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        fertilizer_model = pickle.load(f)

    scaled_features = fertilizer_scaler.transform(np.array(num_features).reshape(-1, len(num_features)))
    cat_features = np.array(cat_features).reshape(-1, len(cat_features))
    item = np.concatenate([scaled_features, cat_features], axis=1)
    prediction = fertilizer_model.predict(item)[0]
    return fertilizer_classes[prediction]

crop_diseases_classes = {'strawberry': [(0, 'Leaf_scorch'), (1, 'healthy')],

			   'patato': [(0, 'Early_blight'),
				 (1, 'Late_blight'),
				 (2, 'healthy')],

			   'corn': [(0, 'Cercospora_leaf_spot Gray_leaf_spot'),
				 (1, 'Common_rust_'),
				 (2, 'Northern_Leaf_Blight'),
				 (3, 'healthy')],

			   'apple': [(0, 'Apple_scab'),
				 (1, 'Black_rot'),
				 (2, 'Cedar_apple_rust'),
				 (3, 'healthy')],

			   'cherry': [(0, 'Powdery_mildew'),
				 (1, 'healthy')],

			   'grape': [(0, 'Black_rot'),
				 (1, 'Esca_(Black_Measles)'),
				 (2, 'Leaf_blight_(Isariopsis_Leaf_Spot)'),
				 (3, 'healthy')],

			   'peach': [(0, 'Bacterial_spot'), (1, 'healthy')],

			   'pepper': [(0, 'Bacterial_spot'),
				 (1, 'healthy')],
				
			   'tomato': [(0, 'Bacterial_spot'),
				 (1, 'Early_blight'),
				 (2, 'Late_blight'),
				 (3, 'Leaf_Mold'),
				 (4, 'Septoria_leaf_spot'),
				 (5, 'Spider_mites Two-spotted_spider_mite'),
				 (6, 'Target_Spot'),
				 (7, 'Tomato_Yellow_Leaf_Curl_Virus'),
				 (8, 'Tomato_mosaic_virus'),
				 (9, 'healthy')]}

crop_list = list(crop_diseases_classes.keys())


crops = {'apple': 1, 'banana': 2, 'blackgram': 3, 'chickpea': 4, 'coconut': 5, 'coffee': 6, 'cotton': 7, 'grapes': 8, 'jute': 9, 'kidneybeans': 10, 'lentil': 11, 'maize': 12, 'mango': 13, 'mothbeans': 14, 'mungbean': 15, 'muskmelon': 16, 'orange': 17, 'papaya': 18, 'pigeonpeas': 19, 'pomegranate': 20, 'rice': 21, 'watermelon': 22}

crops = list(crops.keys())

soil_types = ['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
Crop_types = ['Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets', 'Oil seeds', 'Paddy', 'Pulses', 'Sugarcane', 'Tobacco', 'Wheat']

fertilizer_classes = ['10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP', 'Urea']
