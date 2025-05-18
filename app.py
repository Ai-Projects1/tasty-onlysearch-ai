import tempfile
import os
import torch
import joblib
import clip
import face_recognition
import requests
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
from nudenet import NudeDetector
from deepface import DeepFace
from dotenv import load_dotenv
from torchvision import models,transforms
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

##### ONE TIME LOADS #####
load_dotenv()
app = Flask(__name__)
nudity_classifier = NudeDetector()
X_API_KEY = os.getenv('X-API-KEY')

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
llm_pipeline = pipeline("text-generation", model="gpt2")  # Replace with OpenAI API if preferred


url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(url, key)

# Define the same model architecture used during training
class GenderClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(GenderClassifier, self).__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base(x)

# Load LabelEncoder
le = joblib.load(r'./pickles/label_encoder.pkl')
# Load ResNet18 model directly
model = models.resnet18()
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(le.classes_))  # Adjust output layer
model.load_state_dict(torch.load(r'./pickles/gender_classifier.pth', map_location=torch.device('cpu')))
model.eval()

def load_image(input_file=None, input_url=None):
    if input_file:
        return face_recognition.load_image_file(input_file)
    elif input_url:
        try:
            response = requests.get(input_url)
            response.raise_for_status()
            return face_recognition.load_image_file(BytesIO(response.content))
        except Exception as e:
            raise ValueError(f"Failed to fetch image from URL: {e}")
    else:
        raise ValueError("No input provided.")

#### COMPARE FACES #####
@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    provided_key = request.headers.get("X-API-Key")
    if provided_key != X_API_KEY:
        return jsonify({"error": "Unauthorized. Invalid API Key."}), 401
    
    try:
        # Get JSON input from the request body
        data = request.get_json()

        original_url = data.get('original_url')
        comparison_url = data.get('comparison_url')

        if not original_url or not comparison_url:
            return jsonify({'error': 'Both original_url and comparison_url are required.'}), 400

        # Load images from URLs
        original_image = load_image(input_url=original_url)
        comparison_image = load_image(input_url=comparison_url)

        # Get encodings
        original_encodings = face_recognition.face_encodings(original_image)
        comparison_encodings = face_recognition.face_encodings(comparison_image)

        if not original_encodings or not comparison_encodings:
            return jsonify({'error': 'Face not found in one or both images.'}), 400

        original_encoding = original_encodings[0]
        comparison_encoding = comparison_encodings[0]

        # Face distance to confidence score
        face_distance = face_recognition.face_distance([original_encoding], comparison_encoding)[0]
        confidence = (1 - face_distance) * 100
        
        threshold = float(data.get('threshold', 0.50)) * 100
        is_match = bool(confidence >= threshold)
        print('confidence',confidence)
        print('is_match',is_match)
        
        return jsonify({
            'confidence_score': round(confidence, 2),
            'match': is_match
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

##### DETECT NUDITY ######
@app.route('/detect_nudity', methods=['POST'])
def detect_nudity():
    provided_key = request.headers.get("X-API-Key")
    if provided_key != X_API_KEY:
        return jsonify({"error": "Unauthorized. Invalid API Key."}), 401
    
    try:
        # Case 1: User uploads an image file
        if 'image' in request.files:
            image_file = request.files['image']
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                image_path = temp_file.name
                image_file.save(image_path)

        # Case 2: User sends an image URL via JSON
        elif request.is_json:
            data = request.get_json()
            image_url = data.get('image_url')
            if not image_url:
                return jsonify({'error': 'image_url is required'}), 400

            response = requests.get(image_url)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                image_path = temp_file.name
                temp_file.write(response.content)

        else:
            return jsonify({'error': 'Image file or image_url is required'}), 400

        # Classify using NudeNet
        results = nudity_classifier.detect(image_path)
        nude_labels = ['FEMALE_BREAST_EXPOSED','FEMALE_GENITALIA_EXPOSED','MALE_BREAST_EXPOSED',
                       'ANUS_EXPOSED','MALE_GENITALIA_EXPOSED']
        
        is_nude = False
        for item in results:
            if item['class'] in nude_labels:
                is_nude = True
                break

        # Clean up the temp file
        os.remove(image_path)

        if not results:
            return jsonify({'error': 'Failed to classify image'}), 500

        return jsonify({
            'nudity': is_nude,
            'confidence_scores': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

##### DETECT FACE COORDINATES #####
@app.route('/face_coordinates', methods=['POST'])
def face_coordinates():
    provided_key = request.headers.get("X-API-Key")
    if provided_key != X_API_KEY:
        return jsonify({"error": "Unauthorized. Invalid API Key."}), 401
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid or missing JSON body'}), 400

        # Load image from URL or base64
        if 'image_url' in data:
            image = load_image(input_url=data['image_url'])
        else:
            return jsonify({'error': 'Provide image_url or image_base64'}), 400

        face_locations = face_recognition.face_locations(image)

        if not face_locations:
            return jsonify({'faces': []})  # no face found

        # Format: (top, right, bottom, left)
        boxes = []
        for top, right, bottom, left in face_locations:
            boxes.append({
                'top': top,
                'right': right,
                'bottom': bottom,
                'left': left,
                'width': right - left,
                'height': bottom - top
            })

        return jsonify({'faces': boxes})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

#### PREDICT GENDER  #####
def predict_gender_from_custom_model(url, model, label_encoder):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    try:
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            confidence = probs[0][pred_idx].item()

        return {
            'predicted_by': 'custom_model',
            'gender': pred_label,
            'confidence': round(confidence, 4),
        }
    except Exception as e:
        return {
            'error': str(e),
            'gender': 'unknown',
            'confidence': 0.0
        }

def analyze_gender_logic(image_url):
    try:
        print(f"Running DeepFace for img_url: {image_url}")
        result = DeepFace.analyze(image_url, actions=['gender'])
        gender = result[0]['dominant_gender']
        confidence = result[0]['gender']
        if gender.lower() != 'unknown':
            return jsonify({
                'predicted_by': 'deepface',
                'gender': gender.lower(),
                'confidence': confidence
            })
    except Exception as e:
        print("DeepFace Error:", e)

    fallback_result = predict_gender_from_custom_model(image_url,model,le)
    return fallback_result

@app.route('/analyze-gender', methods=['POST'])
def analyze_gender():
    data = request.get_json()
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({'error': 'image_url is required'}), 400

    result = analyze_gender_logic(image_url)
    return result

@app.route('/bulk-analyze-gender', methods=['POST'])
def run_profile_and_predict_gender(num_pages, profile_url, start_page=3):

    for page in range(start_page, num_pages + 1):
        print(f"\n=== Processing page {page} ===")
        gender_data = []

        try:
            response = requests.post(url=profile_url, json={"page": page})
            response.raise_for_status()
            data = response.json().get('data', [])
        except Exception as e:
            print(f"Failed to fetch profiles for page {page}: {e}")
            data = []  # Continue with an empty list so CSV still gets saved

        for item in data:
            try:
                profile = item.get('ProfileDatum', {})
                username = item.get('username', None)
                avatar = profile.get('avatar', None)

                if avatar:
                    print(f'username: {username} - avatar: {avatar}')
                    gender_response = analyze_gender_logic(avatar)
                    
                    gender_data.append(gender_response)

            except Exception as e:
                print(f"Error processing profile for username {username}: {e}")
                continue  # Skip to next profile

#### USER RECOMMENDATIONS BASED ON SEARCH CRITERIA ####
def get_face_embedding(image_path):
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name='Facenet')[0]['embedding']
        return embedding
    except Exception as e:
        print(f"[Face] No face found: {e}")
        return None
    
def get_clip_image_embedding(image_path):
    try:
        response = requests.get(image_path, timeout=10)
        response.raise_for_status()  # Raise error for bad status
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image_input = clip_preprocess(image).unsqueeze(0)

        with torch.no_grad():
            embedding = clip_model.encode_image(image_input).squeeze().numpy()
        return embedding
    except Exception as e:
        print(f"Failed to get embedding for URL {image_path}: {e}")
        return None
    
def get_clip_text_embedding(text):
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_input).squeeze().cpu().numpy()
    return text_features

@app.route('/recommend', methods=['POST'])
def recommend_similar_images():
    data = request.json
    query_text = data.get('query_text')
    query_image_url = data.get('query_image_url')
    top_k = int(data.get('top_k', 5))

    if not query_text and not query_image_url:
        return jsonify({"error": "Provide at least 'query_text' or 'query_image_url'"}), 400

    query_embeddings = []

    if query_text:
        text_embedding = get_clip_text_embedding(query_text)
        query_embeddings.append(text_embedding)

    if query_image_url:
        image_embedding = get_clip_image_embedding(query_image_url)
        if image_embedding is not None:
            query_embeddings.append(image_embedding)

    if not query_embeddings:
        return jsonify([])

    query_vector = np.mean(query_embeddings, axis=0).reshape(1, -1)

    # Fetch data from Supabase
    response = supabase.table("of_profiles").select("*").execute()
    records = response.data
    df = pd.DataFrame(records)
    df = df[df['embeddings'].notna()]
    df['embedding_array'] = df['embeddings'].apply(lambda x: np.array(x, dtype=np.float32))

    image_features_matrix = np.vstack(df['embedding_array'].values.tolist())

    similarity_scores = cosine_similarity(query_vector, image_features_matrix)[0]
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    top_results = df.iloc[top_indices].copy()
    top_results['similarity_score'] = similarity_scores[top_indices]

    # Return only relevant fields
    return jsonify(top_results[['username', 'avatar', 'similarity_score']].to_dict(orient="records"))

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=8080)

