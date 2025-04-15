from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import requests
from io import BytesIO
from nudenet import NudeDetector
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
nudity_classifier = NudeDetector()
X_API_KEY = os.getenv('X-API-KEY')

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

