import pandas as pd
import requests
from dotenv import load_dotenv
import os
from deepface import DeepFace

load_dotenv("../.env")

X_API_KEY = os.getenv('X-API-KEY')  # Assuming the API key is in the environment variable
only_search_ai_endpoint = os.getenv('ONLY_SEARCH_AI_ENDPOINT')
num_pages = 2  # or any number of pages you want to process

def analyze_gender_from_image(image_url):
    try:
        print(f"Running DeepFace for img_url: {image_url}")
        result = DeepFace.analyze(image_url, actions=['gender'])
        print("Result:", result)
        gender = result[0]['gender']
        confidence = result[0]['gender_confidence'] / 100
        return {'gender': gender.lower(), 'confidence': round(confidence, 2)}

    except Exception as e:
        print("Error:", e)
        return {'gender': 'unknown', 'confidence': 0.0, 'error': str(e)}
    
def run_profile_and_predict_gender(num_pages, profile_url, only_search_ai_endpoint):
    gender_data = []  # List to store data

    for page in range(1, num_pages + 1):  # inclusive of num_pages
        try:
            response = requests.post(
                url=profile_url,
                json={"page": page}
            )
            response.raise_for_status()
            data = response.json()['data']

            for item in data:
                profile = item['ProfileDatum']
                if profile:
                    username = item.get('username', None)
                    avatar = profile.get('avatar', None)
                    if avatar:
                        gender = None
                        print(f'username {username} - profile_url {avatar}')
                        try:
                            gender_response = analyze_gender_from_image(avatar)
                            # gender_response.raise_for_status()  # Handle potential errors
                            # gender_result = gender_response.json()
                            gender = gender_response.get('gender', None)  # Get gender
                            confidence_score = gender_response.get('confidence', 0.0)
                        except KeyError as e:
                            print(f"KeyError for {username} : {e}")
                            break
                    else:
                        gender = 'unknown'
                        confidence_score = 0

                    gender_data.append({'username': username, 'gender': gender, 'confidence_score': confidence_score, 'avatar': avatar})

        except requests.exceptions.RequestException as e:
            print(f"Failed to get profiles for page {page}: {e}")

    df = pd.DataFrame(gender_data)
    return df


# Call the function
profile_url = os.getenv('OF_PROFILES_URL')  # Ensure the URL is set in the environment
df = run_profile_and_predict_gender(num_pages, profile_url, only_search_ai_endpoint)

# Display the dataframe
df.to_csv(r'../outputs/user_attributes.csv')
