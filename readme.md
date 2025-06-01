## 📱 API Endpoints Documentation

All endpoints require authentication using an API key passed via the request headers:

```http
X-API-KEY: your_api_key_here
```

---
### 1. ✅ Analyze Gender in an Image

**Endpoint:** `POST /analyze-gender`  
**URL:** `https://only-search-ai-275499389350.us-central1.run.app/analyze-gender`

**Description:**  
Analyzes a face image to predict the gender (male or female) using a machine learning model. Returns the predicted gender and confidence score.

**Request Body (JSON)**
```json
{
  "image_url": "https://example.com/image.jpg"
}
```

**Response**
```json
{
  "gender": "female",
  "confidence": 0.921
}
```

---

### 2. 👥 Bulk Analyze Gender from Profile Pages

**Endpoint:** `POST /bulk-analyze-gender`  
**URL:** `https://only-search-ai-275499389350.us-central1.run.app/bulk-analyze-gender`

**Description:**  
Fetches user profiles from a paginated endpoint, analyzes the `avatar` image of each profile, and predicts gender. Useful for processing large batches of users at once.

**Request Body (JSON)**
```json
{
  "num_pages": 5,
  "profile_url": "http://128.199.13.94:8015/api/adminpanel/all-profiles",
  "start_page": 3
}
```

**Response**
```json
{
  "message": "Processing complete"
}
```

---

### 3. 🤖 Recommend Similar Images

**Endpoint:** `POST /recommend`  
**URL:** `https://only-search-ai-275499389350.us-central1.run.app/recommend`

**Description:**  
Recommends similar profile images based on either a text prompt or image. Uses a CLIP embedding model to calculate similarity and returns top K most similar profiles.

**Request Body (JSON)**
```json
{
  "query_text": "blonde girl with sunglasses", #OPTIONAL
  "query_image_url": "https://example.com/image.jpg", #OPTIONAL
  "top_k": 5
}

```

**Response**
```json
[
  {
    "username": "emily_b123",
    "avatar": "https://site.com/avatars/abc.jpg",
    "similarity_score": 0.947
  },
  {
    "username": "samantha_xo",
    "avatar": "https://site.com/avatars/xyz.jpg",
    "similarity_score": 0.913
  }
]
```

---

### 4. 🔍 Compare Two URLs for Similarity

**Endpoint**: `POST /compare_faces`  
**URL**: `https://only-search-ai-275499389350.us-central1.run.app/compare_faces`

**Description**: Compares two URLs and returns a confidence score and match flag indicating similarity.

#### Request Body (JSON)
```json
{
  "original_url": "https://betterfans.app/sv1/3684516978?page=AlayaPaid&folder=SEXT4(Day)PinkBows&amp;Hearts",
  "comparison_url": "https://betterfans.app/sv1/3742963494?page=AlixFree&folder=Sext1-(Day)MidnightBlueLaceLingerie"
}
```

#### Response
```json
{
  "confidence_score": 36.44,
  "match": false
}
```

---

### 5. 🚫 Detect Nudity in an Image

**Endpoint**: `POST /detect_nudity`  
**URL**: `https://only-search-ai-275499389350.us-central1.run.app/detect_nudity`

**Description**: Analyzes an image for potential nudity and returns object detection results with classes, confidence scores, and bounding boxes.

#### Request Body (JSON)
```json
{
  "image_url": "https://betterfans.app/sv1/3684516978?page=AlayaPaid&folder=SEXT4(Day)PinkBows&amp;Hearts"
}
```

#### Response
```json
{
  "confidence_scores": [
    {
      "box": [609, 345, 562, 636],
      "class": "FACE_FEMALE",
      "score": 0.8920111656188965
    },
    {
      "box": [16, 1244, 595, 708],
      "class": "FEMALE_BREAST_COVERED",
      "score": 0.7720419764518738
    },
    {
      "box": [513, 1386, 739, 660],
      "class": "FEMALE_BREAST_COVERED",
      "score": 0.6877492070198059
    }
  ],
  "nudity": false
}
```

---

### 6. 🧐 Get Face Coordinates

**Endpoint**: `POST /face_coordinates`  
**URL**: `https://only-search-ai-275499389350.us-central1.run.app/face_coordinates`

**Description**: Detects faces in the image and returns their bounding box coordinates.

#### Request Body (JSON)
```json
{
  "image_url": "https://betterfans.app/sv1/3684516978?page=AlayaPaid&folder=SEXT4(Day)PinkBows&amp;Hearts"
}
```

#### Response
```json
{
  "faces": [
    {
      "bottom": 911,
      "height": 554,
      "left": 603,
      "right": 1157,
      "top": 357,
      "width": 554
    }
  ]
}
```

---

### 7. 📝 Analyze About Section

**Endpoint**: `POST /analyze-about`  
**URL**: `https://only-search-ai-275499389350.us-central1.run.app/analyze-about`

**Description**: Analyzes the 'about' section of a profile using GPT-4o-mini to determine the gender of the user. Also supports fallback that uses vision.

If `about` can't be processed to get the `gender`, it will use the `image_url` as fallback and return based on the image

> Added fallabck due to:
> 1. Some users not having `about` not having any gender indicators or empty values
> 2. Instances with some users not having any face being shown in the `avatar` 

#### image_url that doesn't exists or are TOO NSFW may be blocked by openai itself

#### Request Body (JSON)
```json
{
  "about": "*We both share this account <br />\nInterracial 30’s Couple. <br />\nFit athletic  8” BBC Stud🍆 sharing nice content with Fit-Thick Pawg 👩🏻💦.<br />\nBlowjobs, Handjobs, Creampie, hardcore, POV, Squirting, Cumshots &amp; More. Join us to see😈💦<br />\nPLEASE READ❗️Any institutions or individuals making use of this site and its members/participants, or any associated sites for academic studies or projects are given notice: this profile is the copyright of the author. I DO NOT GRANT YOU ANY PERMISSION TO USE ANY OF THIS PROFILE, its content, its images, its video files or any other component, in whole or in part, in any form, extracted or not, for any purpose or forum whatsoever, either now or in perpetuity. If you have or do, it will be considered a violation of copyright and privacy, and will be subject to legal ramifications.",
  "image_url": "https://public.onlyfans.com/files/r/rd/rdc/rdcmjmu84cgnln4ocnfvktlj2agbdxno1702031311/267479658/avatar.jpg"
}
```

#### Response Sample for text_analysis
```json
{
    "gender": "female",
    "source": "text_analysis",
    "thoughts": "[ANALysis]\nIn analyzing the provided text, several phrases and words suggest gender indicators. The account describes itself as belonging to a couple, using the term \"interracial,\" which doesn't directly indicate gender but establishes a relationship dynamic. The phrases \"Fit athletic 8” BBC Stud\" and \"Fit-Thick Pawg\" contain specific gender-related terms. \"BBC\" stands for \"Big Black Cock,\" which is a term predominantly associated with male genitalia and indicates the presence of a male partner..."
}
```

#### Response Sample for text_analysis
``

**Possible values**: `male`, `female`, or `unknown`

---

## 🛡️ Authentication

All requests must include an API key in the headers:

```http
X-API-KEY: your_api_key
```
