## 📱 API Endpoints Documentation

All endpoints require authentication using an API key passed via the request headers:

```http
X-API-KEY: your_api_key_here
```

---

### 1. 🔍 Compare Two URLs for Similarity

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

### 2. 🚫 Detect Nudity in an Image

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

### 3. 🧐 Get Face Coordinates

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

## 🛡️ Authentication

All requests must include an API key in the headers:

```http
X-API-KEY: your_api_key
```
