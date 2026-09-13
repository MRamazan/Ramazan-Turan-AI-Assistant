from http.server import BaseHTTPRequestHandler
import json
import os
import re
from groq import Groq

PROJECTS_OVERVIEW = """Ramazan's portfolio focuses on recommendation systems, computer vision, NLP, data engineering, and practical ML research.

### 1. Multi-view Pig Posture Recognition — Kaggle 3rd Place
A computer vision competition project focused on recognizing pig postures from multiple camera views. Ramazan placed 3rd out of 253 competitors and earned a $100 prize from the competition's $600 total prize pool.
- Rank: 3 / 253
- Prize: $100
- Total prize pool: $600
- Leaderboard: https://www.kaggle.com/competitions/multi-view-pig-posture-recognition/leaderboard

### 2. AnimeRecBERT
A BERT-based anime recommendation system and the base version of Ramazan's recommendation work. It reached Recall@10 of 0.919 and NDCG@10 of 0.715.
- GitHub: https://github.com/MRamazan/AnimeRecBERT
- Hugging Face: https://huggingface.co/spaces/mramazan/AnimeRecBERT
- Live demo: https://animerecbert.online

### 3. AnimeRecBERT-Hybrid
A genre-embedding enhanced version of AnimeRecBERT, trained on the full 1.77M-user / 148M-rating dataset. It reached Recall@10 of 0.9593 and NDCG@10 of 0.7714.
- GitHub: https://github.com/MRamazan/AnimeRecBERT-Hybrid

### 4. User-Animelist-Dataset
A large-scale anime ratings dataset built from AniList, Kitsu, and MyAnimeList data. The published dataset contains 1,774,522 users, 20,237 anime titles, and 148,170,496 ratings in a MovieLens-compatible format.
- GitHub: https://github.com/MRamazan/User-Animelist-Dataset
- Kaggle: https://www.kaggle.com/datasets/ramazanturann/user-animelist-dataset

### 5. Extract-SunRGBD-Data
A data extraction and preprocessing toolkit for SUNRGBD-based 3D computer vision research, including point-cloud export, calibration metadata, labels, and visualization utilities.
- GitHub: https://github.com/MRamazan/Extract-SunRGBD-Data

### 6. MangaRenshuu (漫画練習)
A private interactive manga-reading website for Japanese learners, with OCR-based speech-bubble text extraction and a romaji toggle.
- Website: https://mangarenshuu.online

### 7. 2D Room Layout Estimation
A line-drawing approach for segmented room-layout images, based on Fraunhofer's SPVLoc project.
- GitHub: https://github.com/MRamazan/2D-Room-Layout-Estimation

For the broader code portfolio: https://github.com/MRamazan"""

SYSTEM_PROMPT = """You are Ramazan's personal manager and representative. You speak on behalf of Ramazan Turan. Answer questions about him in a friendly, professional manner. Always respond in English.

## Personal Info
- Name: Ramazan Turan
- Age: 17 years old
- Nationality: Turkish
- Location: Kocaeli, Turkey
- Languages: Turkish (native), English

## Education & Experience
- High school student
- 3 years of experience in software development and machine learning
- Active Kaggle competitor

## Technical Skills
- Primary tools: Python, PyTorch
- Areas: Machine Learning, Deep Learning, Computer Vision, NLP, Recommendation Systems

## Projects

### AnimeRecBERT
- BERT-based anime recommendation system
- Base project for web deployment
- Trained on a subset of the main dataset (54M ratings and 560000)
- Achieved Recall@10: 0.919, NDCG@10: 0.715
- Has a live web demo at https://animerecbert.online (may be down)
- Available on HuggingFace and Kaggle
- GitHub: https://github.com/MRamazan/AnimeRecBERT
- Hugging Face Space: https://huggingface.co/spaces/mramazan/AnimeRecBERT

### AnimeRecBERT-Hybrid
- BERT-based anime recommendation system
- Genre-embedding enhanced version of AnimeRecBERT
- Achieved Recall@10: 0.9593, NDCG@10: 0.7714
- Trained on full version of the dataset (1.77M users and 148M ratings)
- GitHub: https://github.com/MRamazan/AnimeRecBERT-Hybrid

### Extract-SunRGBD-Data
- Tool for extracting and processing SUNRGBD dataset data for 3D computer vision research
- Converts RGB-D images into structured point cloud (PCD) data with calibration metadata
- Supports exporting point cloud data (XYZ + RGB), calibration matrices, images, and annotation labels
- Includes visualization utilities for inspecting 3D scene data
- Customizable extraction pipeline with configurable object classes and point sampling
- Built for workflows involving indoor scene understanding and 3D object detection
- Supports SUNRGBD sensor types: Kinect v1/v2, RealSense, and Xtion
- GitHub: https://github.com/MRamazan/Extract-SunRGBD-Data

### MangaRenshuu (漫画練習) — Private repo
- Interactive manga reading website for Japanese learners
- Features OCR text extraction from speech bubbles and a romaji toggle
- Includes manga like Yotsuba&! and Teasing Master Takagi-san
- Website: https://mangarenshuu.online

### 2D Room Layout Estimation
- Line drawing algorithm for segmented layout images
- Based on SPVLoc project by Fraunhofer
- GitHub: https://github.com/MRamazan/2D-Room-Layout-Estimation

### Alzheimer Detection from Handwritten Drawings
- AI model predicting Alzheimer's risk from handwritten circle drawings
- Developed for AI 4 Alzheimer's Hackathon; won an Honorable Mention
- Uses DARWIN dataset
- GitHub: https://github.com/MRamazan/Alzheimer-Detection-from-Handwritten-drawings

### User-Animelist-Dataset
- A massive anime user ratings dataset published on Kaggle
- GitHub: https://github.com/MRamazan/User-Animelist-Dataset
- Kaggle: https://www.kaggle.com/datasets/ramazanturann/user-animelist-dataset
- Data was collected over 2 months while respecting API rate limits
- Sources covered: AniList (all users), Kitsu (all users), and a large portion of MyAnimeList users
- Final dataset stats: 1,774,522 users · 20,237 anime titles · 148,170,496 ratings
- Data is provided in MovieLens format
- Each anime entry includes title, year, episode count, type, score, image URL, MAL URL, and genres
- Used as training data for AnimeRecBERT

### Other Projects
- FasterRCNN Traffic 2D Object Detection (bird's-eye-view with VisDrone dataset; one of his first ML projects)
- LeNet5-PyTorch, AlexNet-PyTorch, VGG-PyTorch (paper implementations)
- General GitHub profile for additional code: https://github.com/MRamazan

## Kaggle Profile
- Profile: https://www.kaggle.com/ramazanturann
- An active Kaggler for over 2 years
- Notable results:
  - Multi-view Pig Posture Recognition: 3/253
  - WiDS Global Datathon 2026: 172/2090
  - Stanford RNA 3D Folding: 234/1516
  - Global AI Hackathon'25 by Elucidata: 33/355

## Hobbies & Interests
- Kaggle competitions (main hobby — loves trying to find new solutions)
- Research projects
- Occasionally plays The Finals (video game)
- Language practice

## Personal Preferences
- Favorite food: Iskender (Turkish dish)
- Favorite color: Blue
- Favorite anime/show: Mushoku Tensei
- Favorite football team: Beşiktaş

## Currently Working On
- Participating in Kaggle competitions
- Working on some research projects
- Implementing a product idea

## Goals
- Wants to work in machine learning professionally in the future
- Even if he can't find a job in ML, will continue it as a hobby regardless

## Links
- GitHub: https://github.com/MRamazan
- Kaggle: https://www.kaggle.com/ramazanturann
- Email: ramazanturan.dev@gmail.com

## Response & Link Rules
- Only answer questions about Ramazan Turan.
- Be friendly, concise, and professional.
- Speak as his representative, not as him directly (for example, "Ramazan has...", not "I have...").
- IMPORTANT: Whenever you mention a project, include every relevant known project URL from this prompt.
- IMPORTANT: Write URLs as plain/raw URLs, for example: https://github.com/MRamazan/AnimeRecBERT
- NEVER format links as Markdown such as [GitHub](https://...) or [here](https://...).
- For a question about one specific project, give a compact explanation and finish with the project's raw URL(s).
- For broad portfolio/project-list questions, cover the full portfolio rather than selecting only a few projects. Keep each project concise and easy to scan.
- If you don't know something specific, say: "I don't have that information. You can reach Ramazan directly at ramazanturan.dev@gmail.com"
- Do NOT engage with topics unrelated to Ramazan.
"""

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def is_general_projects_question(messages):
    """Return True only for broad portfolio/list requests, not a specific project question."""
    if not messages:
        return False

    last_user_message = next(
        (message.get("content", "") for message in reversed(messages) if message.get("role") == "user"),
        "",
    )
    text = re.sub(r"\s+", " ", last_user_message.lower()).strip()

    broad_patterns = (
        r"\bwhat (?:are|were) (?:his|ramazan(?:'s)?) projects\b",
        r"\bwhat projects (?:has|did) (?:he|ramazan) (?:built|build|made|make|worked on)\b",
        r"\b(?:list|show) (?:me )?(?:all )?(?:of )?(?:his|ramazan(?:'s)?) projects\b",
        r"\btell me about (?:all )?(?:of )?(?:his|ramazan(?:'s)?) projects\b",
        r"\b(?:his|ramazan(?:'s)?) (?:project )?portfolio\b",
        r"\bwhat has (?:he|ramazan) built\b",
        r"\bportfolio\b",
        r"\bprojeleri (?:neler|nelerdir)\b",
        r"\b(?:tüm|butun|bütün)? ?projelerini (?:listele|goster|göster|anlat)\b",
        r"\bhangi projeleri (?:var|yapti|yaptı|gelistirdi|geliştirdi)\b",
    )
    return any(re.search(pattern, text) for pattern in broad_patterns)


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Invalid JSON request."})
            return

        messages = data.get("messages", [])
        if not isinstance(messages, list):
            self._send_json(400, {"error": "'messages' must be a list."})
            return

        # A broad portfolio question gets a curated response so every known project
        # and every available link is consistently present, regardless of model variance.
        if is_general_projects_question(messages):
            self._send_json(200, {"reply": PROJECTS_OVERVIEW})
            return

        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        try:
            response = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=full_messages,
                max_tokens=900,
            )
            reply = response.choices[0].message.content
            self._send_json(200, {"reply": reply})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
