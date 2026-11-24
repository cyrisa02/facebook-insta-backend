import os
import base64
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List

from groq import Groq
import instructor
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Atelier Seats AI API")

# --- CONFIGURATION CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:3000",
    "https://facebook-insta-front.vercel.app",  # URL exacte, sans slash final
],  # frontend Next.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION CLIENT (VIA .ENV) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Variable d'environnement GROQ_API_KEY manquante")

# --- DOSSIER UPLOADS (service statique) ---
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# --- SCHÉMA PYDANTIC POUR LA GÉNÉRATION DE POSTS ---
class PostResult(BaseModel):
    facebook: str = Field(description="Texte pour Facebook, ton chaleureux et professionnel, 150 mots max, avec emojis")
    instagram: str = Field(description="Texte pour Instagram, ton visuel et inspirant, 100 mots max, avec hashtags")

# --- CLIENT GROQ (même logique que votre code existant) ---
client = instructor.from_groq(
    Groq(api_key=GROQ_API_KEY),
    mode=instructor.Mode.JSON
)

# --- PROMPT SYSTÈME ---
SYSTEM_PROMPT = """Tu es un expert en communication digitale spécialisé dans l'artisanat d'art, la sellerie automobile, moto et médicale, et aussi l'ameublement, c'est-à-dire réparation de fauteuils. 
Ton client est SELLERIE TEXTURES, un artisan d'excellence travaillant exclusivement le cuir et le similicuir et de temps en temps le tissu.

ÉTAPE 1 : ANALYSE VISUELLE OBLIGATOIRE (Chain of Thought)
Avant de rédiger, analyse l'image fournie avec une extrême précision. Identifie formellement l'élément restauré ou créé (ex: capote souple de cabriolet, siège auto, volant, ciel de toit, sellerie moto, table médicale). 
Règle d'or : Ne fais aucune supposition. Si l'image montre une capote, tu dois parler de capote. Si elle montre un siège, tu parles de siège. Décris brièvement ce que tu vois pour valider ta compréhension.

ÉTAPE 2 : RÉDACTION DES POSTS
Rédige deux textes en te basant strictement sur l'élément identifié à l'étape 1.

1. Facebook : 
- Ton chaleureux, professionnel, mettant en avant la qualité du travail du cuir ou du similicuir, ou du tissu (en prinicpe une seule matière est utilisée) et l'expertise de SELLERIE TEXTURES. 
- Utilise le "Je", pas le nous, je suis seul.
- 150 mots maximum, avec des emojis pertinents.

2. Instagram : 
- Ton plus visuel, inspirant et premium. Mets en valeur la noblesse de la matière (cuir/similicuir) et le savoir-faire. 
- Utilise le "Je", pas le nous, je suis seul.
- 100 mots maximum, avec des hashtags pertinents (#SELLERIETEXTURES #cuir #similicuir #artisanat #avantapres #sellerieauto #restauration #capote).

RÈGLES STRICTES À RESPECTER :
- Intègre le nom "SELLERIE TEXTURES" naturellement dans les textes (pas de crochets ni de placeholders).
- Adopte un ton fier, valorisant le geste artisanal et donnant envie aux clients de contacter l'atelier pour leurs projets."""

# --- HELPER : sauvegarde des images sur disque ---
async def save_images(files: List[UploadFile]) -> List[str]:
    urls = []
    for file in files:
        ext = Path(file.filename or "photo.jpg").suffix or ".jpg"
        filename = f"{uuid.uuid4().hex}{ext}"
        dest = UPLOAD_DIR / filename
        content = await file.read()
        dest.write_bytes(content)
        urls.append(f"/uploads/{filename}")
    return urls

# --- HELPER : conversion fichier -> data URI base64 pour Qwen ---
async def file_to_data_uri(file: UploadFile) -> str:
    content = await file.read()
    await file.seek(0)  # permet de relire le fichier pour la sauvegarde
    b64 = base64.b64encode(content).decode("utf-8")
    mime = file.content_type or "image/jpeg"
    return f"data:{mime};base64,{b64}"

# --- ENDPOINT PRINCIPAL : GÉNÉRATION DES TEXTES ---
@app.post("/generate-posts")
async def generate_posts(
    photos: List[UploadFile] = File(...),
    contexte: str = Form(default="Réparation de sièges de voiture"),
):
    try:
        if not photos:
            raise HTTPException(status_code=400, detail="Aucune photo fournie")
        if len(photos) > 4:
            raise HTTPException(status_code=400, detail="Maximum 4 photos")

        # Qwen accepte jusqu'à 3 images : on envoie les 3 premières à l'IA,
        # mais on sauvegarde TOUTES les photos (pour un futur multi-images)
        ai_photos = photos[:3]

        image_parts = []
        for file in ai_photos:
            data_uri = await file_to_data_uri(file)
            image_parts.append({"type": "image_url", "image_url": {"url": data_uri}})

        # Génération avec instructor (même pattern que /extract et /correct-copy)
        result = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            response_model=PostResult,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Contexte : {contexte}. Rédige les deux posts."},
                        *image_parts,
                    ],
                },
            ],
            max_tokens=900,
            temperature=0.1, 
        )

        # Sauvegarde de toutes les photos (les 3 envoyées à l'IA + celles en plus)
        saved_urls = await save_images(photos)

        return {
            "facebook": result.facebook,
            "instagram": result.instagram,
            "images": saved_urls,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))