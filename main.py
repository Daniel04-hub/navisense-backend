"""
NaviSense Backend — FastAPI
Run: pip install fastapi uvicorn python-multipart && uvicorn main:app --host 0.0.0.0 --port 5000 --reload
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
import uuid, time, os, json, base64, hashlib, hmac
from datetime import datetime

app = FastAPI(title="NaviSense API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

# ─── In-memory DB ────────────────────────────────────────────────
USERS = {}
SESSIONS = {}
TRIPS = {}
UPLOADS = {}

# ─── Models ──────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    email: str
    password: str
    language: str = "en"
    home_location: str = ""
    emergency_contact: str = ""

class VoiceCommand(BaseModel):
    command: str
    lang: str = "en"
    context: Optional[dict] = {}

class DetectionRequest(BaseModel):
    image_base64: str
    mode: str = "object"

class NavigateRequest(BaseModel):
    origin_lat: float
    origin_lng: float
    dest_lat: float
    dest_lng: float
    destination_name: str = ""

# ─── Auth Helper ─────────────────────────────────────────────────
def make_token(user_id: str) -> str:
    payload = f"{user_id}:{int(time.time())}"
    sig = hmac.new(b"navisense_secret", payload.encode(), hashlib.sha256).hexdigest()
    return base64.b64encode(f"{payload}:{sig}".encode()).decode()

def verify_token(token: str) -> Optional[str]:
    try:
        decoded = base64.b64decode(token.encode()).decode()
        user_id, ts, sig = decoded.rsplit(":", 2)
        expected = hmac.new(b"navisense_secret", f"{user_id}:{ts}".encode(), hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig, expected):
            return user_id
    except Exception:
        pass
    return None

def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if creds:
        uid = verify_token(creds.credentials)
        if uid and uid in USERS:
            return USERS[uid]
    return None

# ─── Endpoints ───────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"status": "ok", "service": "NaviSense API", "message": "Backend is running!"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "NaviSense API", "version": "2.0.0", "timestamp": datetime.utcnow().isoformat()}


@app.post("/register")
def register(body: RegisterRequest):
    for u in USERS.values():
        if u["email"] == body.email:
            raise HTTPException(400, "Email already registered")
    uid = str(uuid.uuid4())
    USERS[uid] = {
        "id": uid,
        "email": body.email,
        "name": f"{body.first_name} {body.last_name}",
        "language": body.language,
        "home_location": body.home_location,
        "emergency_contact": body.emergency_contact,
        "created_at": datetime.utcnow().isoformat(),
        "password_hash": hashlib.sha256(body.password.encode()).hexdigest(),
    }
    token = make_token(uid)
    return {"success": True, "token": token, "user": {k: v for k, v in USERS[uid].items() if k != "password_hash"}}


@app.post("/login")
def login(body: LoginRequest):
    for uid, u in USERS.items():
        if u["email"] == body.email:
            pw_hash = hashlib.sha256(body.password.encode()).hexdigest()
            if pw_hash == u["password_hash"]:
                token = make_token(uid)
                return {"success": True, "token": token, "user": {k: v for k, v in u.items() if k != "password_hash"}}
    raise HTTPException(401, "Invalid email or password")


@app.post("/upload")
async def upload_dataset(files: List[UploadFile] = File(...)):
    results = []
    for f in files:
        content = await f.read()
        fid = str(uuid.uuid4())
        UPLOADS[fid] = {
            "id": fid,
            "filename": f.filename,
            "content_type": f.content_type,
            "size": len(content),
            "uploaded_at": datetime.utcnow().isoformat(),
            "status": "ready",
        }
        results.append({"id": fid, "filename": f.filename, "size": len(content), "status": "ready"})
    return {"success": True, "files": results, "total": len(results)}


@app.get("/dataset-stats")
def dataset_stats():
    total_size = sum(u["size"] for u in UPLOADS.values())
    images = [u for u in UPLOADS.values() if u["content_type"].startswith("image/")]
    videos = [u for u in UPLOADS.values() if u["content_type"].startswith("video/")]
    return {
        "total_files": len(UPLOADS),
        "images": len(images),
        "videos": len(videos),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / 1_048_576, 2),
    }


@app.post("/run-detection")
def run_detection(body: DetectionRequest):
    """Simulates YOLOv8 + OCR + Traffic detection."""
    import random
    mode = body.mode
    if mode == "object":
        classes = ["person","car","truck","bicycle","motorcycle","bus","traffic light","stop sign","obstacle","dog"]
        dets = []
        count = random.randint(1, 5)
        for _ in range(count):
            cls = random.choice(classes)
            conf = round(random.uniform(0.72, 0.97), 2)
            dist = round(random.uniform(1.5, 15.0), 1)
            dets.append({"class": cls, "confidence": conf, "distance_m": dist, "bbox": [random.randint(50,400), random.randint(50,300), random.randint(60,200), random.randint(60,200)]})
        return {"mode": "object", "detections": dets, "count": len(dets), "processing_ms": random.randint(40, 120)}
    elif mode == "ocr":
        samples = ["CAUTION WET FLOOR", "BUS STOP 14B", "EXIT →", "PLATFORM 2", "NO ENTRY", "SLOW DOWN", "HOSPITAL AHEAD"]
        return {"mode": "ocr", "text": random.choice(samples), "confidence": round(random.uniform(0.75, 0.95), 2), "processing_ms": random.randint(80, 200)}
    elif mode == "currency":
        notes = [10, 20, 50, 100, 200, 500, 2000]
        val = random.choice(notes)
        return {"mode": "currency", "denomination": val, "currency": "INR", "authentic": random.random() > 0.05, "confidence": round(random.uniform(0.80, 0.98), 2)}
    elif mode == "traffic":
        states = ["RED", "GREEN", "YELLOW", "UNKNOWN"]
        state = random.choice(states)
        return {"mode": "traffic", "signal_state": state, "confidence": round(random.uniform(0.78, 0.96), 2), "vehicles_detected": random.randint(0, 8)}
    return {"error": "Unknown mode"}


@app.post("/voice-command")
def voice_command(body: VoiceCommand):
    """NLP intent classification + response generation."""
    cmd = body.command.lower().strip()
    
    intents = {
        ("go forward", "move", "walk", "proceed"): ("NAVIGATE", "Path ahead appears clear. Proceed forward with caution."),
        ("stop", "halt", "freeze", "wait"): ("STOP", "Stopping navigation. Stay still."),
        ("what", "describe", "see", "front", "around"): ("DESCRIBE", "Scanning environment. I can see a footpath ahead with moderate pedestrian activity."),
        ("read", "text", "sign", "board"): ("OCR", "Reading text in view: BUS STOP 14B, 50 meters ahead."),
        ("currency", "money", "note", "rupee"): ("CURRENCY", "Currency detection active. Place banknote in front of camera."),
        ("battery", "charge", "power"): ("BATTERY", f"Battery at 78 percent. Approximately 4 hours of navigation remaining."),
        ("navigate", "route", "go to", "take me"): ("NAVIGATE", "Setting route. Turn left in 80 meters, then proceed straight for 200 meters."),
        ("help", "commands", "what can"): ("HELP", "You can say: go forward, stop, what do you see, read text, check battery, navigate to, or ask me anything."),
        ("weather", "rain", "temperature"): ("WEATHER", "Current conditions: partly cloudy, 32 degrees Celsius. Light breeze from southwest."),
        ("emergency", "danger", "alert", "sos"): ("EMERGENCY", "Emergency alert activated. Notifying emergency contact. Stay calm."),
    }
    
    intent = "UNKNOWN"
    response = "I heard you. Please repeat your command clearly."
    
    for keywords, (intent_name, reply) in intents.items():
        if any(k in cmd for k in keywords):
            intent = intent_name
            response = reply
            break
    
    return {
        "input": body.command,
        "intent": intent,
        "response": response,
        "lang": body.lang,
        "confidence": round(0.80 + (len(cmd) % 20) / 100, 2),
    }


@app.post("/navigate")
def navigate(body: NavigateRequest):
    """Simulates OSRM routing with Haversine distance."""
    import math
    lat1, lon1 = math.radians(body.origin_lat), math.radians(body.origin_lng)
    lat2, lon2 = math.radians(body.dest_lat), math.radians(body.dest_lng)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    dist_km = round(6371 * 2 * math.asin(math.sqrt(a)), 2)
    
    steps = [
        {"icon": "↑", "instruction": "Head north on current road", "distance": "120m"},
        {"icon": "←", "instruction": "Turn left at intersection", "distance": "80m"},
        {"icon": "↑", "instruction": "Continue straight past the market", "distance": "200m"},
        {"icon": "→", "instruction": f"Turn right toward {body.destination_name or 'destination'}", "distance": "60m"},
        {"icon": "★", "instruction": f"Arrive at {body.destination_name or 'destination'}", "distance": "0m"},
    ]
    return {
        "distance_km": dist_km,
        "duration_min": round(dist_km * 12),
        "steps": steps,
        "geohash": "tf0fz9",
    }


@app.get("/trips")
def get_trips(user=Depends(get_current_user)):
    uid = user["id"] if user else "guest"
    user_trips = [t for t in TRIPS.values() if t.get("user_id") == uid]
    return {"trips": user_trips}


@app.get("/results")
def get_results():
    return {"detections": [], "message": "No recent detections stored"}


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
