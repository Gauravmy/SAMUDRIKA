# Samudrika - Deployment Guide

## Overview
Samudrika is an underwater image enhancement system with:
- **Frontend**: React 19 + TypeScript + Vite (port 3000)
- **Backend**: Flask (port 5000) with ML models and image enhancement
- **Alternative Backend**: FastAPI (for reference)

---

## Prerequisites
- **Node.js** v18+ (for frontend)
- **Python** 3.10+ (for backend)
- **pip** (Python package manager)

---

## Frontend Setup

### 1. Install Dependencies
```bash
cd /path/to/samudrika
npm install
```

### 2. Environment Variables
Create `.env.local` (or use existing):
```
MODEL_API_KEY=your_api_key_here
FLASK_ENV=development
FLASK_DEBUG=1
```

### 3. Development
```bash
npm run dev
```
Runs on `http://localhost:3000`

### 4. Production Build
```bash
npm run build
npm run preview
```

---

## Backend Setup (Flask)

### 1. Create Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
cd flask_backend
pip install -r requirements.txt
```

### 3. Run Flask Server
```bash
# Option 1: Direct
python app.py

# Option 2: Flask command
flask run

# Option 3: Explicit
python -m flask --app app run
```
Runs on `http://localhost:5000`

### 4. Test Endpoints
```bash
# Health check
curl http://127.0.0.1:5000/health

# Train linear model
curl -X POST http://127.0.0.1:5000/train/linear

# Predict
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "linear", "input": [1.5]}'

# Enhance image
curl -X POST http://127.0.0.1:5000/enhance/image \
  -F "file=@underwater_image.jpg"
```

---

## Directory Structure

```
samudrika/
├── package.json                  # Frontend dependencies
├── tsconfig.json                 # TypeScript config
├── vite.config.ts                # Vite build config
├── index.html                    # Entry point
├── index.css                     # Global styles
├── index.tsx                     # React root
├── App.tsx                       # Main app component
│
├── pages/                        # Page components
│   ├── Landing.tsx
│   ├── UploadPage.tsx
│   ├── HistoryPage.tsx
│   ├── EdgePage.tsx
│   ├── AboutPage.tsx
│   └── SettingsPage.tsx
│
├── components/                   # Reusable components
│   ├── EdgePanel.tsx
│   ├── NetworkChart.tsx
│   ├── CompareSlider.tsx
│   ├── MetricsCard.tsx
│   ├── ThreeScene.tsx
│   └── ...
│
├── flask_backend/                # Flask ML backend
│   ├── app.py                    # Main Flask app (8 endpoints)
│   ├── models_store.py           # In-memory model persistence
│   ├── requirements.txt           # Python dependencies
│   ├── test_api.py               # Test suite
│   │
│   └── utils/
│       ├── ml_utils.py           # Training: linear, multiple, logistic, dummy
│       ├── image_utils.py        # Enhancement: CLAHE, denoise, sharpen, metrics
│       ├── plot_utils.py         # Graph generation (base64)
│       └── __init__.py
│
├── backend/                      # FastAPI backend (reference)
│   ├── main.py
│   ├── utils/
│   ├── models/
│   └── requirements.txt
│
├── .env.local                    # Environment variables (not in git)
├── .gitignore                    # Git ignore patterns
├── README.md                     # Project overview
├── metadata.json                 # Project metadata
└── types.ts                      # Shared TypeScript types
```

---

## Flask Backend Endpoints

| Endpoint | Method | Purpose | Example |
|----------|--------|---------|---------|
| `/health` | GET | Server status | `curl http://localhost:5000/health` |
| `/train/linear` | POST | Linear regression (1 feature) | See test_api.py |
| `/train/multiple` | POST | Multiple linear regression | File or dummy data |
| `/train/logistic/binary` | POST | Binary classification | UMAP/CSV support |
| `/train/logistic/multiclass` | POST | Multiclass classification | 3+ classes |
| `/train/dummy` | POST | Dummy model (rule-based) | y = 2x + noise |
| `/predict` | POST | Make predictions | `{"model_name": "linear", "input": [1.5]}` |
| `/enhance/image` | POST | Enhance underwater image | Returns base64 + metrics |

---

## Testing

### Frontend Type Check
```bash
npm run type-check
```

### Backend Tests
```bash
cd flask_backend
python test_api.py
```

---

## Deployment

### Vercel (Frontend)
1. Push to GitHub
2. Connect repo to Vercel
3. Set environment variables in Vercel dashboard
4. Deploy

### Heroku/Railway (Backend)
1. Add `Procfile` for Flask:
```
web: gunicorn -w 4 -b 0.0.0.0:$PORT flask_backend.app:app
```
2. Install gunicorn: `pip install gunicorn`
3. Deploy via platform CLI

### Docker (Full Stack)
```dockerfile
# Create Dockerfile and docker-compose.yml
# See deployment platform docs
```

---

## Troubleshooting

### Port Already in Use
```bash
# Frontend (3000)
lsof -i :3000
kill -9 <PID>

# Backend (5000)
lsof -i :5000
kill -9 <PID>
```

### Module Not Found (Flask)
Ensure `flask_backend/utils/__init__.py` exists with imports.

### CORS Issues
- Flask backend has CORS enabled (`flask_cors.CORS(app)`)
- Frontend Vite proxy can be configured in `vite.config.ts`

### Type Errors
```bash
npm run type-check
# Fix any TypeScript errors
```

---

## Environment Variables

**Development (.env.local):**
```
MODEL_API_KEY=your_key
FLASK_ENV=development
FLASK_DEBUG=1
```

**Production:**
```
MODEL_API_KEY=prod_key
FLASK_ENV=production
FLASK_DEBUG=0
```

---

## Key Features

✅ **6 ML Models**: Linear, Multiple Linear, Binary Logistic, Multiclass Logistic, Dummy  
✅ **Image Enhancement**: CLAHE + color balance + denoise + sharpen + metrics (UIQM, UCIQE)  
✅ **Base64 Output**: All images and graphs as base64 for web display  
✅ **CORS Support**: Frontend-backend communication enabled  
✅ **CSV Upload**: Train models on custom datasets  
✅ **Type Safety**: Full TypeScript support frontend + Python typing  

---

## Support

- **Issues**: Check GitHub issues or create new one
- **Docs**: See `DOCUMENTATION_INDEX.md`, `FLASK_BACKEND_SUMMARY.md`, etc.
- **Tests**: Run `python test_api.py` for comprehensive API tests

---

**Last Updated**: November 22, 2025  
**Version**: 1.0.0
