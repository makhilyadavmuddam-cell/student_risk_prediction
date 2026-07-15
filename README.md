# EduSense — Student Academic Risk Intelligence System
## PS-03 Hackathon Project

---

## What It Does

EduSense is a real-time faculty dashboard that:
- **Predicts dropout risk** for every student 6 weeks in advance using ML
- **Explains WHY** each student is at risk (SHAP-style factor attribution)
- **Routes to interventions** automatically (tutoring, financial aid, mental health)
- **Tracks equity** — identifies if rural/first-gen students face higher risk
- Shows complete **academic profile** per student: subjects, attendance, SGPA, weekly risk trend

---

## Project Structure

```
student-risk-platform/
├── backend/
│   ├── data/
│   │   ├── generate_data.py     ← Generates 300 realistic students
│   │   ├── students.json        ← Generated student records
│   │   └── students.csv         ← ML training data
│   ├── model/
│   │   ├── train.py             ← ML training (GradientBoosting)
│   │   ├── model.pkl            ← Trained model
│   │   ├── feature_importance.json
│   │   ├── feature_labels.json
│   │   └── metrics.json         ← AUC, precision, recall
│   └── api/
│       └── main.py              ← FastAPI backend (all routes)
├── frontend/
│   └── index.html               ← Complete React app (no build needed)
├── requirements.txt
├── start.sh
└── README.md
```

---

## Setup (3 steps)

### Step 1 — Install dependencies
```bash
pip install fastapi uvicorn scikit-learn numpy pandas python-multipart
```

### Step 2 — Run the platform
```bash
cd student-risk-platform
bash start.sh
```
This will:
1. Generate 300 synthetic students with realistic academic data
2. Train the GradientBoosting classifier (AUC ~1.0 on synthetic data)
3. Start the API at `http://localhost:8000`

### Step 3 — Open the frontend
Open `frontend/index.html` in your browser directly.

> **If CORS issues**: Serve frontend via `python3 -m http.server 3000` in the frontend/ folder.

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/students` | All students (filterable by risk, dept, search) |
| `GET /api/students/{id}` | Full student profile + risk factors + interventions |
| `GET /api/dashboard/summary` | Cohort stats, dept breakdown, equity metrics |
| `GET /api/departments` | List of departments |
| `GET /api/model/metrics` | AUC, feature importance, model info |

---

## ML Model

- **Algorithm**: GradientBoostingClassifier (sklearn) with Sigmoid Calibration
- **Features**: 15 features — attendance rate/trend, SGPA/trend, assignment submission, library visits, failed subjects, backlogs, days inactive, first-gen status, rural origin, scholarship
- **Target**: Dropout (binary)
- **AUC-ROC**: ~1.0 on synthetic data (strong signal separation)
- **Explainability**: Manual SHAP-style feature contributions per student

---

## Dashboard Pages

1. **Faculty Dashboard** — Cohort overview, risk distribution pie, 6-week trend, top at-risk table
2. **Student List** — Filterable/searchable table of all 300 students
3. **Student Profile** — Individual academic deep-dive with subject scores, radar chart, risk factors, interventions
4. **Equity Analytics** — Risk rates by first-gen status, gender, geographic origin
5. **Model Metrics** — Feature importance, algorithm info, performance metrics

---

## Deliverables Met

- [x] Working faculty dashboard with risk scores and intervention recommendations
- [x] Risk model with documented AUC on held-out synthetic data
- [x] Root cause attribution (SHAP-style) for every student
- [x] Intervention routing for 5 support categories
- [x] Equity analytics across gender, first-gen, rural/urban
- [x] Synthetic dataset generation script (reproducible)
- [x] Both dashboards connected via shared student data and API

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | scikit-learn GradientBoosting + Calibration |
| Backend | FastAPI + Python |
| Frontend | React 18 + Recharts (no build step) |
| Data | pandas + numpy (synthetic generation) |
| Fonts | DM Serif Display + DM Sans + JetBrains Mono |
