from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import pickle
import os
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime

app = FastAPI(title="Student Risk Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load data & model at startup ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE, "data/students.json")) as f:
    STUDENTS: list = json.load(f)

with open(os.path.join(BASE, "model/model.pkl"), "rb") as f:
    MODEL = pickle.load(f)

with open(os.path.join(BASE, "model/feature_importance.json")) as f:
    FEATURE_IMPORTANCE = json.load(f)

with open(os.path.join(BASE, "model/feature_labels.json")) as f:
    FEATURE_LABELS = json.load(f)

with open(os.path.join(BASE, "model/metrics.json")) as f:
    MODEL_METRICS = json.load(f)

# Load SHAP samples
shap_path = os.path.join(BASE, "model/shap_samples.json")
SHAP_SAMPLES = []
if os.path.exists(shap_path):
    with open(shap_path) as f:
        SHAP_SAMPLES = json.load(f)

# Load or init interventions log
interventions_path = os.path.join(BASE, "data/interventions.json")
if os.path.exists(interventions_path):
    with open(interventions_path) as f:
        INTERVENTIONS_LOG: list = json.load(f)
else:
    INTERVENTIONS_LOG = []

FEATURES = [
    "attendance_rate", "attendance_trend", "avg_score", "score_trend",
    "assignments_ontime_pct", "avg_submission_delay", "library_visits_per_week",
    "failed_subjects", "sgpa", "sgpa_trend", "days_inactive", "backlogs",
    "first_gen", "is_rural", "has_scholarship"
]

DF = pd.read_csv(os.path.join(BASE, "data/students.csv"))
FEATURE_MEANS = DF[FEATURES].mean().to_dict()
FEATURE_STDS  = DF[FEATURES].std().to_dict()

# Demo advisor credentials
DEMO_ADVISORS = {
    "advisor1": {"password": "edusense2025", "name": "Dr. Ramachandran S.", "role": "Faculty Advisor"},
    "advisor2": {"password": "edusense2025", "name": "Dr. Priya Nair", "role": "Faculty Advisor"},
    "admin": {"password": "admin123", "name": "Admin User", "role": "Administrator"},
}

# ── Pydantic Models ──────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class InterventionRequest(BaseModel):
    type: str
    notes: Optional[str] = ""
    advisor: Optional[str] = "Faculty Advisor"

class OpportunityRequest(BaseModel):
    category: str
    title: str
    description: str
    deadline: str

# ── Opportunities seed data ───────────────────────────────────────────────────
OPPORTUNITIES: list = [
    {
        "id": 1,
        "category": "Hackathon",
        "title": "Smart India Hackathon 2026 \u2014 CSE Edition",
        "description": "National-level hackathon for CSE students. Build innovative solutions for real-world government problems. Teams of 6, mentored by industry experts. Winners receive \u20b91,00,000 prize and internship offers.",
        "deadline": "2026-06-15",
    },
    {
        "id": 2,
        "category": "Scholarship",
        "title": "Merit-cum-Means Engineering Scholarship",
        "description": "Scholarship for CSE students with SGPA \u2265 7.5 and annual family income below \u20b98,00,000. Covers full tuition and provides \u20b95,000/month stipend for one academic year.",
        "deadline": "2026-05-30",
    },
    {
        "id": 3,
        "category": "Funding",
        "title": "AI Research Seed Grant for Final Year Projects",
        "description": "Seed funding of up to \u20b950,000 for final-year CSE students pursuing AI/ML research projects. Includes access to GPU compute resources and faculty mentorship program.",
        "deadline": "2026-07-01",
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def student_to_features(s: dict) -> dict:
    return {
        "attendance_rate": s["attendance_rate"],
        "attendance_trend": s["attendance_trend"],
        "avg_score": s["avg_score"],
        "score_trend": s["score_trend"],
        "assignments_ontime_pct": s["assignments_ontime_pct"],
        "avg_submission_delay": s["avg_submission_delay"],
        "library_visits_per_week": s["library_visits_per_week"],
        "failed_subjects": s["failed_subjects"],
        "sgpa": s["sgpa"],
        "sgpa_trend": s["sgpa_trend"],
        "days_inactive": s["days_inactive"],
        "backlogs": s["backlogs"],
        "first_gen": int(s["first_gen"]),
        "is_rural": int(s["origin"] == "Rural"),
        "has_scholarship": int(s["scholarship"] != "None"),
    }

def compute_contributions(feat_dict: dict) -> list:
    """Manual SHAP-style contributions per student."""
    protective = {
        "attendance_rate", "avg_score", "assignments_ontime_pct",
        "library_visits_per_week", "sgpa", "sgpa_trend",
        "attendance_trend", "score_trend",
    }
    contribs = []
    for feat in FEATURES:
        val  = feat_dict.get(feat, FEATURE_MEANS[feat])
        mean = FEATURE_MEANS[feat]
        std  = FEATURE_STDS[feat] if FEATURE_STDS[feat] > 0 else 1
        z    = (val - mean) / std
        direction = -1 if feat in protective else 1
        impact = round(float(FEATURE_IMPORTANCE.get(feat, 0)) * z * direction, 4)
        contribs.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "value": round(val, 2),
            "impact": impact,
            "direction": "risk" if impact > 0 else "protective",
        })
    contribs.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return contribs[:8]

def intervention_router(s: dict) -> list:
    interventions = []
    if s["attendance_rate"] < 60:
        interventions.append({
            "type": "Academic Counselor",
            "reason": f"Attendance critically low at {s['attendance_rate']}%",
            "priority": "High",
            "icon": "🎓"
        })
    if s["sgpa"] < 5.5 or s["failed_subjects"] >= 2:
        interventions.append({
            "type": "Peer Tutoring",
            "reason": f"SGPA {s['sgpa']} with {s['failed_subjects']} failed subjects",
            "priority": "High",
            "icon": "📚"
        })
    if s["first_gen"] and s["scholarship"] == "None":
        interventions.append({
            "type": "Financial Aid Counseling",
            "reason": "First-gen student with no scholarship enrolled",
            "priority": "Medium",
            "icon": "💰"
        })
    if s["days_inactive"] > 10:
        interventions.append({
            "type": "Mental Health Referral",
            "reason": f"Inactive for {s['days_inactive']} days — possible disengagement",
            "priority": "Medium",
            "icon": "🧠"
        })
    if s["backlogs"] >= 3:
        interventions.append({
            "type": "Academic Re-planning",
            "reason": f"{s['backlogs']} backlogs may prevent semester promotion",
            "priority": "High",
            "icon": "📋"
        })
    if s.get("scholarship") != "None" and s["sgpa"] < 6.0:
        interventions.append({
            "type": "Scholarship Information",
            "reason": f"Scholarship at risk — SGPA {s['sgpa']} below renewal threshold",
            "priority": "Medium",
            "icon": "🎫"
        })
    if not interventions:
        interventions.append({
            "type": "Routine Check-in",
            "reason": "Student performing well — schedule monthly advisory",
            "priority": "Low",
            "icon": "✅"
        })
    return interventions

# ── Routes ────────────────────────────────────────────────────────────────────

# ── Auth ──

@app.post("/api/login")
def login(req: LoginRequest):
    user = DEMO_ADVISORS.get(req.username)
    if not user or user["password"] != req.password:
        raise HTTPException(401, "Invalid credentials")
    return {
        "success": True,
        "username": req.username,
        "name": user["name"],
        "role": user["role"],
        "token": f"demo-token-{req.username}",
    }

# ── Students ──

@app.get("/api/students")
def get_students(
    risk_level: Optional[str] = None,
    department: Optional[str] = None,
    search: Optional[str] = None,
):
    result = STUDENTS
    if risk_level and risk_level != "All":
        result = [s for s in result if s["risk_level"] == risk_level]
    if department and department != "All":
        result = [s for s in result if s["department"] == department]
    if search:
        q = search.lower()
        result = [s for s in result if q in s["name"].lower() or q in s["student_id"].lower()]
    # Return summary fields only
    summary = []
    for s in result:
        summary.append({
            "student_id": s["student_id"],
            "name": s["name"],
            "department": s["department"],
            "semester": s["semester"],
            "gender": s["gender"],
            "risk_score": s["risk_score"],
            "risk_level": s["risk_level"],
            "attendance_rate": s["attendance_rate"],
            "sgpa": s["sgpa"],
            "failed_subjects": s["failed_subjects"],
            "backlogs": s["backlogs"],
        })
    summary.sort(key=lambda x: x["risk_score"], reverse=True)
    return summary

@app.get("/api/students/{student_id}")
def get_student(student_id: str):
    s = next((x for x in STUDENTS if x["student_id"] == student_id), None)
    if not s:
        raise HTTPException(404, "Student not found")
    feat = student_to_features(s)
    contributions = compute_contributions(feat)
    interventions = intervention_router(s)
    # Include runtime interventions
    runtime_interventions = [iv for iv in INTERVENTIONS_LOG if iv["student_id"] == student_id]
    return {
        **s,
        "contributions": contributions,
        "interventions": interventions,
        "logged_interventions": runtime_interventions,
    }

# ── Dashboard ──

@app.get("/api/dashboard/summary")
def dashboard_summary():
    total = len(STUDENTS)
    high   = sum(1 for s in STUDENTS if s["risk_level"] == "High")
    medium = sum(1 for s in STUDENTS if s["risk_level"] == "Medium")
    low    = sum(1 for s in STUDENTS if s["risk_level"] == "Low")

    # Dept breakdown
    from collections import defaultdict
    dept_risk = defaultdict(lambda: {"High": 0, "Medium": 0, "Low": 0, "total": 0})
    for s in STUDENTS:
        dept_risk[s["department"]][s["risk_level"]] += 1
        dept_risk[s["department"]]["total"] += 1

    # Equity breakdown
    first_gen_risk = {
        "first_gen_high": sum(1 for s in STUDENTS if s["first_gen"] and s["risk_level"] == "High"),
        "non_first_gen_high": sum(1 for s in STUDENTS if not s["first_gen"] and s["risk_level"] == "High"),
        "first_gen_total": sum(1 for s in STUDENTS if s["first_gen"]),
        "non_first_gen_total": sum(1 for s in STUDENTS if not s["first_gen"]),
    }
    gender_risk = {
        "male_high": sum(1 for s in STUDENTS if s["gender"] == "Male" and s["risk_level"] == "High"),
        "female_high": sum(1 for s in STUDENTS if s["gender"] == "Female" and s["risk_level"] == "High"),
        "male_total": sum(1 for s in STUDENTS if s["gender"] == "Male"),
        "female_total": sum(1 for s in STUDENTS if s["gender"] == "Female"),
    }
    origin_risk = {}
    for origin in ["Rural", "Urban", "Semi-Urban"]:
        tot = sum(1 for s in STUDENTS if s["origin"] == origin)
        hi  = sum(1 for s in STUDENTS if s["origin"] == origin and s["risk_level"] == "High")
        origin_risk[origin] = {"total": tot, "high": hi, "pct": round(hi/tot*100, 1) if tot else 0}

    # Scholarship equity breakdown
    scholarship_risk = {}
    for sc in ["None", "Merit", "Need-Based", "SC/ST", "OBC"]:
        tot = sum(1 for s in STUDENTS if s["scholarship"] == sc)
        hi  = sum(1 for s in STUDENTS if s["scholarship"] == sc and s["risk_level"] == "High")
        scholarship_risk[sc] = {"total": tot, "high": hi, "pct": round(hi/tot*100, 1) if tot else 0}

    # Heatmap data: dept x week risk averages
    heatmap = {}
    for s in STUDENTS:
        dept = s["department"]
        if dept not in heatmap:
            heatmap[dept] = [[] for _ in range(6)]
        for w in range(min(6, len(s.get("weekly_risks", [])))):
            heatmap[dept][w].append(s["weekly_risks"][w])
    heatmap_avg = {}
    for dept, weeks in heatmap.items():
        heatmap_avg[dept] = [round(np.mean(w)*100, 1) if w else 0 for w in weeks]

    return {
        "total": total,
        "high_risk": high,
        "medium_risk": medium,
        "low_risk": low,
        "model_metrics": MODEL_METRICS,
        "dept_breakdown": dict(dept_risk),
        "heatmap": heatmap_avg,
        "equity": {
            "first_gen": first_gen_risk,
            "gender": gender_risk,
            "origin": origin_risk,
            "scholarship": scholarship_risk,
        }
    }

@app.get("/api/departments")
def get_departments():
    depts = sorted(set(s["department"] for s in STUDENTS))
    return depts

# ── Model Metrics ──

@app.get("/api/model/metrics")
def get_metrics():
    return {**MODEL_METRICS, "feature_importance": FEATURE_IMPORTANCE, "feature_labels": FEATURE_LABELS}

# ── SHAP Samples ──

@app.get("/api/shap/samples")
def get_shap_samples():
    return SHAP_SAMPLES

# ── Interventions ──

@app.post("/api/interventions/{student_id}")
def create_intervention(student_id: str, req: InterventionRequest):
    s = next((x for x in STUDENTS if x["student_id"] == student_id), None)
    if not s:
        raise HTTPException(404, "Student not found")

    intervention = {
        "intervention_id": f"INT-{student_id}-RT{len(INTERVENTIONS_LOG)+1}",
        "student_id": student_id,
        "student_name": s["name"],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "type": req.type,
        "notes": req.notes or f"Initiated {req.type} intervention for {s['name']}",
        "advisor": req.advisor,
        "risk_at_intervention": s["risk_score"],
        "status": "Scheduled",
    }
    INTERVENTIONS_LOG.append(intervention)

    # Persist to file
    try:
        with open(interventions_path, "w") as f:
            json.dump(INTERVENTIONS_LOG, f, indent=2)
    except Exception:
        pass

    return {"success": True, "intervention": intervention}

@app.get("/api/interventions")
def get_all_interventions():
    """Get all interventions — both historical (from student data) and runtime."""
    all_ivs = []

    # Collect historical interventions from student data
    for s in STUDENTS:
        for iv in s.get("intervention_history", []):
            all_ivs.append({
                **iv,
                "student_id": s["student_id"],
                "student_name": s["name"],
                "department": s["department"],
                "risk_level": s["risk_level"],
            })

    # Add runtime interventions
    for iv in INTERVENTIONS_LOG:
        all_ivs.append(iv)

    # Sort by date descending
    all_ivs.sort(key=lambda x: x.get("date", ""), reverse=True)
    return all_ivs

@app.get("/api/interventions/effectiveness")
def get_intervention_effectiveness():
    """Aggregate intervention effectiveness stats."""
    from collections import defaultdict

    type_stats = defaultdict(lambda: {"count": 0, "total_reduction": 0, "effective": 0})

    for s in STUDENTS:
        for iv in s.get("intervention_history", []):
            t = iv["type"]
            type_stats[t]["count"] += 1
            reduction = iv.get("risk_before", 0) - iv.get("risk_after", 0)
            type_stats[t]["total_reduction"] += reduction
            if reduction > 0.05:
                type_stats[t]["effective"] += 1

    results = []
    for itype, stats in type_stats.items():
        results.append({
            "type": itype,
            "count": stats["count"],
            "avg_risk_reduction": round(stats["total_reduction"] / stats["count"] * 100, 1) if stats["count"] else 0,
            "effectiveness_rate": round(stats["effective"] / stats["count"] * 100, 1) if stats["count"] else 0,
        })

    results.sort(key=lambda x: x["effectiveness_rate"], reverse=True)

    # Overall stats
    total = sum(r["count"] for r in results)
    avg_reduction = round(sum(r["avg_risk_reduction"] * r["count"] for r in results) / total, 1) if total else 0

    return {
        "by_type": results,
        "total_interventions": total,
        "avg_risk_reduction": avg_reduction,
        "total_students_with_interventions": sum(1 for s in STUDENTS if s.get("intervention_history")),
    }

@app.get("/api/interventions/{student_id}")
def get_student_interventions(student_id: str):
    s = next((x for x in STUDENTS if x["student_id"] == student_id), None)
    if not s:
        raise HTTPException(404, "Student not found")
    historical = s.get("intervention_history", [])
    runtime = [iv for iv in INTERVENTIONS_LOG if iv["student_id"] == student_id]
    return {"historical": historical, "runtime": runtime}

# ── Opportunities ──

@app.get("/api/opportunities")
def get_opportunities():
    return OPPORTUNITIES

@app.post("/api/opportunities")
def create_opportunity(req: OpportunityRequest):
    new_id = max((o["id"] for o in OPPORTUNITIES), default=0) + 1
    opportunity = {
        "id": new_id,
        "category": req.category,
        "title": req.title,
        "description": req.description,
        "deadline": req.deadline,
    }
    OPPORTUNITIES.append(opportunity)
    return {"success": True, "opportunity": opportunity}

# ── Serve Frontend ────────────────────────────────────────────────────────────
FRONTEND_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend", "index.html")

@app.get("/{full_path:path}")
def serve_frontend(full_path: str):
    if os.path.exists(FRONTEND_PATH):
        return FileResponse(FRONTEND_PATH)
    return {"message": "ARIS API is running. Frontend not found at expected path."}
