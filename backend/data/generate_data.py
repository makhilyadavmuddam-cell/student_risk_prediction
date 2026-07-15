import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta

np.random.seed(42)

DEPARTMENTS = ["Computer Science", "Mechanical", "Electronics", "Civil", "Information Technology"]
GENDERS = ["Male", "Female"]
FIRST_GEN = [True, False]
ORIGINS = ["Rural", "Urban", "Semi-Urban"]
SCHOLARSHIP = ["None", "Merit", "Need-Based", "SC/ST", "OBC"]

NAMES_MALE = ["Arjun", "Ravi", "Kiran", "Suresh", "Mohan", "Vijay", "Arun", "Deepak",
               "Rahul", "Nikhil", "Sanjay", "Pradeep", "Ramesh", "Anand", "Kartik",
               "Harish", "Naveen", "Sachin", "Ajay", "Rohit", "Manoj", "Dinesh",
               "Gopal", "Balu", "Vimal", "Senthil", "Muthu", "Raja", "Kumaran", "Selvam"]
NAMES_FEMALE = ["Priya", "Meena", "Kavya", "Divya", "Lakshmi", "Ananya", "Rekha",
                 "Sunitha", "Pooja", "Nithya", "Saranya", "Revathi", "Geetha", "Padma",
                 "Sowmya", "Janani", "Keerthana", "Dharini", "Brindha", "Vaishnavi",
                 "Aishwarya", "Nandhini", "Lavanya", "Mythili", "Shanthi"]
SURNAMES = ["Kumar", "Raj", "Singh", "Sharma", "Reddy", "Nair", "Pillai", "Iyer",
            "Patel", "Gupta", "Das", "Mishra", "Verma", "Yadav", "Joshi",
            "Murugan", "Krishnan", "Venkat", "Rajan", "Pandey"]

ADVISORS = [
    "Dr. Ramachandran S.", "Dr. Priya Nair", "Prof. Karthik Iyer",
    "Dr. Sunita Sharma", "Prof. Vijay Kumar", "Dr. Meenakshi R.",
    "Prof. Anil Gupta", "Dr. Lakshmi Venkat", "Prof. Rajesh Das",
    "Dr. Anitha Pillai"
]

INTERVENTION_TYPES = [
    "Academic Counselor", "Peer Tutoring", "Financial Aid Counseling",
    "Mental Health Referral", "Academic Re-planning", "Scholarship Information"
]

def generate_intervention_history(student, at_risk):
    """Generate simulated past intervention records for a student."""
    history = []
    if not at_risk:
        # Low-risk students: 0-1 interventions (routine)
        n = np.random.choice([0, 1], p=[0.7, 0.3])
    else:
        # At-risk students: 1-4 interventions
        n = np.random.randint(1, 5)

    base_date = datetime(2025, 8, 1)
    risk_before = student["risk_score"]

    for j in range(n):
        intervention_date = base_date + timedelta(days=int(np.random.randint(0, 180)))
        itype = np.random.choice(INTERVENTION_TYPES)

        # Simulate effectiveness: interventions reduce risk by 5-25%
        reduction = np.random.uniform(0.05, 0.25) if at_risk else np.random.uniform(0.01, 0.08)
        risk_after = round(max(risk_before - reduction, 0.05), 3)

        history.append({
            "intervention_id": f"INT-{student['student_id']}-{j+1}",
            "date": intervention_date.strftime("%Y-%m-%d"),
            "type": itype,
            "risk_before": round(risk_before, 3),
            "risk_after": risk_after,
            "status": np.random.choice(["Completed", "In Progress", "Scheduled"], p=[0.6, 0.25, 0.15]),
            "advisor": student["assigned_advisor"],
            "notes": generate_intervention_note(itype),
            "effectiveness": round((risk_before - risk_after) / risk_before * 100, 1) if risk_before > 0 else 0,
        })
        risk_before = risk_after  # Cascading effect

    return history

def generate_intervention_note(itype):
    notes = {
        "Academic Counselor": "Scheduled academic counseling sessions to discuss attendance and engagement strategies.",
        "Peer Tutoring": "Matched with senior student mentor for subject-specific tutoring support.",
        "Financial Aid Counseling": "Referred to financial aid office for scholarship and emergency fund eligibility review.",
        "Mental Health Referral": "Connected with campus counseling center for wellness check-in and support.",
        "Academic Re-planning": "Revised course load and created structured backlog clearance plan.",
        "Scholarship Information": "Provided information about available merit and need-based scholarship programs.",
    }
    return notes.get(itype, "General intervention support provided.")

def generate_students(n=300):
    students = []
    for i in range(n):
        gender = np.random.choice(GENDERS, p=[0.55, 0.45])
        if gender == "Male":
            first_name = np.random.choice(NAMES_MALE)
        else:
            first_name = np.random.choice(NAMES_FEMALE)
        last_name = np.random.choice(SURNAMES)
        name = f"{first_name} {last_name}"

        first_gen = np.random.choice([True, False], p=[0.45, 0.55])
        origin = np.random.choice(ORIGINS, p=[0.35, 0.45, 0.20])
        scholarship = np.random.choice(SCHOLARSHIP, p=[0.35, 0.20, 0.15, 0.15, 0.15])
        department = np.random.choice(DEPARTMENTS)
        semester = np.random.randint(1, 7)
        age = 17 + semester + np.random.randint(0, 3)
        assigned_advisor = np.random.choice(ADVISORS)

        # Base risk score based on demographics
        base_risk = 0.15
        if first_gen:
            base_risk += 0.12
        if origin == "Rural":
            base_risk += 0.10
        if scholarship in ["Need-Based", "SC/ST"]:
            base_risk += 0.08
        if semester <= 2:
            base_risk += 0.05

        # Academic features (Kaggle-style: attendance, scores, submissions)
        at_risk = np.random.random() < base_risk + 0.10

        if at_risk:
            attendance_rate = np.random.uniform(30, 65)
            attendance_trend = np.random.uniform(-20, -5)
            avg_score = np.random.uniform(35, 60)
            score_trend = np.random.uniform(-15, -3)
            assignments_ontime_pct = np.random.uniform(20, 55)
            avg_submission_delay = np.random.uniform(3, 10)
            library_visits_per_week = np.random.uniform(0, 1.5)
            failed_subjects = np.random.randint(1, 4)
            sgpa = np.random.uniform(4.0, 6.5)
            sgpa_trend = np.random.uniform(-1.5, -0.3)
            days_inactive = np.random.randint(7, 25)
            backlogs = np.random.randint(1, 5)
            dropout = 1
            # Facility usage — at-risk students use facilities less
            lab_hours_per_week = round(np.random.uniform(0.5, 4), 1)
            sports_facility_visits = np.random.randint(0, 2)
            canteen_transactions_per_day = round(np.random.uniform(0.5, 2), 1)
        else:
            attendance_rate = np.random.uniform(65, 98)
            attendance_trend = np.random.uniform(-5, 10)
            avg_score = np.random.uniform(55, 92)
            score_trend = np.random.uniform(-3, 10)
            assignments_ontime_pct = np.random.uniform(60, 100)
            avg_submission_delay = np.random.uniform(0, 3)
            library_visits_per_week = np.random.uniform(1.5, 6)
            failed_subjects = np.random.randint(0, 2)
            sgpa = np.random.uniform(6.0, 9.5)
            sgpa_trend = np.random.uniform(-0.3, 1.2)
            days_inactive = np.random.randint(0, 7)
            backlogs = np.random.randint(0, 2)
            dropout = 0
            # Facility usage — active students use facilities more
            lab_hours_per_week = round(np.random.uniform(3, 12), 1)
            sports_facility_visits = np.random.randint(1, 5)
            canteen_transactions_per_day = round(np.random.uniform(1.5, 4), 1)

        # Weekly risk scores (6 weeks)
        weekly_risks = []
        base_w = (1 - attendance_rate/100) * 0.4 + (1 - avg_score/100) * 0.4 + (1 - assignments_ontime_pct/100) * 0.2
        for w in range(6):
            noise = np.random.uniform(-0.05, 0.05)
            weekly_risks.append(round(min(max(base_w + noise + w * 0.02 * (1 if at_risk else -0.5), 0), 1), 3))

        # Subject scores
        subjects = {
            "Mathematics": round(np.random.uniform(35, 95) if not at_risk else np.random.uniform(25, 65), 1),
            "Physics": round(np.random.uniform(35, 95) if not at_risk else np.random.uniform(25, 65), 1),
            "Programming": round(np.random.uniform(35, 95) if not at_risk else np.random.uniform(25, 65), 1),
            "Communication": round(np.random.uniform(40, 95) if not at_risk else np.random.uniform(30, 70), 1),
            "Core Subject": round(np.random.uniform(35, 95) if not at_risk else np.random.uniform(25, 65), 1),
        }

        risk_score = round(weekly_risks[-1], 3)
        if risk_score > 0.65:
            risk_level = "High"
        elif risk_score > 0.35:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        student = {
            "student_id": f"STU{1000 + i}",
            "name": name,
            "age": age,
            "gender": gender,
            "department": department,
            "semester": semester,
            "first_gen": bool(first_gen),
            "origin": origin,
            "scholarship": scholarship,
            "assigned_advisor": assigned_advisor,
            "attendance_rate": round(attendance_rate, 1),
            "attendance_trend": round(attendance_trend, 1),
            "avg_score": round(avg_score, 1),
            "score_trend": round(score_trend, 1),
            "assignments_ontime_pct": round(assignments_ontime_pct, 1),
            "avg_submission_delay": round(avg_submission_delay, 1),
            "library_visits_per_week": round(library_visits_per_week, 2),
            "failed_subjects": failed_subjects,
            "sgpa": round(sgpa, 2),
            "sgpa_trend": round(sgpa_trend, 2),
            "days_inactive": days_inactive,
            "backlogs": backlogs,
            "dropout": dropout,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "weekly_risks": weekly_risks,
            "subject_scores": subjects,
            # Facility usage logs
            "facility_usage": {
                "lab_hours_per_week": lab_hours_per_week,
                "sports_facility_visits": sports_facility_visits,
                "canteen_transactions_per_day": canteen_transactions_per_day,
            },
        }

        # Generate intervention history
        student["intervention_history"] = generate_intervention_history(student, at_risk)

        students.append(student)

    return students

def generate_and_save():
    students = generate_students(300)
    os.makedirs("backend/data", exist_ok=True)
    with open("backend/data/students.json", "w") as f:
        json.dump(students, f, indent=2)

    # Save initial empty interventions log (for runtime additions)
    with open("backend/data/interventions.json", "w") as f:
        json.dump([], f)

    # Also save as CSV for model training
    rows = []
    for s in students:
        rows.append({
            "student_id": s["student_id"],
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
            "lab_hours_per_week": s["facility_usage"]["lab_hours_per_week"],
            "sports_facility_visits": s["facility_usage"]["sports_facility_visits"],
            "canteen_transactions_per_day": s["facility_usage"]["canteen_transactions_per_day"],
            "dropout": s["dropout"],
        })
    df = pd.DataFrame(rows)
    df.to_csv("backend/data/students.csv", index=False)
    print(f"Generated {len(students)} students")
    print(f"At-risk students: {sum(1 for s in students if s['risk_level'] == 'High')}")
    print(f"Medium risk: {sum(1 for s in students if s['risk_level'] == 'Medium')}")
    print(f"Low risk: {sum(1 for s in students if s['risk_level'] == 'Low')}")

    # Stats
    total_interventions = sum(len(s["intervention_history"]) for s in students)
    print(f"Total intervention records: {total_interventions}")
    print(f"Advisors assigned: {len(set(s['assigned_advisor'] for s in students))}")
    return students

if __name__ == "__main__":
    generate_and_save()
