"""Stage 4 test: hit /health and /predict directly, no server needed."""

from app import app

client = app.test_client()

print("GET /health ->", client.get("/health").get_json())

genuine = {
    "title": "Logistics Coordinator",
    "company_profile": "An established freight company operating since 2009 with 180 staff across two offices.",
    "description": "You will own the delivery schedule for inbound and outbound shipments and report weekly to the operations director.",
    "requirements": "Bachelor's degree in logistics. Three years of experience.",
    "benefits": "Medical cover, pension, 24 days annual leave.",
    "has_company_logo": 1,
    "has_questions": 1,
    "telecommuting": 0,
}

scam = {
    "title": "DATA ENTRY CLERK - WORK FROM HOME",
    "company_profile": "",
    "description": "EARN $5000 WEEKLY! No experience needed! Process payments through your bank account. Contact hiring@gmail.com now!!!",
    "requirements": "No experience required.",
    "benefits": "",
    "has_company_logo": 0,
    "has_questions": 0,
    "telecommuting": 1,
}

for name, payload in [("genuine", genuine), ("scam", scam)]:
    r = client.post("/predict", json=payload)
    print(f"\n[{name}] status={r.status_code}  {r.get_json()}")

r = client.post("/predict", json={"description": ""})
print(f"\n[empty] status={r.status_code}  {r.get_json()}")