# LinkedIn Post — IDEWS Portfolio Launch

---

**💡 I built a production-grade AI system for one of banking's biggest unsolved problems. Here's what I learned.**

Credit card delinquency costs banks over **$500 billion annually**. Most banks still react to it *after* it happens.

I built IDEWS — an Intelligent Delinquency Early Warning System — to show what it looks like when you do this *right*: with real ML engineering, explainability, regulatory compliance, and MLOps built in from day one.

Not a Jupyter notebook. A production-ready system.

---

**What IDEWS does:**

🔍 Detects delinquency risk **60–90 days before the first missed payment** — before the damage is done

⚡ Returns a risk score in **<50ms** via REST API, with full SHAP explanations

🏛️ Generates **SR 11-7 compliant** adverse action reason codes automatically — no black box decisions

📊 Monitors for **data drift daily** using Evidently AI — so the model doesn't silently degrade

🔄 Retrains automatically when drift is detected via **GitHub Actions CI/CD**

---

**The tech stack (and why I chose it):**

→ **XGBoost** — the industry gold standard for credit risk; passes banking regulators' SR 11-7 model risk guidelines

→ **SHAP** — every score comes with plain-English explanations; required under ECOA/Reg B for adverse action notices

→ **FastAPI** — sub-50ms scoring API, deployed on GCP Vertex AI, scales to 10K+ requests/second

→ **MLflow** — full experiment tracking, model registry, and audit trail (one-click export for regulators)

→ **Evidently AI** — Population Stability Index monitoring across all 35 features

→ **Streamlit** — live risk monitoring dashboard for credit officers

All of this runs on **GCP Vertex AI for ~$25–40** in cloud costs. Your $100 free credit is more than enough to replicate it.

---

**The real lesson?**

Data scientists build models. MLOps engineers deploy them. Risk teams govern them. Regulators audit them.

Most AI projects fail because **no one person understands all four layers**.

My background in BFSI program leadership, quality engineering, and AI gave me an unusual vantage point — I could see all four layers simultaneously. That's what I tried to encode in IDEWS.

---

**Everything is open source:**

📁 Full GitHub repo: github.com/Satyapraveenv/idews-credit-delinquency-platform

Includes:
- Working XGBoost model (UCI Credit Card Default dataset)
- FastAPI scoring microservice
- Streamlit risk dashboard
- Evidently drift monitoring
- Docker Compose local stack
- GitHub Actions CI/CD with quality gates
- Full architecture deck (14 slides)

---

If you're a CRO, CDO, or Head of Risk at a bank and want to see how this maps to your delinquency challenge — I'd love a conversation.

If you're an AI practitioner curious about production ML in regulated industries — let's connect.

---

#CreditRisk #MachineLearning #MLOps #Banking #FinTech #BFSI #XGBoost #ExplainableAI #ProgramManagement #QualityEngineering #AIInnovation #VertexAI #IFRS9 #ModelRisk

---

*Tip for maximum reach: Post between 8–10am Tuesday or Wednesday. Add a banner image using the architecture slide from the deck. Tag 2–3 relevant people (former colleagues in banking risk, or people who engage with your content). Engage with comments in the first 60 minutes — LinkedIn's algorithm rewards early engagement heavily.*
