# 🎯 Recommendation System

> A game recommendation engine with a Streamlit dashboard, built as a take-home interview assignment.

---

## 📌 Situation

The goal of this project is to identify high-potential games to bet on by leveraging historical betting data and the season's game schedule.

> ⚠️ **Note:** This was a take-home interview assignment completed within a set timeline. All real data has been replaced with dummy data. The recommendation model is built but the production workflow is currently a work in progress.

---

## ✅ Task

### 1) Production Workflow
- Build a recommendation engine to identify high-value games for each player daily
- Pull the day's game schedule from the vendor feed every day at 8:00 AM EST
- Generate ranked top-3 game recommendations per player with confidence scores and personalized messaging

### 2) Validation
- Simulate a 2-week betting period using historical season data
- Compare pre-season schedule recommendations against actual games played

### 3) Reporting & Deployment
- Build a Streamlit dashboard to visualize model performance and validation results
- Deploy to a cloud platform with public access via a secure URL (excluding Streamlit Community Cloud)
- Accept a masked player ID to display their historical betting behavior
- Allow marketing managers and analysts to interactively query model recommendations

---

## ⚙️ Action

### 1) Data Preparation
- Handled date and text columns for consistency
- Removed duplicates and performed general data cleaning

### 2) Model Selection
- Evaluated multiple approaches for the ranking use-case
- Selected BPR-MF (Bayesian Personalized Ranking – Matrix Factorization) for its strong ranking performance
- Key tradeoff: lower explainability compared to simpler models

### 3) Model Building
- Trained the BPR-MF model on historical betting data
- Generated player-game predictions using dot product of latent factors passed through a sigmoid function

### 4) Output Generation
- Produced ranked recommendations exported to Excel
- Appended AI-generated personalized messaging at the player level

### 5) Streamlit App
- Built an interactive dashboard for visualization and querying of model results



