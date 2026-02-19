# Recommendation-System
This project is to build a game recommendation system along with a Streamlit app 

Situation: The goal of this exercise is to identify high-potential games to bet on leveraging historical betting data and the season's game schedule.
This project was a take home job interview required to be completed within a timeline. I have used new dummy data. 
Task: 
1)Production workflow
a) Build a recommendation engine to identify high-value games for each player daily.
b) Pull the day's game schedule from the vendor feed every day at 8:00 AM EST.
c) Generate a ranked top-3 game recommendations per player with confidence scores and personalized messaging.
2) Validation 
a) Simulate a 2-week betting period using historical season data.
b) Compare pre-season schedule recommendations against actual games played.
c) Detect and document discrepancies from schedule drift or mismatches.
3) Reporting & Deployment
a) Build a Streamlit dashboard to visualize model performance and validation results.
b) Deploy to a cloud platform with public access via a secure URL (excluding Streamlit Community Cloud).
c) Accept a masked player ID to display their historical betting behavior.
d) Allow marketing managers and analysts to interactively query model recommendations.
Action:
a) Data preparation - handling date and text columns, removing duplicates etc. 
b) Determine best model for use-case: Picked Bayesian Personalized Ranking – Matrix Factorization since its better in ranking however 
falters on explainability

c) Build Model and predictions
d) Generate output in excel along with personalized messaging (AI generated text appended to user level data) 
e) Build Streamlit app 
 
Result:

