# Contributing to JUPR

## Branching
main = production
rebuild = experimental

## Run Locally
pip install -r requirements.txt
streamlit run app.py

## Migrations
Located in /migrations
Run in Supabase SQL editor.

## Stability Rules
- Guard against None
- Never block UI with heavy compute
- Use caching carefully
