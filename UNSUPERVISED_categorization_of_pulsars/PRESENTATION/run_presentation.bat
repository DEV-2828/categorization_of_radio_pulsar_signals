@echo off
title Unsupervised Pulsar Categorization - ML Presentation
cd /d "%~dp0"
echo ====================================================================
echo  Launching Streamlit Presentation Dashboard...
echo  Dashboard will open at: http://localhost:8501
echo ====================================================================
streamlit run app.py --browser.gatherUsageStats false
pause
