# Vityarthi-AI-ML-Project

Overview  
Features  
Technologies/ tools used  
Steps to install and run   
Installation steps  
Instructions for testing  
Expected outcomes  


# AI Project: LPG Demand Prediction

## Overview
This small project predicts LPG demand using **linear regression** with Python libraries.  
It’s designed to be easy to run and easy to understand — great for coursework, demos, or a quick portfolio piece.

## Why this project
- Shows the full workflow: data handling, model training, prediction, and visualization.  
- Uses common, industry tools (`pandas`, `scikit-learn`, `matplotlib`).  
- Can be run in **IDLE**, **Jupyter Notebook**, or **VS Code** so you can use whichever environment you prefer.

---

## Features
- **Data handling** with **pandas**.  
- **Linear regression** using **scikit-learn**.  
- **Visualization** with **matplotlib** (actual vs predicted).  
- Predicts LPG demand for future population values.

---

## Requirements
- **Python 3.11 (64-bit)** recommended.  
- Required libraries:
  - **pandas**
  - **scikit-learn**
  - **matplotlib**  


# Instructions for Testing
- Unit test style manual checks
  -  Run the script and confirm it prints predicted values for the sample future populations (1120 and 1140).
  - Confirm a plot window opens showing blue points for actual data and red points for predicted values.
- Interactive checks
  - In Jupyter Notebook, run cells step by step and inspect intermediate DataFrame contents.
  - Modify the sample dataset in data/sample_data.csv or in the script and re-run to see how predictions change.
 

# Expected Outcomes
Predicted LPG Demand (Tonnes):  
Month 1: 80.00  
Month 2: 85.00  


Visualization: a scatter plot titled LPG Demand: Actual vs Predicted with:
- Blue markers for actual sample data points (populations 1000–1100).
- Red markers for predicted points (population 1120 → 80 tonnes, population 1140 → 85 tonnes).
- Interpretation: the model should show a steady increase in predicted LPG demand as population increases. Because the dataset is small and synthetic, expect the model to be illustrative rather than production‑grade.

<img width="621" height="477" alt="Screenshot 2026-03-29 222120" src="https://github.com/user-attachments/assets/22569631-3e7a-4cc5-8831-4af26a18d6a1" />


