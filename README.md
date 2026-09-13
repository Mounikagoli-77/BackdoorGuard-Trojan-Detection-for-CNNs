# TrojanGuard AI Model Security

## Project Description

TrojanGuard AI Model Security is a deep learning-based security tool that detects hidden Backdoor/Trojan attacks in trained neural network models. It analyzes the model for unusual triggers and suspicious behavior.

## Objective

The main objective is to identify hidden backdoor triggers, analyze suspicious model behavior, and determine whether the model is likely clean or backdoored. The results are presented through an easy-to-use Streamlit dashboard.

## Methodology

1. Create a clean CNN model using CIFAR-10.
2. Create a backdoored model by adding a trigger and changing labels.
3. Analyze the model using:
   - Neural Cleanse
   - MAD Anomaly Scoring
   - Activation Clustering
   - Source-to-Target Analysis
4. Combine the detection evidence.
5. Generate the final security verdict.
6. Display the results using a Streamlit dashboard.

## Detection Techniques

- **Neural Cleanse** – Finds possible hidden triggers in the model.
- **MAD Score** – Identifies unusual trigger sizes as anomalies.
- **Activation Clustering** – Detects suspicious patterns in model activations.
- **Source-to-Target Analysis** – Checks whether a trigger changes one class into another.
  
## Technologies Used

- Python – Used to develop the complete project and implement the detection algorithms.
- PyTorch – Used to build, train, and analyze the CNN neural network models.
- CIFAR-10 – Used as the image dataset for training and testing clean and backdoored models.
- Scikit-learn – Used for PCA, K-Means clustering, and anomaly analysis.
- NumPy – Used for numerical operations and handling image/model data.
- Pandas – Used to organize, analyze, and display detection results.
- Matplotlib – Used to create graphs and visualize triggers and detection results.
- Streamlit – Used to create the interactive web dashboard for the project.

## Key Features

- Detects possible Backdoor/Trojan attacks
- Identifies suspicious classes
- Analyzes hidden triggers
- Provides trigger visualization
- Displays detection evidence
- Generates security reports
- Interactive Streamlit dashboard
  
## System Architecture

        STAGE 1: MODEL CREATION
                 │
        ┌────────┴─────────┐
        ▼                  ▼
    Clean Model       Backdoored Model
    CIFAR-10 + CNN    Trigger + Relabel
        │                  │
        └────────┬─────────┘
                 ▼
        STAGE 2: DETECTION
                 │
        ┌────────┼────────┬──────────────┐
        ▼        ▼        ▼              ▼
    Neural      MAD    Activation     Source →
    Cleanse    Score    Clustering      Target
        │        │        │              │
        └────────┴────────┴──────────────┘
                 ▼
        Combined Evidence
                 │
                 ▼
        STAGE 3: FINAL RESULT
                 │
        ┌────────┴────────┐
        ▼                 ▼
    Model CLEAN      Model BACKDOORED
                           │
                           ▼
                  Suspicious Class
                           │
                           ▼
              Streamlit Dashboard
              • Detection
              • Verification
              • Trigger
              • Reports
  

## How to Run
    pip install -r requirements.txt
    streamlit run app.py

## Results

Successfully detects possible backdoor attacks and displays the security verdict, suspicious class, trigger analysis, and detection evidence through the Streamlit dashboard.

## Future Scope

- Support more datasets and neural network architectures
- Improve detection accuracy
- Detect different types of backdoor attacks
- Support larger pretrained models

## Conclusion

TrojanGuard provides a simple and practical approach to analyzing neural network models for possible hidden Backdoor/Trojan attacks using multiple detection techniques.
