# TrojanGuard AI Model Security

## Project Description

TrojanGuard AI Model Security is a deep learning-based security tool that detects hidden Backdoor/Trojan attacks in trained neural network models. It analyzes the model for unusual triggers and suspicious behavior.

## Objective

The main objective is to identify hidden backdoor triggers, analyze suspicious model behavior, and determine whether the model is likely clean or backdoored. The results are presented through an easy-to-use Streamlit dashboard.

## Methodology

1. **Dataset Preparation:** Use the CIFAR-10 dataset to prepare clean and poisoned image data.
2. **Model Training:** Train clean and backdoored CNN models using PyTorch.
3. **Image Processing:** Resize, normalize, and convert uploaded images into tensors.
4. **Model Inference:** Predict image classes and probabilities using trained CNN models.
5. **Trigger Analysis:** Apply a 3×3 trigger patch and compare normal and triggered predictions.
6. **Backdoor Detection:** Use Neural Cleanse and Activation Clustering to identify suspicious behavior.
7. **Result Visualization:** Display security scores, detection reports, and trigger visualizations through the Streamlit dashboard.

## Detection Techniques

- **Neural Cleanse** – Finds possible hidden triggers in the model.
- **MAD Score** – Identifies unusual trigger sizes as anomalies.
- **Activation Clustering** – Detects suspicious patterns in model activations.
- **Source-to-Target Analysis** – Checks whether a trigger changes one class into another.
  
## Technologies Used

- Python (3.10 version) – Used to develop the complete project and implement the detection algorithms.
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

  <img width="1536" height="1024" alt="System Architecture 1" src="https://github.com/user-attachments/assets/0eef5563-31a3-4ab6-bbf9-072a64b56a6b" />

## Installation

1. Clone the repository: 
git clone https://github.com/Mounikagoli-77/BackdoorGuard-Trojan-Detection-for-CNNs.git
cd BackdoorGuard-Trojan-Detection-for-CNNs
2. Create a virtual environment: 
python -m venv venv
3. Activate the environment: 
venv\Scripts\activate
4. Install dependencies: 
pip install -r requirements.txt

## Model Training

1. Prepare the CIFAR-10 dataset.
2. Train the clean CNN model using train_clean.py.
3. Generate poisoned data using poison_dataset.py.
4. Train the backdoored model using train_backdoor.py.
5. Save the trained models in the models/ directory.
   
## How to Run
    pip install -r requirements.txt
    streamlit run app.py

## Example Detection Results

The system generates detection reports containing:

- Model security verdict
- Suspicious class
- Neural Cleanse score
- Activation Clustering evidence
- Combined detection score
- Trigger analysis and visualization

**Example output:**

Model Verdict: MODEL LIKELY CLEAN
Suspicious Class: Deer
Combined Score: 0.3993

**Note:** These values are example results from an experimental
run and do not guarantee that the model is free of backdoors.

## Limitations

- Detection results may contain false positives or false negatives.
- Detection performance depends on the model and attack type.
- The system is primarily evaluated using the CIFAR-10 dataset.
- Detection does not guarantee complete removal of backdoors.
- Larger and more complex models may require additional resources.
  
## Results

Successfully detects possible backdoor attacks and displays the security verdict, suspicious class, trigger analysis, and detection evidence through the Streamlit dashboard.

## Future Scope

- Support more datasets and neural network architectures
- Improve detection accuracy
- Detect different types of backdoor attacks
- Support larger pretrained models

## Conclusion

TrojanGuard provides a simple and practical approach to analyzing neural network models for possible hidden Backdoor/Trojan attacks using multiple detection techniques.
