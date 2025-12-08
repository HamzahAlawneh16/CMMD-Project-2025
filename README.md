# 🎓 Graduation Project: Hybrid Radiogenomics Multi-Task Breast Cancer Classification

<div align="center">
  <h3>Submitted in Partial Fulfillment of the Requirements for the Bachelor's Degree in</h3>
  <h3>**Artificial Intelligence and Data Science**</h3>
  <h3>The University of Jordan - School of Engineering/Science</h3>
  <hr>
  <h4><strong>Submitted By:</strong> Hamza Alawneh</h4>
  <h4><strong>Submission Date:</strong> December 2025</h4>
</div>

### 🚫 Copyright and Intellectual Property

**All rights reserved to Hamza Alawneh © 2025.**
This portfolio piece is authorized for academic evaluation and professional review only. Any unauthorized reproduction, commercial sale, or use without prior express written consent from the developer is strictly prohibited.

---

## 1. Executive Summary: The Crisis and The Breakthrough (60% to 85%)

### 1.1. The Critical Problem I Solved

The project started with severe limitations that made the model unusable:

* **Initial Low Accuracy (60%):** A baseline CNN model struggled with the diagnosis task, yielding a validation accuracy of only **60%**. This was the primary technical challenge.
* **Training Volatility:** The training process was unstable, characterized by erratic loss curves, indicating a fundamental flaw in the optimization setup.
* **Clinical Gap:** The original model could not address the crucial **Radiogenomics Gap**—the inability to predict the tumor's **Genetic Subtype** (which is essential for patient treatment).

### 1.2. The Technical Intervention & Outcome

I transformed the project by designing and implementing a robust **Ultimate Hybrid CNN-Transformer** architecture. This intervention stabilized the training, eliminated data leakage risks, and, most importantly, successfully raised the core diagnostic accuracy from 60% to **85%**. This demonstrates expertise in stabilizing complex deep learning systems under multi-task constraints.

---

## 2. Technical Intervention: Solutions Applied to Achieve 85%

The success of the project hinged on specific, advanced Machine Learning techniques applied to address the initial failure points:

| Methodology | Technical Implementation | Impact and Result (Portfolio Value) |
| :--- | :--- | :--- |
| **1. Architecture Overhaul** | **Hybrid CNN-Transformer Fusion** | **Accuracy Jump:** Solved the 60% crisis by fusing image features (CNN + Transformer) with **processed clinical data**. This jumpstarted performance and confirmed the value of multi-modal data. |
| **2. Stability & Convergence** | **AdamW Optimizer** & **Gradient Clipping (1.0)** | **Solved Instability:** Gradient Clipping successfully prevented exploding gradients, which were causing the volatile loss, enabling the model to converge smoothly and consistently. |
| **3. Multi-Task Optimization** | **Dual Loss + Label Smoothing (0.1)** | **Improved Generalization:** Forced the model to learn shared, robust representations useful for both Diagnosis (Binary) and Subtype (Multi-Class) predictions. **Label Smoothing** was key to improving the reliability of the Subtype prediction. |
| **4. Feature Handling** | **Custom HybridDataset (CELL 5)** | **Mitigated Data Leakage Risk:** Ensured safe, synchronized handling of all four components (`image`, `clinical_data`, `target_diag`, `target_subtype`) in every batch, confirming data integrity. |

---

## 3. Project Execution Flow: A Cell-by-Cell Breakdown (Cells 1-11)

This section provides a detailed look at the technical work accomplished in each phase of the project:

| Phase | Cells | Technical Achievement and Implementation |
| :--- | :--- | :--- |
| **A. Setup & Environment** | **CELL 1-2** | Installation of specialized libraries (`einops` for Transformer, `pydicom` for DICOM reading, `gradio` for UI). Defined `DEVICE` (CUDA/CPU) and implemented `set_seed(42)` for full **reproducibility**. |
| **B. Data Pipeline** | **CELL 3-6** | **Critical Data Engineering:** Preprocessing of clinical data (Normalization/Encoding). Defined the **`HybridDataset`** class (CELL 5) to safely read and output **four synchronized data elements** per batch, central to the hybrid architecture. |
| **C. Model Architecture** | **CELL 7** | **Model Construction:** Defined the **`UltimateHybridModel`**. Implemented the feature flow: ResNet $\to$ Transformer Encoder $\to$ Concatenation with **Clinical Data** $\to$ Dual Heads. |
| **D. Advanced Training** | **CELL 8** | **Core ML Engineering:** Implemented the complex training loop. Configured **Dual Loss functions**, applied **AdamW**, executed **Gradient Clipping (1.0)** for stabilization, and integrated **Cosine Annealing** scheduling. |
| **E. Evaluation & Reporting** | **CELL 9-10** | **Validation:** Loaded the best model. Correctly implemented prediction logic (`torch.round(torch.sigmoid())` for binary output). Generated and visualized formal **Classification Reports** and **Confusion Matrices** for both tasks, proving the 85% accuracy. |
| **F. Deployment** | **CELL 11** | **Proof-of-Concept Deployment:** Built an interactive web application using **Gradio**. Defined the single prediction function that accepts hybrid input (image + clinical data) for real-time demonstration. |

---

## 4. Final Architecture Overview

The model integrates multiple sophisticated components to achieve its multi-task goal:



---

## 5. Contact Information

I am open to discussing the technical challenges, optimization strategies, and the successful implementation of the multi-task learning paradigm.

* **Name:** Hamza Alawneh
* **Email:** [hamzamdhatalawneh@gmail.com](mailto:hamzamdhatalawneh@gmail.com)
* **GitHub:** [https://github.com/HamzahAlawneh16](https://github.com/HamzahAlawneh16)
