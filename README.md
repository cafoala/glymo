# **Glymo: A Foundational Model for Type 1 Diabetes Glucose Dynamics**

Glymo is an advanced machine learning project designed to revolutionize the understanding and prediction of glucose dynamics in individuals with Type 1 Diabetes (T1D). By leveraging self-supervised learning on large-scale Continuous Glucose Monitoring (CGM) data, Glymo creates a foundational model that captures general glucose trends such as circadian rhythms, postprandial spikes, and basal glucose oscillations.

In its second stage, Glymo is tailored to predict glucose trends around exercise, factoring in individual criteria such as age, BMI, HbA1c, insulin modality, exercise intensity, and meal timing. This powerful tool aims to support personalized diabetes management and enhance research applications in T1D care.

---

### **Key Features**
- **Foundational Model**: Learns task-agnostic glucose patterns from T1D-specific CGM data using self-supervised learning.
- **Exercise Prediction**: Fine-tuned to simulate glucose trends during and after exercise based on lifestyle and physiological inputs.
- **Reusable Embeddings**: Provides general-purpose representations of glucose dynamics for diverse downstream tasks.
- **T1D-Specific**: Designed exclusively for the unique challenges and needs of individuals with Type 1 Diabetes.

---

### **Applications**
- Exercise glucose prediction and management.
- Personalized insulin regimen optimization.
- Stress and sleep impact modeling on glucose trends.
- Research into glycemic variability and T1D-specific patterns.

---

### **Technology Stack**
- **Self-Supervised Learning**: Techniques like masked modeling and contrastive learning to train the foundational model.
- **Deep Learning Architectures**: Transformers, Temporal Convolutional Networks, and more.
- **Python Libraries**: PyTorch, TensorFlow, NumPy, Pandas, and Matplotlib.
- **Preprocessing Tools**: Data normalization, time-series segmentation, and synthetic data augmentation.

---

### **Getting Started**
1. **Pretrained Model**: Use Glymo’s pretrained foundational model for your research or development needs.
2. **Fine-Tuning**: Tailor Glymo for specific tasks like exercise glucose prediction with your own labeled data.
3. **Interactive Tool**: (Coming soon!) A user-friendly interface to simulate glucose trends for T1D management.

---

### **Contributions**
We welcome contributions from researchers, developers, and clinicians passionate about advancing T1D care. Check out our contribution guidelines in the repository.

---

### **License**
This project is licensed under the [MIT License](LICENSE).
