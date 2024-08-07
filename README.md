This repository refers to the published method "Knowledge Distillation for Scalable NILM" by Tanoni et al. 2024 (https://ieeexplore.ieee.org/document/10314043). DOI: 10.1109/TII.2023.3328436 

Codes for pre-training, fine-tuning and distillation are available. Also, the same pre-training and fine-tuning models are available to be used in the distillation process. 

![image](https://github.com/user-attachments/assets/0afead23-bfc7-423b-9265-568da048cf76)

Smart meters allow the grid to interface with individual buildings and extract detailed consumption information using Non-Intrusive Load Monitoring (NILM) algorithms applied to the acquired data. Deep Neural Networks, which represent the state-of-the-art for NILM, are affected by scalability issues since they require high computational and memory resources, and by reduced performance when training and target domains mismatched. This paper proposes a knowledge distillation approach for NILM, in particular for multi-label appliance classification, to reduce model complexity and improve generalisation on unseen data domains. The approach uses weak supervision to reduce labelling effort, which is useful in practical scenarios. Experiments, conducted on UK-DALE and REFIT datasets, demonstrated that a low-complexity network can be obtained for deployment on edge devices while maintaining high performance on unseen data domains. The proposed approach outperformed benchmark methods in unseen target domains achieving a F1-score 0.14 higher than a benchmark model 78 times more complex.

