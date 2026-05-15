## Introduction:

- Senses and Physics
  - Vision and Language input
- Action Part
  - Movement karwata hai
- Closed Loop Feedback
  - Simple control system
---
## Pretrained Visual Representations:

- CLIP (Global Semantic Understanding)
  - Image-text correlation dhundta hai
  - Geometry weak hoti hai
  - Fine pixel tasks me struggle karta hai

- MAE (Masked Autoencoder)
  - Masked patches predict karta hai
  - Spatial understanding better hoti hai
  - CLIP ke saath combine karte hain
---
## Control Policies:

- Large VLA (Tokenization Approach):

  - LLM ke words ke jagah movement ke liye prediction:
	  - Useful for the same reasons as LLM can reason well.

  - Diffusion Approach:
     - Jaise Image generation me 0 se sab banta hai same way me yaha bhi random noise ko proper signals me convert karke use karte hai. Ye better hai kyuki agar bich me obstacle aaye to  :$$E[X] = 0.5 \cdot 1 + 0.5 \cdot (-1) = 0$$ na karke wo signal clear karta hai aur ek proper side decide karke crash nahi krta hai

---
## Dataset Issues:

-  Manual data creation leading to device based data formation :
    - Open Source karke Open VLA ne fix kar diya system
-  Simulated Data se train karna:
    - Obviously Physics exact nahi hogi to bekar rahega.
---


