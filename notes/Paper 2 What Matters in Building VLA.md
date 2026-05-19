___
## How to make a VLA model:

 - Action Space (Discrete vs Continuous):  Continuous because then the sample space is larger! 
 - History (Memory vs Moment): Convolution because it considers a bit of history as in WAP similar to the YOLO ka Kalmer Filter.
 - Now to decide where to add the history factor there are two ways:
	 -  Alongside or Interleaved
	 -  Or simply add the entire context at the very end, this is called Policy Head. This is cheaper faster and better overall.
 - Action Generation Method : 
	 - Diffusion Based (Flow Matching):
		 - Isme pehle noise based image me hum log mapping shuru karte hai velocity ka to find which is the correct direction.
	 - Deterministic Loss (MSE):
		 - Normal wala hai.
	 - Question to be answered in the next part: which is better?
 - When to use Open X-Emboidement (OXE) data:
	 - Question : Do we train on OXE and our specific robot at the same time? Or train on OXE first, then our robot?
	 - Answer: Best way is to pretrain on the OXE data for basic work and then fine tune as per the robot data. Basically pretrain on internet and post train on local data.
---
# Diffusion vs MSE :
### Problem Statement:
- Robot can move both left and right of the cup which to choose if both are correct.
### Answer:
- MSE : Average error 0 ho jayega as : $E[X] = 0.5 \cdot 1 + 0.5 \cdot (-1) = 0$ aur aise gir jayega.
- Octo / Pi - 0 : Wo ek direction choose karke hi move karega and hence surely girayega to nahi using the inital idea of choosing one direction of velocity.
### Issue with the proposed solution:
 - Diffusion bahut slow rehta which is expensive for low latency models.
### Proposed Solution:
 - Next frame predict karne ke jagah next 10 frame karle ( k/a Action Chunking ) , iske wajah se average out nahi hota poora and we get a non zero value not knocking the cup and easily passing what we need.
---
## Mixture of Experts (MoE) as in Pi-0 :
### The Idea :
- Instead of forcing the Vision-Language network to learn motor joint math (which might "corrupt" its ability to speak English or recognize images), Pi-0 adds a dedicated "Action Expert" (a separate neural pathway) attached to the end of the VLM. The visual/language tokens go one way, and the action tokens are routed strictly to the action expert.
### The Good and Bad : 
- If you drop the robot into an entirely _unseen_ room with _unseen_ objects (the "ABC split" in their tests), MoE shines. Because the core VLM's weights were protected from being "overwritten" by robot motor training, it retains all its broad internet knowledge and can reason about the completely weird scene.
- If the robot is working in a _seen_ environment (the "ABCD split"), MoE actually _hurts_ performance. If you know exactly what room the robot is in and what objects it will grab, you _want_ to overwrite the whole network to specialize in that exact setup. Keeping the VLM "general" via MoE prevents it from mastering the specific layout.
---


 
 
 