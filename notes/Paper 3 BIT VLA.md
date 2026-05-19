# Core Idea :
 - Instead of double values of neural network weights we simpy use 0, $\pm 1$
# Issue:
 - Clearly pata chal raha ki for obvious reasons model collapse kar jayega because of wrong weights if switched after like -0.0000001 bhi -1 hoga which is really bad clearly but agar start se hi train kare to thoda better ho sakta.
---
# Now the Internal Working:

### Firstly to make it very clear:
- If u use the approach in a vision encoder then it will break down as that needs detailed analysis.
### Now the solution:
 - We train initally one full model and a quantized one as well for the vision model, then both of the models are compared in results side by side and then the MSE is reduced over that as per our needs. basically we force it to show results as a overfit model would do which is what we exactly need.
### The Results / Benchmarks:
| Model            | Latency | Throughput |
| ---------------- | ------- | ---------- |
| OpenVLA          | 321 ms  | 77 Hz      |
| Pi-0 (Diffusion) | 86 ms   | 291 Hz     |
| BitVLA           | 73 ms   | 341 Hz     |
- Here the Throughput is basically the frequency of actions our model takes.
---
