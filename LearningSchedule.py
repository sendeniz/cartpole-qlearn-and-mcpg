import numpy as np
import matplotlib.pyplot as plt


epsilons = []
alphas = []
for epoch in range(1000):
    epsilon = max(0.01, min( 1, 1.0 - np.log10(( epoch+1 ) / 25 )))
    alpha = max(0.01, min( 0.5, 1.0 - np.log10(( epoch+1 ) / 25 )))
    epsilons.append(epsilon)
    alphas.append(alpha)
plt.figure(figsize=(4,3))
plt.plot(epsilons, linewidth=3,label=r'$\epsilon$', color='C0')
plt.plot(alphas, linewidth=3, ls='--', label=r'$\alpha$', color='black')
plt.xlabel('Episode')
plt.ylabel('Value')
plt.title('Learning schedule parameters\n as a function of episode nr')
plt.legend()
plt.show()