
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 64, 1, float)

y=np.exp(-((x - 32) ** 2 ) / (2 * (2) ** 2))
print(len(x),len(y))
plt.plot(x, y)

plt.title('Cosine Function')
plt.xlabel('x')
plt.ylabel('cos(x)')

print('fuck')
# Show the plot
plt.show()

