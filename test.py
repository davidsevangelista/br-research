import numpy  as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.linspace(0, 2*np.pi)
index = pd.Index(x)
y1 = pd.Series(np.sin(x), index);
y2 = pd.Series(0.01 * np.cos(x), index);


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(y1)
ax1.set_ylabel('y1')

ax2 = ax1.twinx()
ax2.plot(y2, 'r-')
ax2.set_ylabel('y2', color='r')

plt.show()
