import numpy as np
import matplotlib.pyplot as plt

#sudo apt-get install python3-tk
import tkinter

# total 1000 dogs
greyhounds = 500
labs = 500

# for both, we create margin of 4 inches additional
grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)


plt.hist([grey_height, lab_height], stacked=True, color=['r','b'])
plt.show()