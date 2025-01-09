import numpy as np
import matplotlib.pyplot as plt

carbon_price = [0, 51, 185]

plt.figure(figsize=(8, 6), dpi=200)
plt.plot(carbon_price, [33, 38, 45], 'o--', label="EVs cost 30% more than ICE vehicles (present)")
plt.plot(carbon_price, [46, 49, 59], 'o--', label="EVs cost 16% more than ICE vehicles")
plt.plot(carbon_price, [92, 93, 94], 'o--', label="EVs and ICE vehicles have equal cost")
plt.xlabel("Price of additional carbon tax [$0 / tCO2]")
plt.ylabel("Percentage of fleet that are EVs [%]")
plt.legend()
plt.show()
