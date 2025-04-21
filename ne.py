import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

heatmap_data = pd.DataFrame({
    "latitude": [26.9124, 26.9200, 26.9250, 26.9124, 26.9000],
    "longitude": [75.7873, 75.8000, 75.8100, 75.7873, 75.7500],
    "delivery_delay_hours": [1, 2, 1.5, 0.5, 3],
})

# Create pivot table for heatmap visualization
heatmap_pivot = heatmap_data.pivot_table(values="delivery_delay_hours", index="latitude", columns="longitude")

sns.heatmap(heatmap_pivot, annot=True, cmap='viridis')
plt.title('Heatmap of Delivery Delays Across GPS Routes')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
print()
