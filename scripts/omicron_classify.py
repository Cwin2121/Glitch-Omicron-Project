import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv("/home/charliewilliam.winborn/projects/glitch/omicron_project/temp_csvs/table-LSC_POP_A_LF_OUT_DQ-snr_40d0-start_1437436820-end_1437523218.csv")

X = data.select_dtypes(include='number').values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the PCA result
plt.figure(figsize=(8,5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA")

plt.savefig("./classify/omicron/omicron_classify.png")
plt.show()




