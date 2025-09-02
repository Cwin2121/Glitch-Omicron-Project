import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. Load images
img_dir = "./qscans"
img_size = (224, 224)
images = []
filenames = []

for fname in os.listdir(img_dir):
    if fname.endswith(".png"):
        img_path = os.path.join(img_dir, fname)
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img)
        images.append(img_array)
        filenames.append(fname)

images = np.array(images)
images = preprocess_input(images)

# 2. Feature extraction with pretrained ResNet50
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')  # avg pooling -> 1D vector
features = model.predict(images)

# 3. Cluster features
n_clusters = 5  # adjust
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(features)

# 4. Optional: visualize with TSNE
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

plt.scatter(features_2d[:,0], features_2d[:,1], c=labels)
plt.savefig("./classify/qscan/qscan_classify.png")

plt.show()

