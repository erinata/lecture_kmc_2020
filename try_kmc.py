from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans



data, target = make_blobs(n_samples=400, centers=4, cluster_std=0.95, random_state=0)

# print(data)

plt.scatter(data[:,0],data[:,1])
plt.savefig('scatterplot.png')


def run_kmeans(n):
	machine = KMeans(n_clusters=n)
	machine.fit(data)
	results = machine.predict(data)
	centroids = machine.cluster_centers_
	ssd = machine.inertia_
	print(ssd)
	# print(results)
	# print(centroids)
	plt.scatter(data[:,0],data[:,1], c=results)
	plt.scatter(centroids[:,0], centroids[:,1], c='red', marker="*", s=200)
	plt.savefig('scatterplot_color.png')

# run_kmeans(1)
# run_kmeans(2)
# run_kmeans(3)
run_kmeans(4)
# run_kmeans(5)
# run_kmeans(6)
# run_kmeans(7)



