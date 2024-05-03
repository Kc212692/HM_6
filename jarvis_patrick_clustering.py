import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import pickle

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def jarvis_patrick(data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    def create_shared_neighbor_matrix(data, k, t):
        """Create matrix of shared neighbors based on Jarvis-Patrick criteria."""
        distance_matrix = squareform(pdist(data, 'euclidean'))
        neighbors = np.argsort(distance_matrix, axis=1)[:, 1:k+1]
        n = len(data)
        adjacency_matrix = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                shared_neighbors = len(set(neighbors[i]).intersection(neighbors[j]))
                if shared_neighbors >= t:
                    adjacency_matrix[i, j] = True
                    adjacency_matrix[j, i] = True
        return adjacency_matrix

    def calculate_sse(data, labels, cluster_centers, cluster_map):
        """Calculate Sum of Squared Errors (SSE) for clusters."""
        sse = 0.0
        for k, center_index in cluster_map.items():
            if k >= 0: 
                cluster_data = data[labels == k] 
                center = cluster_centers[center_index]  
                distances = np.linalg.norm(cluster_data - center, axis=1)  
                squared_distances = distances**2  # Square the distances
                sse += np.sum(squared_distances) 
        return sse

    def adjusted_rand_index(labels_true, labels_pred):
        # Find the unique classes and clusters
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)
        # Create the contingency table
        contingency_table = np.zeros((classes.size, clusters.size), dtype=int)
        for class_idx, class_label in enumerate(classes):
            for cluster_idx, cluster_label in enumerate(clusters):
                contingency_table[class_idx, cluster_idx] = np.sum((labels_true == class_label) & (labels_pred == cluster_label))
        # Compute the sum over the rows and columns
        sum_over_rows = np.sum(contingency_table, axis=1)
        sum_over_cols = np.sum(contingency_table, axis=0)
        n_combinations = sum([n_ij * (n_ij - 1) / 2 for n_ij in contingency_table.flatten()])
        sum_over_rows_comb = sum([n_ij * (n_ij - 1) / 2 for n_ij in sum_over_rows])
        sum_over_cols_comb = sum([n_ij * (n_ij - 1) / 2 for n_ij in sum_over_cols])
        n = labels_true.size
        total_combinations = n * (n - 1) / 2
        expected_index = sum_over_rows_comb * sum_over_cols_comb / total_combinations
        max_index = (sum_over_rows_comb + sum_over_cols_comb) / 2
        denominator = (max_index - expected_index)
        if denominator == 0:
            return 1 if n_combinations == expected_index else 0
        return (n_combinations - expected_index) / denominator

    def dbscan_custom(matrix, data, minPts):
        """Custom implementation of DBSCAN using shared neighbor matrix."""
        n = matrix.shape[0]
        pred_labels = -np.ones(n)
        cluster_id = 0
        cluster_centers = []
        cluster_map = {}
        for i in range(n):
            if pred_labels[i] != -1:
                continue
            neighbors = np.where(matrix[i])[0]
            if len(neighbors) < minPts:
                continue
            seed_set = set(neighbors)
            cluster_points = [data[i]]
            while seed_set:
                current_point = seed_set.pop()
                if pred_labels[current_point] == -2:
                    pred_labels[current_point] = cluster_id
                if pred_labels[current_point] != -1:
                    continue
                pred_labels[current_point] = cluster_id
                current_neighbors = np.where(matrix[current_point])[0]
                if len(current_neighbors) >= minPts:
                    seed_set.update(current_neighbors)
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(cluster_center)
            cluster_map[cluster_id] = len(cluster_centers) - 1
            cluster_id += 1
        return pred_labels, np.array(cluster_centers), cluster_map

    adjacency_matrix = create_shared_neighbor_matrix(data, k=params_dict['k'], t=2)
    pred_labels, cluster_centers, cluster_map = dbscan_custom(adjacency_matrix, data, minPts=params_dict['smin'])
    sse = calculate_sse(data, pred_labels, cluster_centers, cluster_map)
    ari = adjusted_rand_index(labels, pred_labels)
    computed_labels: NDArray[np.int32] | None = pred_labels
    SSE: float | None = sse
    ARI: float | None = ari
    return computed_labels, SSE, ARI

def jarvis_patrick_clustering():
    """
    Performs Jarvis-Patrick clustering on a dataset.
    Returns:
        answers (dict): A dictionary containing the clustering results.
    """
    answers = {}
    data = np.load("question1_cluster_data.npy")
    true_labels = np.load("question1_cluster_labels.npy")
    answers["jarvis_patrick_function"] = jarvis_patrick
    sse = []
    ari = []
    predictions = []
    final = []
    ks = [3, 4, 5, 6, 7, 8]
    smins = [4, 5, 6, 7, 8, 9, 10]
    for counter, i in enumerate(ks):
        for alp, s_val in enumerate(smins):
            datav = data[:1000]
            true_labelsv = true_labels[:1000]
            params_dict = {'k': i, 'smin': s_val}
            preds, sse_hyp, ari_hyp = jarvis_patrick(datav, true_labelsv, params_dict)
            final.append([sse_hyp, ari_hyp, i, s_val])
            sse.append(sse_hyp)
            ari.append(ari_hyp)
            predictions.append(preds)
    sse_numpy = np.array(sse)
    ari_numpy = np.array(ari)
    max_val = 0
    hig_k = 0
    hig_smin = 0
    for i in final:
        if i[1] > max_val:
            max_val = i[1]
            hig_k = i[2]
            hig_smin = i[3]
    alpha = final
    my_array = np.array(alpha)
    k_plot = np.array(my_array[:, 2], dtype='int')
    smins_plot = np.array(my_array[:, 3], dtype='int')
    ari_plot = np.array(my_array[:, 1])
    sse_plot = np.array(my_array[:, 0])
    plt.figure(figsize=(6, 6))
    plt.scatter(x=k_plot, y=smins_plot, c=ari_plot)
    plt.xlabel('k values')
    plt.ylabel('Smin values')
    plt.title('k vs Smins colored based on ARI')
    plt.colorbar()
    plt.savefig('BestARI_tuning_JarvisPatrick.png')
    plt.figure(figsize=(6, 6))
    plt.scatter(x=k_plot, y=smins_plot, c=sse_plot)
    plt.xlabel('k values')
    plt.ylabel('Smin values')
    plt.title('k vs Smins colored based on SSE')
    plt.colorbar()
    plt.savefig('BestSSE_tuning_JarvisPatrick.png')
    sse_final = []
    preds_final = []
    ari_final = []
    eigen_final = []
    for i in range(5):
        datav = data[i * 1000:(i + 1) * 1000]
        true_labelsv = true_labels[i * 1000:(i + 1) * 1000]
        params_dict = {'k': hig_k, 'smin': hig_smin}
        preds, sse_hyp, ari_hyp = jarvis_patrick(datav, true_labelsv, params_dict)
        sse_final.append(sse_hyp)
        ari_final.append(ari_hyp)
        preds_final.append(preds)
        if i not in groups:
            groups[i] = {'k': hig_k, 'ARI': ari_hyp, "SSE": sse_hyp, 'smin': hig_smin}
        else:
            pass
    sse_numpy = np.array(sse_final)
    ari_numpy = np.array(ari_final)
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]['SSE']
    least_sse_index = np.argmin(sse_numpy)
    highest_ari_index = np.argmax(ari_numpy)
    lowest_ari_index = np.argmin(ari_numpy)
    plt.figure(figsize=(6, 6))
    plot_ARI = plt.scatter(data[1000 * highest_ari_index:(highest_ari_index + 1) * 1000, 0], data[1000 * highest_ari_index:(highest_ari_index + 1) * 1000, 1], c=preds_final[highest_ari_index], cmap='viridis', marker='.')
    plt.title(f'Clustering Results Largest ARI for Dataset{highest_ari_index + 1}')
    plt.xlabel(f'Feature 1 for Dataset{highest_ari_index + 1}')
    plt.ylabel(f'Feature 2 for Dataset{highest_ari_index + 1}')
    plt.savefig('ARI_Clustering_Results.png')
    plt.figure(figsize=(6, 6))
    plot_SSE = plt.scatter(data[1000 * least_sse_index:(least_sse_index + 1) * 1000, 0], data[1000 * least_sse_index:(least_sse_index + 1) * 1000, 1], c=preds_final[least_sse_index], cmap='viridis', marker='.')
    plt.title(f'Clustering Results Least SSE for Dataset{least_sse_index}')
    plt.xlabel(f'Feature 1 for Dataset{least_sse_index + 1}')
    plt.ylabel(f'Feature 2 for Dataset{least_sse_index + 1}')
    plt.savefig('SSE_Clustering_Results.png')
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE
    answers["mean_ARIs"] = float(np.mean(ari_numpy))
    answers["std_ARIs"] = float(np.std(ari_numpy))
    answers["mean_SSEs"] = float(np.mean(sse_numpy))
    answers["std_SSEs"] = float(np.std(sse_numpy))
    return answers

if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
