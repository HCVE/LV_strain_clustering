import os
import plotly
import xlsxwriter
import align_ecg
import upsampling as us
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.decomposition import PCA


# create a list of numpy arrays. Each numpy array contains the ids of the patients that are assigned to the cluster
# indicated by the position in the list
def analyze_patient(label, patient):
    ids = [None] * len(np.unique(label))
    patient = np.array(patient)
    for i in range(len(np.unique(label))):
        ids[i] = patient[np.where(label == np.unique(label)[i])]
    return ids


# write the clustering results to an Excel file
def write2excel(label, patient, path, cluster_labels):
    workbook = xlsxwriter.Workbook(os.path.join(path, 'Clustering_assignments.xlsx'))
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'Patient ID')
    worksheet.write(0, 1, 'Cluster')
    for i in range(len(label)):
        worksheet.write(i + 1, 0, patient[i])
        worksheet.write(i + 1, 1, cluster_labels[label[i]])
    workbook.close()


# function to exclude patients as defined by the user
def exclude_patients(discarded_patients1, discarded_patients2, whole_data, cut_data, patient_names, time_interval):
    # delete the data for the excluded patient
    deleted_patients = np.unique(np.concatenate((discarded_patients1, discarded_patients2)))
    deleted_patients = list(map(str, deleted_patients))
    indexes = []

    for dp in deleted_patients:
        try:
            indexes.append(patient_names.index(dp))
        except ValueError:
            print('Patient with ID: {} is not in dataset'.format(dp))

    for ind in sorted(indexes, reverse=True):
        del whole_data[ind]
        del cut_data[ind]
        del patient_names[ind]
        del time_interval[ind]

    return whole_data, cut_data, patient_names, time_interval


# function to align all strain traces with respect to one template. The template can be defined by the user, or it is
# automatically selected as the first id in the dataset
def get_aligned_signals(extracted_data, select_point, interval, patient_id, reference_id, avc_times, p_wave_times):
    # normalize the ECG signals
    norm_original_data = extracted_data
    for i in range(len(extracted_data)):
        norm_original_data[i][2] = np.array([(float(j) - min(extracted_data[i][2])) /
                                             (max(extracted_data[i][2]) - min(extracted_data[i][2]))
                                             for j in extracted_data[i][2]])
    print('Start slicing data')
    sliced_data = []
    for i, d in enumerate(norm_original_data):
        start = np.where(norm_original_data[i][0] == interval[i][0])[0][0]
        ending = np.where(norm_original_data[i][0] == interval[i][1])[0][0]
        sliced_data.append(align_ecg.sliceup(d, start, ending, patient_id[i], avc_times, p_wave_times, select_point))

    # set as a reference ECG a clear signal. All the ECGs will be aligned with respect to the
    # reference signal
    try:
        patient_ref = patient_id.index(reference_id)
    except ValueError:
        print(f"Participant with ID {reference_id} does not exist. "
              f"Patient with ID: {patient_id[0]} is used as reference")
        patient_ref = patient_id.index(patient_id[0])

    print('Start Alignment')
    # interpolate the ecg signals so that the markers are aligned
    for i in range(len(patient_id)):
        align_ecg.stretch(sliced_data[i], ref=sliced_data[patient_ref])

    # from the time, strain and ecg take only the values that are between the
    # Left Time Marker and Right Time Marker in the txt file
    # this will produce arrays with variable length
    deformation_curve = []
    ecg = []
    time = []
    for i in range(len(norm_original_data)):
        temp1 = []
        temp2 = []
        temp3 = []
        for t, s, e in sliced_data[i].rescaled_slices:
            temp1 = temp1 + s.tolist()
            temp2 = temp2 + e.tolist()
            temp3 = temp3 + t.tolist()
        deformation_curve.append(np.array(temp1))
        ecg.append(np.array(temp2))
        time.append(np.array(temp3))

    # find the patient that have the most samples
    patient_index = np.argmax([len(deformation_curve[i]) for i in range(len(deformation_curve))])

    # interpolate all the other signals so that all the examples have the same number of samples
    interp_ecg, interp_deformation_curve, index, norm_time, interp_time = us.data_interpolation(ecg, deformation_curve,
                                                                                                time, patient_index)
    return interp_ecg, interp_deformation_curve, index, norm_time, interp_time


# plot the derivative of the LV strain traces. The derivative is used to approximate the LV strain rate.
# It produces figures in .png, .svg format using matplotlib and .html using plotly library
def plot_gradients(fit_data, interp_time, clustering_results, ids, patient_id, path, cluster_labels, cluster_colours):
    fig1, ax1 = plt.subplots(figsize=[8, 8])
    fig2, ax2 = plt.subplots(figsize=[8, 8])
    plot_data = []
    for c in range(len(np.unique(clustering_results))):
        centroid = np.zeros(fit_data.shape[1] - 1, dtype=object)
        for counter, id_value in enumerate(ids[c]):
            indice = patient_id.index(id_value)
            if counter == len(ids[c]) - 1:
                trace = go.Scatter(x=interp_time[1:len(interp_time)],
                                   y=np.diff(fit_data[indice]), mode='lines',
                                   opacity=0.5, marker=dict(color=cluster_colours[c]), text='ID: ' + str(id_value),
                                   name="Cluster " + str(cluster_labels[c]), showlegend=True)
            else:
                trace = go.Scatter(x=interp_time[1:len(interp_time)],
                                   y=np.diff(fit_data[indice]), mode='lines',
                                   opacity=0.5, marker=dict(color=cluster_colours[c]), text='ID: ' + str(id_value),
                                   name="Cluster " + str(cluster_labels[c]), showlegend=False)
            plot_data.append(trace)
            ax1.plot(interp_time[1:len(interp_time)], np.diff(fit_data[indice]),
                     cluster_colours[c], alpha=0.6, label="Cluster " + str(cluster_labels[c]))
            centroid += np.diff(fit_data[indice])
        centroid = centroid / len(ids[c])
        ax2.plot(interp_time[1:len(interp_time)], centroid,
                 cluster_colours[c], alpha=0.8, label="Cluster " + str(cluster_labels[c]))

    ax1.set_xlabel("Time (% Cycle)")
    ax1.set_ylabel("LV Strain Rate")
    legend_without_duplicate_labels(ax1)
    fig1.savefig(os.path.join(path, "Gradient.png"))
    fig1.savefig(os.path.join(path, "Gradient.svg"))
    fig1.show()
    plt.close(fig1)

    ax2.set_xlabel("Time (% Cycle)")
    ax2.set_ylabel("LV Strain Rate")
    ax2.set_title("Centroids of LV Strain Gradients")
    ax2.legend()
    fig2.savefig(os.path.join(path, "Gradient Centroids.png"))
    fig2.savefig(os.path.join(path, "Gradient Centroids.svg"))
    fig2.show()
    plt.close(fig2)

    layout = go.Layout(title="Gradient Analysis of LV Strain", hovermode='closest',
                       xaxis=dict(title='Time (% of Cycle)'),
                       yaxis=dict(title='LV Strain Rate'), font=dict(size=25))
    fig = dict(data=plot_data, layout=layout)
    plotly.offline.plot(fig, filename=os.path.join(path, 'Gradient.html'))


# function to fic the lenend in the figures
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    legend_attributes = sorted(unique, key=lambda tup: tup[1])
    ax.legend(*zip(*legend_attributes))


# plots the clustering results in .png, .svg and .html format. It produces figures with the individuals strain traces
# assigned in each cluster and one figure with the centroids only separately.
# The user should explicitly define the labels and the colours of the clusters
def visualize_clustering_results(interp_time, curve, ids, clustering_results, patient_id, centers, path, cluster_labels,
                                 cluster_colours):
    min_y_value = np.min(curve)
    max_y_value = np.max(curve)

    for c in range(len(np.unique(clustering_results))):
        plt.figure(figsize=[8, 8])
        plot_data = []
        for j in ids[c]:
            indice = patient_id.index(j)
            trace = go.Scatter(x=interp_time, y=curve[indice], mode='lines', opacity=0.5,
                               marker=dict(color='gray'), text='ID: ' + str(j))
            plot_data.append(trace)
            plt.plot(interp_time, curve[indice], "gray", alpha=0.3)

        if centers.all():
            trace = go.Scatter(x=interp_time, y=centers[c], mode="lines", marker=dict(color=cluster_colours[c]),
                               text="Cluster Centroid")
            plot_data.append(trace)
            plt.plot(interp_time, centers[c], cluster_colours[c])

        plt.xlabel("Time (% Cycle)")
        plt.ylabel("Strain (%)")
        plt.title("Cluster " + str(cluster_labels[c]))
        plt.ylim(min_y_value - 1, max_y_value + 1)
        plt.savefig(os.path.join(path, "Groupings of Cluster " + str(cluster_labels[c]) + ".png"))
        plt.savefig(os.path.join(path, "Groupings of Cluster " + str(cluster_labels[c]) + ".svg"))
        plt.show()
        layout = go.Layout(title='Cluster ' + str(cluster_labels[c]), hovermode='closest',
                           xaxis=dict(title='Time (% of Cycle)', range=[0, 1]),
                           yaxis=dict(title='Strain (%)', range=[min_y_value - 1, max_y_value + 1]),
                           font=dict(size=25))
        fig = dict(data=plot_data, layout=layout)
        plotly.offline.plot(fig, filename=os.path.join(path, 'Groupings of Cluster '+str(cluster_labels[c])+'.html'))

    fig, ax = plt.subplots(figsize=[8, 8])
    ax.set_title('Cluster Centroids')
    for c in range(len(np.unique(clustering_results))):
        ax.plot(interp_time, centers[c], label='Cluster ' + str(cluster_labels[c]), color=cluster_colours[c],
                linewidth=3)
    ax.set_xlabel('Time (% of Cycle)')
    ax.set_ylabel('Strain (%)')
    # ax.grid()

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    ax.legend(handles, labels)
    plt.ylim(min_y_value - 1, max_y_value + 1)
    plt.savefig(os.path.join(path, 'Cluster Results.png'))
    plt.savefig(os.path.join(path, 'Cluster Results.svg'))
    plt.show()
    plt.close()


# calculate the centroids of the time series LV strain curves by averaging the clustered curves.
def produce_centroids(predictions, fit_data):
    centers = []
    for i in np.unique(predictions):
        idx = np.where(i == predictions)
        centers.append(np.mean(fit_data[idx], axis=0))
    return centers


# visualize the first 3 components of PCA and colour it based on the clustering results.
# It produces figures in .png, .svg and .html format.
def plot_pca(clustering_results, fitted_data, pat_id, path, cluster_labels, cluster_colours):
    pca = PCA(n_components=3)
    pca_components = pca.fit_transform(fitted_data)

    plotted_data = []
    pat_id = np.array(pat_id)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(len(np.unique(clustering_results))):
        # ind = clustering_results.loc[clustering_results['Cluster'] == i + 1].index
        ind = np.where(clustering_results == i)[0]
        trace = go.Scatter3d(x=pca_components[ind, 0], y=pca_components[ind, 1], z=pca_components[ind, 2],
                             mode='markers', opacity=0.45, marker=dict(color=cluster_colours[i]),
                             name='Cluster: ' + str(cluster_labels[i]),
                             hovertext='ID: ' + str(pat_id[ind]) + '\n Cluster: ' + str(cluster_labels[i]))
        plotted_data.append(trace)
        ax.scatter(pca_components[ind, 0], pca_components[ind, 1], pca_components[ind, 2], alpha=0.45,
                   color=cluster_colours[i], label="Cluster " + str(cluster_labels[i]))
    layout = go.Layout(title='PCA of Strain Traces', hovermode='closest', showlegend=True,
                       scene=go.layout.Scene(xaxis=go.layout.scene.XAxis(title='1st PC'),
                                             yaxis=go.layout.scene.YAxis(title='2nd PC'),
                                             zaxis=go.layout.scene.ZAxis(title='3rd PC')))

    ax.set_xlabel("1st PC")
    ax.set_ylabel("2nd PC")
    ax.set_zlabel("3rd PC")
    ax.set_title("Principal Component Analysis of k-medoids clusters")

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    ax.legend(handles, labels)
    plt.savefig(os.path.join(path, "PCA Analysis.png"))
    plt.savefig(os.path.join(path, "PCA Analysis.svg"))
    plt.show()

    fig = dict(data=plotted_data, layout=layout)
    plotly.offline.plot(fig, filename=os.path.join(path, "PCA Analysis.html"))
