import numpy as np
import pandas as pd
import os, sys
from io import StringIO
import tempfile

import holoviews as hv
from holoviews import opts

import bokeh.models as bkm
import bokeh.palettes as bkp
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.layouts import layout
from holoviews.plotting.util import process_cmap
from bokeh.models import Range1d

import panel as pn
hv.extension('bokeh')

def get_nodes_and_edges(sttmap_output, 
                        deviation_by_sample,
                        sample_names
                       ):
    
    tiers_ttmap = ['all', 'low_map', 'mid1_map', 'mid2_map', 'high_map']

    # create a number list correspondence for the clusters
    node_correspondency = {}
    positions = {}
    x = []
    y = []
    node_labels = []
    weight_labels = []
    node_indices = []
    node_sizes = []
    base_size_node = 30
    clusters = {}
    global cluster_info
    cluster_info = {}
    
    for sample in sample_names:
        clusters[sample] = []

    counter = 1
    for which_interval, tier in enumerate(tiers_ttmap):
        
        # list with all clusters in the corresponding tier 
        clusters_interval = sttmap_output[tier]

        # check number of clusters in the interval
        number_of_clusters = len(clusters_interval)
        number_of_clusters = range(number_of_clusters)

        # y position of the node 
        y_pos = which_interval*3

        node_correspondency[tier] = []
        for which_cluster, cluster in enumerate(clusters_interval):

            node_correspondency[tier].append(counter)

            # convert samples to string
            all_samples = ''
            for sample_cluster in cluster:
                corrected_cluster = sample_cluster[:sample_cluster.find('.')] 
                clusters[corrected_cluster].append(str(counter))
                all_samples = all_samples + corrected_cluster + '\n'

            # add data to the node
            # names of the samples in the node
            node_labels.append(all_samples)
            # deviation by sample in the cluster
            weight_labels.append(np.average([deviation_by_sample[sample[:sample.find('.')]] 
                                             for sample in cluster]))

            # x position of the node
            x_pos = which_cluster*0.2
            x.append(x_pos)
            y.append(y_pos)

            # node size is 60 * number of samples in the cluster
            node_sizes.append(base_size_node * np.log(len(cluster)+1))
            node_indices.append(counter)
            
            cluster_info[counter] = [tier.split('_')[0]]

            counter += 1

    # get edges
    source_edges = []
    target_edges = []
    for top_counter, cluster in enumerate(sttmap_output['all']):
        # check if there is an overlap in the other intervals
        for tier in tiers_ttmap[1:]:
            for counter, sub_cluster in enumerate(sttmap_output[tier]):
                if set(cluster) & set(sub_cluster):
                    node_1 = node_correspondency['all'][top_counter]
                    node_2 = node_correspondency[tier][counter]
                    source_edges.append(node_1)
                    target_edges.append(node_2)
                    
    node_data = pd.DataFrame(data = {'x' : x, 'y' : y, 
                                 'index' : node_indices, 
                                 'samples' : node_labels,
                                 'mean_deviation' : weight_labels,
                                 'node_sizes' : node_sizes})
    
    node_and_edges_data = {'node_data'    : node_data, 
                           'source_edges' : source_edges,
                           'target_edges' : target_edges,
                           'clusters'     : clusters}
    
    return node_and_edges_data

def hook_graph(plot, element):
    """ 
    Create hook function to add colorbar and other elements to the graph
    """
    data_source = plot.handles['scatter_1_source'].data
    
    node_data_in_graph = data_source
    
    # create color bar to append on the left
    color_mapper_graph = bkm.LogColorMapper(palette="Viridis256", 
                                            low=np.min(node_data_in_graph['node_color']),
                                            high=np.max(node_data_in_graph['node_color']))
    colorbar_graph = bkm.ColorBar(color_mapper=color_mapper_graph, location=(0,0), title = 'Deviation')

    fig = hv.render(plot)
    fig.add_layout(colorbar_graph, 'left')
    
    plot.handles['plot'].sizing_mode = 'stretch_both'
    
    
def get_graph_from_data(node_and_edges_data):
    
    node_data    = node_and_edges_data['node_data']
    source_edges = node_and_edges_data['source_edges']
    target_edges = node_and_edges_data['target_edges']
    
    nodes = hv.Nodes(data=node_data)
    
    tooltips = [('Samples', '@samples'), ('Mean Deviation', '@mean_deviation'), ('Index', '@index')]
    hover = HoverTool(tooltips = tooltips)
    
    # add labels to nodes 
    labels = hv.Labels(nodes, ['x', 'y'], 'index')

    graph = hv.Graph(((source_edges, target_edges), nodes)).opts(
                        node_color='mean_deviation', cmap='viridis',
                        node_size = 'node_sizes',
                        tools=[hover], 
                        hooks=[hook_graph],
                        height = 600,
                        responsive = True,
                        xaxis = None, yaxis=None)
    
    return graph * labels

def fix_x_axis(plot, element):
    
    fig = hv.render(plot)
    fig.x_range=Range1d(0,1)

def get_outlier_analysis(batches, output_directory, filename='ttmap'):
      
    # plot the number of modified genes by control sample 
    if batches == []:
        batches = ['All']
    
    total_number_of_batches = range(len(batches))

    table_with_all_samples = pd.DataFrame({})

    for batch in total_number_of_batches:
        path_to_file = os.path.join(output_directory, filename + 'batch' +
                                    str(batch) + '_na_numbers_per_col.txt')

        dataset = pd.read_csv(path_to_file, header=0, index_col=0, sep='\t')
        dataset['Batches'] = [batches[batch]] * dataset.shape[0]
        dataset['Samples'] = list(dataset.index)

        table_with_all_samples = pd.concat([table_with_all_samples, dataset])

    table_with_all_samples.columns = ['Counts', 'Batches', 'Samples']

    key_dimensions = [('Samples', 'Sample')]
    value_dimensions = [('Counts', '# modified genes'), ('Batches', 'Batch')]

    viridis_cmap = process_cmap("Viridis", provider="bokeh", ncolors=len(batches))
    
    number_modified_genes = hv.Bars(table_with_all_samples, key_dimensions, value_dimensions).opts(tools=['hover'],
                                                                                   color='Batches',
                                                                                   cmap=viridis_cmap,
                                                                                   xticks = 0,
                                                                                   legend_position='top_left',
                                                                                   width = 500,
                                                                                   responsive = True
                                                                                  )
    
    # each gene is modified in x control samples. The plot below shows
    # the frequency of the percentage of corrected values (by samples) per
    # gene

    grid_corrected_genes = pn.GridSpec(sizing_mode='stretch_both', width=500)
    length_in_grid = np.int(np.ceil(np.float(len(batches))/np.float(3)))
    
    for counter, batch in enumerate(total_number_of_batches):
        path_to_file = os.path.join(output_directory, filename + 'batch' +
                                    str(batch) + '_na_numbers_per_row.txt')

        dataset = pd.read_csv(path_to_file, header=0, index_col=0, sep='\t')

        dataset.columns = ['Frequency', 'Other']

        frequencies, edges = np.histogram(dataset['Frequency'], 4, range=(0,1))
        
        histogram_frequencies = hv.Histogram((edges, frequencies), extents = (0,0,1,None)).opts(xlim=(0,1),
                                                                        responsive=True,
                                                                        tools=['hover'],
                                                                        hooks=[fix_x_axis],
                                                                        width = 220,
                                                                        title = batches[batch],
                                                                        xlabel = '% of corrected genes',
                                                                       )
        
        index_column = np.unravel_index(counter, [length_in_grid, 2])

        grid_corrected_genes[index_column[0], index_column[1]] = histogram_frequencies
    
    grid_corrected_genes[length_in_grid, :] = number_modified_genes
    
    return grid_corrected_genes 

