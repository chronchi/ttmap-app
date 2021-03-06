import numpy as np
import pandas as pd
import os, sys
from io import StringIO
import tempfile

import rpy2
from rpy2 import robjects
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

simpletTTMap = importr('simpleTTMap')

import holoviews as hv
from holoviews import opts

import bokeh.models as bkm
import bokeh.palettes as bkp
from bokeh.models import HoverTool
from bokeh.models import Range1d
from holoviews.plotting.util import process_cmap

import panel as pn

hv.extension('bokeh')

class simple_ttmap_output:
    def __init__(self, ttmap_output, deviation_by_sample, samples_name, total_deviation):
        self.ttmap_output = ttmap_output
        self.deviation_by_sample = deviation_by_sample
        self.samples_name = samples_name
        self.total_deviation = total_deviation


def run_simpleTTMap(dataset,
                    output_directory = '.',
                    alpha            = 1.0,
                    batches          = [],
                    test_samples     = [],
                    outlier_value    = 1.0
                   ):
    """
    Run simpleTTMap from R and format its outputs.
    """    
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_dataset = ro.conversion.py2rpy(dataset)

    simp_ttmap_output = simpletTTMap.simple_ttmap(r_dataset,
                                                    output_directory  = output_directory,
                                                    alpha             = alpha,
                                                    batches           = batches,
                                                    test_samples      = test_samples,
                                                    outlier_parameter = outlier_value
                                                )

    py_simple_ttmap_output = dict(zip(simp_ttmap_output.names, simp_ttmap_output))

    samples_name_rpy2 = list(py_simple_ttmap_output['samples_name'].columns[3:])
    samples_name = [sample_name[1:] if 'X' == sample_name[0] else sample_name for sample_name in samples_name_rpy2]
    samples_name_rpy2 = samples_name   
 
    ttmap_output_rpy2 = dict(zip(py_simple_ttmap_output['ttmap_gtlmap'].names, map(list, list(py_simple_ttmap_output['ttmap_gtlmap']))))

    for intervals in ttmap_output_rpy2:
        for counter, clusters in enumerate(ttmap_output_rpy2[intervals]):
            # convert the name of the samples in the clusters
            clusters_correct_names = []
            for cluster in list(clusters):
                correct_names = [sample_name[1:] if 'X' == sample_name[0] \
                                 else sample_name for sample_name in cluster]
                clusters_correct_names.append(correct_names)
            ttmap_output_rpy2[intervals][counter] = list(clusters)
    

    deviation_components_rpy2 = py_simple_ttmap_output['ttmap_hda'] 

    deviation_components_rpy2 = dict(zip(deviation_components_rpy2.names, 
                                         list(deviation_components_rpy2)))

    total_deviations_rpy2 = deviation_components_rpy2['m']
    hda_matrix = deviation_components_rpy2['Dc.Dmat']

    ttmap_hda_deviation = py_simple_ttmap_output['absolute_deviations']

    deviation_by_sample_rpy2 = {}

    for sample_name in samples_name:
        deviation_by_sample_rpy2[sample_name] = ttmap_hda_deviation.loc[sample_name + '.Dis', 
                                                                        'total_absolute_deviation']

    
    return simple_ttmap_output(ttmap_output_rpy2, deviation_by_sample_rpy2, samples_name_rpy2, total_deviations_rpy2)


def convert_output_to_dictionary(sttmap_output):
    """
    Convert the output of two tier mapper to a dictionary, so each cluster 
    corresponds to a key in the dictionary. Each key contains the 
    samples in the cluster and in which partition the cluster is
    """
    
    tiers_ttmap = ['all', 'low_map', 'mid1_map', 'mid2_map', 'high_map']
    
    ttmap_clusters = {}

    counter = 1
    for number_tier, tier in enumerate(tiers_ttmap):
        for number_cluster, cluster in enumerate(sttmap_output[tier]):
            ttmap_clusters[counter] = [cluster, tier] 
            counter += 1 

    return ttmap_clusters

def get_nodes_connected_to(base_node, ttmap_clusters):
    """
    Get nodes connected to the base node.

    base_node : integer representing the key in the ttmap_clusters
    """

    nodes_connected_to_base = []
    for node_index in ttmap_clusters:
        if node_index == base_node or ttmap_clusters[node_index][1] == 'all':
            continue

        if set(ttmap_clusters[base_node][0]) & set(ttmap_clusters[node_index][0]):
            nodes_connected_to_base.append(node_index)

    return nodes_connected_to_base

def get_nodes_and_edges(sttmap_output, 
                        deviation_by_sample,
                        sample_names
                       ):
    
    tiers_ttmap = ['all', 'low_map', 'mid1_map', 'mid2_map', 'high_map']

    tiers_better_names = {
                              'all':'All', 
                          'low_map': 'Lower Quartile', 
                         'mid1_map': '2nd Quartile', 
                         'mid2_map': '3rd Quartile',
                         'high_map': 'Higher Quartile'
                        }

    # define the nodes from bottom up. the base nodes contain the 
    # clusters in the overall partition. 

    ttmap_clusters = convert_output_to_dictionary(sttmap_output)
    base_nodes = [base_node for base_node in ttmap_clusters if ttmap_clusters[base_node][1] == 'all']
    base_nodes.sort()
    
    # for each base node get nodes connected to it and define their positions. 
    xpos = 0
    ypos_height = 3
    ypos = {
                 'all' : 0*ypos_height, 
             'low_map' : 1*ypos_height,
            'mid1_map' : 2*ypos_height,
            'mid2_map' : 3*ypos_height, 
            'high_map' : 4*ypos_height
           }

    # construct lists containing position of the nodes, sample names, indices, to which partition
    # the node is in, node size, the source and target of the edges.

    partitions = []
    x = []
    y = []
    node_indices = []
    
    # base value to node size. to make it bigger, increase this value
    base_size_node = 30
    node_sizes = []
    node_labels = []
    mean_deviation = []
    
    global cluster_info
    cluster_info = {}
    
    source_edges = []
    target_edges = []    
    
    # the clusters that each sample are in 
    clusters = {}
    
    # the partitions the samples are in as a dictionary
    sample_partition = {}

    for sample in sample_names:
        clusters[sample] = []
        sample_partition[sample] = ""

    for counter, base_node in enumerate(base_nodes):
        nodes_connected_to_base = get_nodes_connected_to(base_node, ttmap_clusters)
        for node_connected in nodes_connected_to_base:

            cluster_information = ttmap_clusters[node_connected]            
            # node indice
            node_indices.append(node_connected)              

            # x-position 
            x.append(xpos)
            xpos += 1 
        
            # y position 
            y.append(ypos[cluster_information[1]]) 
             
            # in which partition this cluster is 
            partitions.append(tiers_better_names[cluster_information[1]])

            # source and target of the edges
            source_edges.append(base_node)
            target_edges.append(node_connected)

            # convert samples to string
            all_samples = ''
            for sample_cluster in cluster_information[0]:
                corrected_cluster = sample_cluster[:sample_cluster.find('.')] 
                clusters[corrected_cluster].append(str(node_connected))
                sample_partition[corrected_cluster] = tiers_better_names[cluster_information[1]]
                all_samples = all_samples + corrected_cluster + '\n'
                
            # names of the samples in the node
            node_labels.append(all_samples)
            # deviation by sample in the cluster
            mean_deviation.append(np.average([deviation_by_sample[sample[:sample.find('.')]] 
                                             for sample in cluster_information[0]]))

            node_sizes.append(base_size_node * np.log(len(cluster_information[0])+1))
           
            cluster_info[node_connected] = [cluster_information[1].split('_')[0]]
    
        # add information of base node to the vectors 
        cluster_information = ttmap_clusters[base_node]            
            
        # node indice
        node_indices.append(base_node)              

        # x position. node base position is in the middle of all nodes. 
        x.append(np.mean(list(range(xpos-len(nodes_connected_to_base), xpos))))
        
        # y position 
        y.append(ypos[cluster_information[1]]) 
         
        # in which partition this cluster is 
        partitions.append(tiers_better_names[cluster_information[1]])

        # convert samples to string
        all_samples = ''
        for sample_cluster in cluster_information[0]:
            corrected_cluster = sample_cluster[:sample_cluster.find('.')] 
            clusters[corrected_cluster].insert(0,str(base_node))
            all_samples = all_samples + corrected_cluster + '\n'
            
        # names of the samples in the node
        node_labels.append(all_samples)
        # deviation by sample in the cluster
        mean_deviation.append(np.average([deviation_by_sample[sample[:sample.find('.')]] 
                                         for sample in cluster_information[0]]))

        node_sizes.append(base_size_node * np.log(len(cluster_information[0])+1))
    
        # cluster_info[base_node] is a list with a value in 'all', 'lower', 'mid1',
        # 'mid2' or 'high'. 
        cluster_info[base_node] = [cluster_information[1].split('_')[0]]
        
        # shift xpos for the next tree
        xpos += 1
    
    node_data = pd.DataFrame(data = {'x' : x, 
                                     'y' : y, 
                                 'index' : node_indices, 
                               'samples' : node_labels,
                            'partition'  : partitions,
                        'mean_deviation' : mean_deviation,
                            'node_sizes' : node_sizes})
    
    node_and_edges_data = {'node_data'    : node_data, 
                           'source_edges' : source_edges,
                           'target_edges' : target_edges,
                           'clusters'     : clusters,
                       'sample_partition' : sample_partition}
  
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
    
def hook_text(plot, element):
    plot.handles['plot'].sizing_mode = 'stretch_both'
  
 
def get_graph_from_data(node_and_edges_data):
    
    node_data    = node_and_edges_data['node_data']
    source_edges = node_and_edges_data['source_edges']
    target_edges = node_and_edges_data['target_edges']

    node_data['minus_mean_deviation'] = -node_data['mean_deviation']

    nodes = hv.Nodes(data=node_data)
    
    tooltips = [('Samples', '@samples'), 
                ('Mean Deviation', '@mean_deviation'), 
                ('Index', '@index'),
                ('Partition', '@partition')]

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
    
    # add text to the right of the graph indicating the quartiles
    xpos = np.max(node_data['x'])
    ypos = list(set(node_data['y']))
    ypos.sort()
    quartile_texts = ['All', 'Lower Quartile', 
                      '2nd Quartile', '3rd Quartile', 
                      'Higher Quartile ']
    labels_quartiles = hv.Labels({'x' : xpos, 
                                  'y' : ypos, 
                               'text' : quartile_texts
                                 }, 
                                 ['x','y'], 
                                 'text')
    
    labels_quartiles.opts(xoffset = 1, align='start')

    # TODO: append the quartile texts to the plot
       
    final_graph = graph * labels    

    #final_graph.opts(
    #        opts.Labels(text_color='y', cmap='BrBG', color_levels=5))
 
    return final_graph

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
 
        if dataset.shape[1] == 1:
            dataset.columns = ['Frequency']
        else:
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
        
        index_column = np.unravel_index(counter, (length_in_grid, 3))

        grid_corrected_genes[index_column[0], index_column[1]] = histogram_frequencies
    
    grid_corrected_genes[length_in_grid, :] = number_modified_genes
    
    return grid_corrected_genes 


def get_ttmap_from_inputs(event):
   
    # change the button to in progress
    grid_spec[0][9] = pn.widgets.Progress(name='Indeterminate Progress', active=True, width=270)

    try:
        loaded_data = load_input.value 
        # convert load input to string utf-8
        string_load_input = str(loaded_data, 'utf-8')
        data = StringIO(string_load_input) 
        
        #data = load_input.value
        # convert to pandas dataframe
        dataset = pd.read_csv(data, header=0, index_col=0)
        grid_spec[0][-1] = pn.Column(pn.pane.Markdown("""### Errors"""), background='#f0f0f0', width=290) 
    except: 
        # change button to calculate again
        grid_spec[0][-1] = pn.Column(pn.pane.Markdown(
"""### Errors: 

Check your dataset to see if it is compliant to the rules mentioned in the basic usage."""),        background='#f0f0f0', width=290)
        grid_spec[0][9] = button_to_calculate
        raise 

    print(dataset.head())    

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_dataset = ro.conversion.py2rpy(dataset)
    
    # get alpha value
    alpha = np.float(alpha_value.value)
    outlier_value = np.float(outlier_parameter.value)
    
    if which_batch.value == '':
        batches = []
    else:
        batches = which_batch.value.split(',')
        
    if test_samples.value == '':
        test_samples_names = []
    else:
        test_samples_names = test_samples.value.split(',')    

    global output_directory
    output_directory = tempfile.mkdtemp()
    
    try:    
        sttmap_output = run_simpleTTMap(dataset,
                                    alpha=alpha,
                                    batches=batches,
                                    test_samples=test_samples_names,
                                    outlier_value=outlier_value,
                                    output_directory=output_directory
                                   )
    except:
        grid_spec[0][9] = button_to_calculate

    ttmap_output = sttmap_output.ttmap_output
    deviation_by_sample = sttmap_output.deviation_by_sample
    sample_names = sttmap_output.samples_name
    total_deviation = sttmap_output.total_deviation 

    node_and_edges_data = get_nodes_and_edges(sttmap_output = ttmap_output,
                                          deviation_by_sample = deviation_by_sample,
                                          sample_names = sample_names
                                         )
    global graph_rpy2
    graph_rpy2 = get_graph_from_data(node_and_edges_data)
    
    name_clusters_by_sample = []
    sample_partition = []

 
    for sample in sample_names:
        name_clusters_by_sample.append(', '.join(node_and_edges_data['clusters'][sample]))
        sample_partition.append(node_and_edges_data['sample_partition'][sample])
    
    global data_table
    data_table = pd.DataFrame(data = {
                                      'Sample' : sample_names, 
                         'Total_Abs_Deviation' : total_deviation,
                                    'Clusters' : name_clusters_by_sample,
                                   'Partition' : sample_partition}
                                 )
    
    outlier_analysis = get_outlier_analysis(batches, output_directory=output_directory)
    
    
    grid_spec[1][2] = ('Outlier Analysis', outlier_analysis)
    grid_spec[1][1][0] = graph_rpy2
    grid_spec[1][1][0].sizing_mode='stretch_both'

    # change button and append table to column of queries and samples
    filename = 'samples_table.csv'
    data_table.to_csv(output_directory + '/' + filename)
    download_table_graph_button = pn.widgets.FileDownload(label = 'Download table', filename = 'samples_table.csv', 
                                                          button_type = 'primary',
                                                          margin=margin_size, 
                                                          file = os.path.join(output_directory, filename),
                                                          width=140)
    grid_spec[1][1][1][2][0] = download_table_graph_button
    grid_spec[1][1][1][3] = hv.Table(data_table).opts(width=300, hooks=[fix_size_table]) 

    # save graph and table into a html file
    filename = 'graph.html'
    graph_and_table = pn.Row(grid_spec[1][1][0], grid_spec[1][1][1][3])
    graph_and_table.save(os.path.join(output_directory, filename))

    # create download graph button to download the html of the graph and table
    download_graph_button = pn.widgets.FileDownload(label = 'Download graph',
                                                   filename = 'graph.html',
                                                   margin=margin_size,
                                                   file = os.path.join(output_directory, filename),
                                                   button_type = 'primary',
                                                   width=140
                                                   )
    # add to the grid
    grid_spec[1][1][1][2][1] = download_graph_button

    grid_spec[1][3] = ('Significant Components', 
                        pn.Column(query_box_deviated_genes, 
                                  query_genes, 
                                  download_significant_components_button,
                                  pn.Spacer(), #hv.Table(pd.DataFrame({})).opts(hooks=[fix_size_table]),
                                  sizing_mode = 'stretch_both'))

    grid_spec[1][3].sizing_mode = 'stretch_both'

    grid_spec[0][9] = button_to_calculate

def save_to_file(df):
    filename = 'samples_table.csv'
    df.to_csv(output_directory + '/' + filename)
    download_table_graph_button = pn.widgets.FileDownload(label = 'Download table', filename = 'samples_table.csv',
                                                          button_type = 'primary',
                                                          margin=margin_size,
                                                          file = os.path.join(output_directory, filename),
                                                          width=140
                                                         )
    grid_spec[1][1][1][2][0] = download_table_graph_button


def query_columns(event):
    
    name_of_column = query_dropdown.value 
    text = query_box.value
    
    s = data_table[name_of_column]
    s = s[s.str.contains('|'.join([text]))]
    rows_of_text = list(s.index)
    
    sub_data_table = data_table.iloc[rows_of_text] 
    save_to_file(sub_data_table)
    grid_spec[1][1][1][3] = hv.Table(sub_data_table).opts(width=300, hooks=[fix_size_table])

def fix_size_table(plot, element):
    
    plot.handles['table'].sizing_mode = 'stretch_both'
    
def query_clusters(event):
    
    # number of respective cluster to analyse
    cluster = query_box_deviated_genes.value
    
    if cluster == '':
        grid_spec[1][3][2] = pn.pane.Markdown('Please, input a cluster index first.')
    
    else:
        # genes to extract from table
        genes = query_genes.value

        genes = [gene.strip() for gene in genes.split(',')]

        # output_directory and cluster_info are globally defined .
        path_to_file = os.path.join(output_directory, 
                                    'clusters', 
                                    cluster_info[int(cluster)][0], 
                                    cluster + '.txt')

        significant_genes_table = pd.read_csv(path_to_file, header=0, index_col=0, sep='\t')
        colnames_significant_genes = [sample_name[:-4] for sample_name in significant_genes_table.columns]
        significant_genes_table.columns = colnames_significant_genes
        significant_genes_table.insert(loc=0, column='Genes', value=significant_genes_table.index)

        # select only the rows containing the genes
        true_rows = list(significant_genes_table['Genes'].str.contains('|'.join(genes)))

        filename = 'significant_components_' + cluster + '.csv'

        significant_genes_table.iloc[true_rows, :].to_csv(output_directory + '/' + filename)

        # query genes
        # download the cluster associated table of significant components
        download_significant_components_button = pn.widgets.FileDownload(filename = filename, 
                                                                 button_type = 'primary',
                                                                 file=os.path.join(output_directory, filename),
                                                                 label = 'Download table') 

        grid_spec[1][3][2] = download_significant_components_button
        grid_spec[1][3][3] = hv.Table(significant_genes_table.iloc[true_rows, :]).opts(hooks=[fix_size_table])
        grid_spec[1][3].sizing_mode = 'stretch_both'
        
margin_size = 5
# widgets for the first column
# widget to load matrix
load_input  = pn.widgets.FileInput(name = 'Dataset input', margin=margin_size)

# widget to type the batches
which_batch = pn.widgets.TextInput(name = 'Batch names',
                                   value = '', margin=margin_size)
# widget to type the alpha value
alpha_value = pn.widgets.TextInput(name = 'Alpha value', value = '1.0', margin=margin_size)

# widget to type the output parameter
outlier_parameter = pn.widgets.TextInput(name = 'Outlier parameter', value = '0', margin=margin_size)

# widget to type the subselected test sample dataset
test_samples = pn.widgets.TextInput(name = 'Test samples', value = '', margin=margin_size)


# widget to save the table of significant components of corresponding cluster
data_to_save = pd.DataFrame({})
sio = StringIO()
data_to_save.to_csv(sio)
sio.seek(0)
download_significant_components_button = pn.widgets.FileDownload(filename = 'significant_components.csv', 
                                                                 button_type = 'primary',
                                                                 file = sio, label = 'Download table') 

data_to_save = pd.DataFrame({})
sio = StringIO()
data_to_save.to_csv(sio)
sio.seek(0)
download_table_graph_button = pn.widgets.FileDownload(filename = 'samples_table.csv', button_type = 'primary', 
                                              margin=margin_size, file = sio, label = 'Download table', width=140)

data_to_save = pd.DataFrame({})
sio = StringIO()
data_to_save.to_csv(sio)
sio.seek(0)
download_graph_button = pn.widgets.FileDownload(filename = 'samples_table.csv', button_type = 'primary', 
                                              margin=margin_size, file = sio, label = 'Download graph', width=140)


# widget for the button to run the web app
button_to_calculate = pn.widgets.Button(name = 'Calculate', button_type = 'primary', margin=margin_size, width=270)

# widget explaining what alpha value, outlier parameter and deviation are
explanation_parameters = pn.pane.Markdown(""" 
### Glossary  

**Alpha value**: controls the noise when calculating the mismatch distance between
test samples.

**Outlier Parameter**: controls how far away the control sample should be from
the median in its batch. 

**Deviation**: the colorbar Deviation in the graph represents the 
average deviation component of the samples in each cluster. 
""")


# widget to query for sample or cluster information
query_box = pn.widgets.TextInput(name  = 'Search by sample or cluster number or partition', 
                                 value = '')
# widget to select either sample or cluster
query_dropdown = pn.widgets.Select(name    = 'Parameter to search in the dataframe',
                                   options = ['Sample', 'Clusters', 'Partition'])

# widget to query by cluster
query_box_deviated_genes = pn.widgets.TextInput(name  = 'Insert cluster index here', 
                                                value = '')
# widget to type the specific genes to query
query_genes = pn.widgets.TextInput(name  = 'Insert gene or list of genes separated by comma', 
                                                value = '')

button_to_calculate.param.watch(get_ttmap_from_inputs, parameter_names='clicks')
query_dropdown.param.watch(query_columns, parameter_names='value')
query_box.param.watch(query_columns, parameter_names='value')
query_box_deviated_genes.param.watch(query_clusters, parameter_names='value')
query_genes.param.watch(query_clusters, parameter_names='value')

how_to_use = """
# ttmap 
---

A web-app version of [Two-Tier Mapper](https://www.ncbi.nlm.nih.gov/pubmed/30753284)
to analyse gene expression levels.

## Basic instructions 
---

### Input

In order to use TTMapp, you need to upload a matrix with a matrix containing your data. The format of the 
matrix is given below.

|  | Sample1 | Sample2 | ... | Sample3 |
|--:|:------------:|:----------:|:-----:|:----------:|
| Gene1| x11| x12  | ... |x1N |
| Gene2| x21| x22 | ...  |x2N |
| ...| 
| GeneM |xM1| xM2 | ... |xMN |

This matrix should follow some specific rules: 

- The samples representing the control group should have CTRL in its name. It can be lower case too. 
- The batches should be identified in the colum names also. If Sample1 is from batch X and Sample2 
  is from batch Y, their names should contain X and Y respectively. 

Also, the matrix should be comma separated. Below is an example.

,"sample1","sample2"

"Gene1",0.3450,1.1231

"Gene2",2.1231,2.3523

The name of the samples should only contain alphanumeric characters (a-z, 0-9) and should not start with a number or other non syntatic
valid name as defined by R [here](https://stat.ethz.ch/R-manual/R-devel/library/base/html/make.names.html). This is due
to how R handle the name of the columns. 

### Batch names
The batches need to be specified if you want to batch correct. If you have 2 batches, X and Y,
it should be given as X,Y in the box. 

### Outlier parameter
When performing the control adjusment, Two-Tier Mapper calculates how distant a control sample is from 
the median of samples in the same batch. If this difference is bigger than a threshold, then
its value is corrected. This threshold is given by the *Outlier parameter*. The default is 1. 

### Alpha value
A threshold is necessary to filter noise when calculating the mismatch distance. The alpha
value is this threshold. The bigger the value, the less noise you allow. The default is 1.

### Test samples
If your matrix contains several columns and you want to analyse just a subset of them, you can
give a name in which represents these columns. For example, suppose you have a matrix with 10 columns
whose names are:

- CTRL_1_batch1, CTRL_2_batch1, CTRL_3_batch2
- Treat1_1_batch1, Treat1_2_batch1, Treat1_1_batch2,Treat1_2_batch2
- Treat2_1_batch1, Treat2_2_batch1, Treat2_1_batch2,Treat2_2_batch2

Then, to run TTMapp only on the samples from Treat1, you can type Treat1 in the Test Samples Box.

### Running TTMapp
After loading and setting these variables, you can press the button "Calculate" to run the 
app.

## Output
---
There are several outputs, and they are spread in the tabs *Two-Tier Mapper*, *Outlier Analysis*
and *Significant Components*. 

### Two - Tier Mapper
The main output of TTMapp is in the tab *Two-Tier Mapper*. There you can find the graph and a table. 

The graph is divided into 5 rows. From bottom to up, the first row is the overall interval, where
all samples are analysed together. The second row corresponds to the low quartile, were samples
with small total absolute deviation are clustered. In the third, fourth and fifth row there are
the 2nd quartile, 3rd quartile and high quartile respectively, where the total absolute deviation
increases from one quartile to the other. 

The table is a representantion of the graph, where you can search either by sample or cluster. It has
the same kind of information, but in a tabular manner.

### Outlier Analysis
In this tab, there are two plots. One containing the frequency that the genes were 
corrected in the control adjustment step divided by batch effects.

The other plot shows the number of corrected genes by sample.

These plots are good to look for outliers in your data, given that a sample could have more 
modified genes than the rest or some genes were modified in all samples, meaning they deviate
too much and could harm the analysis. 

### Significant Components
Each cluster has an associated set of genes, those that deviate in the same direction with respect
to the samples in the cluster. Deviating in the same direction means that the deviation components 
for each gene have the same sign. 

In this tab you can filter by cluster and then by gene. To select a cluster, you can type the index
that is in the corresponding node of the graph. Also, you can type a list of genes separated
by comma, e.g., "GREB1, BRCA1". Note that the genes should be the same as in the matrix you provided
at the beginning of the analysis.

## Download Data
---

You can download the samples table and the significant components table by clicking the respective
buttons 'Download table'. 

In every image there is floppy disk icon that you can press to save it.
 
To download the graph as a html, click the button 'Download graph'. It will download the 
graph and the table alongside. 

## Issues and questions 
---
If you have any issue or questions, please either raise an 
[issue on github](https://github.com/chronchi/ttmap-app/issues) or 
[mail me](mailto:carlos.ronchi@epfl.ch)
 
"""

how_to_use_it = pn.Column(pn.pane.Markdown(how_to_use, sizing_mode='stretch_both'),
                          sizing_mode='stretch_both', scroll = True)

dataframe_and_query = pn.Column(query_dropdown, query_box, 
                                pn.Row(download_table_graph_button, download_graph_button, align='center'),
                                hv.Table(pd.DataFrame({})).opts(width=300),
                                width = 300,
                               )

final_display = pn.Row(hv.Graph(()).opts(xaxis=None, yaxis=None, responsive=True),
                       dataframe_and_query,
                       sizing_mode = 'stretch_both'
                      )

significant_genes_analysis = pn.Column(query_box_deviated_genes, query_genes, download_significant_components_button,
                                       pn.Spacer(),
                                       sizing_mode = 'stretch_both')

final_display = pn.Tabs(('How to use ttmap', how_to_use_it),
                        ('Two-Tier Mapper', final_display),
                        ('Outlier Analysis', pn.GridSpec()),
                        ('Significant Components', significant_genes_analysis))

width_first_column = 290
pn_spacer = 1
grid_spec = pn.Row(pn.Column(
                        pn.pane.Markdown("# ttmap"),
                        pn.Column(load_input, width=width_first_column,
                                  background='#f0f0f0'),
                        pn.Spacer(height=pn_spacer),
                        pn.Column(which_batch, width=width_first_column,
                                  background='#f0f0f0'),
                        pn.Spacer(height=pn_spacer),
                        pn.Column(
                            pn.Row(alpha_value, outlier_parameter,
                                   width=width_first_column),
                            width=width_first_column,
                            background='#f0f0f0'
                        ),
                        pn.Spacer(height=pn_spacer),
                        pn.Column(test_samples, width=width_first_column,
                                  background='#f0f0f0'),
                        pn.Spacer(height=pn_spacer),
                        pn.Column(button_to_calculate, width = 270),
                        pn.Spacer(height=pn_spacer),
                        pn.Column(explanation_parameters, width = width_first_column,
                                  background='#f0f0f0'),
                        pn.Spacer(height=pn_spacer),
                        pn.Column(pn.pane.Markdown("### Errors"), background='#f0f0f0', width = width_first_column),
                        width = 300),
                     final_display,
                     sizing_mode='stretch_both'
                  )

grid_spec.servable(title='ttmap')

