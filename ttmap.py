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
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.layouts import layout
from bokeh.models import Range1d
from holoviews.plotting.util import process_cmap

import panel as pn

hv.extension('bokeh')

sys.path.append('code')
from utils import *
from widgets import *
from plot_functions import *
from howtouse import *

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


def get_ttmap_from_inputs(event):
    
    loaded_data = load_input.value 
    # convert load input to string utf-8
    string_load_input = str(loaded_data, 'utf-8')
    data = StringIO(string_load_input) 
    
    #data = load_input.value
    # convert to pandas dataframe
    dataset = pd.read_csv(data, header=0, index_col=0)
    
    #dataset.index = list(dataset['Unnamed: 0'])
    #dataset = dataset.drop('Unnamed: 0', axis='columns')
    
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
        
    sttmap_output = run_simpleTTMap(dataset,
                                                    alpha=alpha,
                                                    batches=batches,
                                                    test_samples=test_samples_names,
                                                    outlier_value=outlier_value,
                                                    output_directory=output_directory
                                                   )
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

    for sample in sample_names:
        name_clusters_by_sample.append(', '.join(node_and_edges_data['clusters'][sample]))
    
    global data_table
    data_table = pd.DataFrame(data = {'Sample' : sample_names, 
                                  'Total_Abs_Deviation' : total_deviation,
                                  'Clusters' : name_clusters_by_sample}
                                 )
    
    outlier_analysis = get_outlier_analysis(batches, output_directory=output_directory)
    
    
    grid_spec[1][2] = ('Outlier Analysis', outlier_analysis)
    grid_spec[1][1][0] = graph_rpy2
    grid_spec[1][1][1][2] = hv.Table(data_table).opts(width=300) 

def query_columns(event):
    
    name_of_column = query_dropdown.value 
    text = query_box.value
    
    s = data_table[name_of_column]
    s = s[s.str.contains('|'.join([text]))]
    rows_of_text = list(s.index)
    
    grid_spec[1][1][1][2] = hv.Table(data_table.iloc[rows_of_text]).opts(width=300)

def fix_size_table(plot, element):
    
    plot.handles['table'].sizing_mode = 'stretch_width'
    
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

        # query genes
        grid_spec[1][3][2] = hv.Table(significant_genes_table.iloc[true_rows, :]).opts(width=1000, hooks=[fix_size_table])
        grid_spec[1][3].sizing_mode = 'stretch_both'

button_to_calculate.param.watch(get_ttmap_from_inputs, parameter_names='clicks')
query_dropdown.param.watch(query_columns, parameter_names='value')
query_box.param.watch(query_columns, parameter_names='value')
query_box_deviated_genes.param.watch(query_clusters, parameter_names='value')
query_genes.param.watch(query_clusters, parameter_names='value')

how_to_use_it = pn.Column(pn.pane.Markdown(how_to_use, sizing_mode='stretch_both'),
                          sizing_mode='stretch_both', scroll = True)

dataframe_and_query = pn.Column(query_dropdown, query_box, hv.Table(pd.DataFrame({})).opts(width=300,
                                                                                           ))

final_display = pn.Row(hv.Graph(()).opts(xaxis=None, yaxis=None, responsive=True),
                       dataframe_and_query,
                       sizing_mode = 'stretch_both'
                      )

significant_genes_analysis = pn.Column(query_box_deviated_genes, query_genes,
                                       hv.Table(pd.DataFrame({})).opts(hooks=[fix_size_table]),
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
                        pn.Column(button_to_calculate, width = width_first_column),
                        width = 300),
                     final_display,
                     sizing_mode='stretch_both'
                  )

grid_spec.servable(title='TTMap')

