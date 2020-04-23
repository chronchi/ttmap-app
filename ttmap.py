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

    col_names = [col_name[1:] if 'X' == col_name[0] else col_name for col_name in list(robjects.r['colnames'](r_dataset))]
    r_dataset.colnames = rpy2.robjects.StrVector(col_names)
    
    simp_ttmap_output = simpletTTMap.simple_ttmap(r_dataset,
                                                    output_directory  = output_directory,
                                                    alpha             = alpha,
                                                    batches           = batches,
                                                    test_samples      = test_samples,
                                                    outlier_parameter = outlier_value
                                                )

    py_simple_ttmap_output = dict(zip(simp_ttmap_output.names, simp_ttmap_output))

    samples_name_rpy2 = list(py_simple_ttmap_output['samples_name'].columns[3:])

    ttmap_output_rpy2 = dict(zip(py_simple_ttmap_output['ttmap_gtlmap'].names, map(list, list(py_simple_ttmap_output['ttmap_gtlmap']))))

    for intervals in ttmap_output_rpy2:
        for counter, clusters in enumerate(ttmap_output_rpy2[intervals]):
            ttmap_output_rpy2[intervals][counter] = list(clusters)
    
    deviation_components_rpy2 = py_simple_ttmap_output['ttmap_hda'] 

    deviation_components_rpy2 = dict(zip(deviation_components_rpy2.names, 
                                         map(list,list(deviation_components_rpy2))))

    total_deviations_rpy2 = deviation_components_rpy2['m']

    deviation_by_sample_rpy2 = {}

    for counter, sample_name in enumerate(samples_name_rpy2):
        deviation_by_sample_rpy2[sample_name] = total_deviations_rpy2[counter]

    return simple_ttmap_output(ttmap_output_rpy2, deviation_by_sample_rpy2, samples_name_rpy2, total_deviations_rpy2)


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
# widget for the button to run the web app
button_to_calculate = pn.widgets.Button(name = 'Calculate', button_type = 'primary', margin=margin_size)

# widget to query for sample or cluster information
query_box = pn.widgets.TextInput(name  = 'Search by sample or cluster number', 
                                 value = '')
# widget to select either sample or cluster
query_dropdown = pn.widgets.Select(name    = 'Parameter to search in the dataframe',
                                   options = ['Sample', 'Clusters'])

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

## Future additions
---
- Download all data produced
"""

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

