import panel as pn

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
