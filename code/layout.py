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

final_display = pn.Tabs(('How to use TTMapp', how_to_use_it),
                        ('Two-Tier Mapper', final_display), 
                        ('Outlier Analysis', pn.GridSpec()),
                        ('Significant Components', significant_genes_analysis))

width_first_column = 290
pn_spacer = 1
grid_spec = pn.Row(pn.Column(
                        pn.pane.Markdown("# TTMapp"),
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
