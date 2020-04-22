import rpy2
from rpy2 import robjects
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

simpletTTMap = importr('simpleTTMap')

import pandas as pd
import numpy as np

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


