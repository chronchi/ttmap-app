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
