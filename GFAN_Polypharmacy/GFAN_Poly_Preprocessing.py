
import pandas as pd
import numpy as np
from IPython.core.display import display, HTML
display(HTML("<style>.container {width:90% !important;}</style>"))
import networkx as nx
import scipy as sp
from math import sqrt
import csv




### Data import
# http://snap.stanford.edu/decagon

row_bio_decagon_combo = pd.read_csv ('./Raw_data/bio-decagon-combo.csv')
row_bio_decagon_targets = pd.read_csv ('./Raw_data/bio-decagon-targets.csv')



### Drug Network
# Node: drug (284)
# Node label: non
# Node feature: target genes (284 x 3648)
# Edge: polypharmacy (14247 x 2)
# Edge label: polypharmacy side effects (14247 x 1308)

df = pd.DataFrame
row_bio_decagon_targets['Gene'] = row_bio_decagon_targets['Gene'].astype(object)
Drug_Node = df(row_bio_decagon_targets['STITCH'].unique())    # Node: drug (284)
Drug_Node.to_csv(r'./Preprocessed_data/Drug_Node.csv', index = False)

temp1 = pd.get_dummies(row_bio_decagon_targets['Gene'])
temp1.insert(0,'STITCH',row_bio_decagon_targets['STITCH'], True)
temp1.set_index('STITCH',append=True)
node_feature_info = temp1.groupby(['STITCH']).size().reset_index().rename(columns={0:'count'})
Drug_Node_Feature = temp1.groupby(['STITCH']).sum()    # Node feature: target genes (284 x 3648)
Drug_Node_Feature.to_csv(r'./Preprocessed_data/Drug_Node_Feature.csv', index = False)

temp2 = row_bio_decagon_combo[row_bio_decagon_combo['STITCH 1'].isin(np.array(Drug_Node[0].tolist()))]
temp3 = temp2[temp2['STITCH 2'].isin(np.array(Drug_Node[0].tolist()))].reset_index(drop=True)
side_effect_dummies = pd.get_dummies(temp3['Polypharmacy Side Effect'])
temp4 = pd.concat([temp3.iloc[:,0:2],side_effect_dummies],axis=1)
edge_polypharmacy_info = temp3.groupby(['STITCH 1','STITCH 2']).size().reset_index().rename(columns={0:'count'})
Drug_Edge = edge_polypharmacy_info.iloc[:,0:2]     # Edge: polypharmacy (14247 x 2)
Drug_Edge_Label = temp4.groupby(['STITCH 1','STITCH 2']).sum()    # Edge label: polypharmacy side effects (14247 x 1308)
Drug_Edge.to_csv(r'./Preprocessed_data/Drug_Edge.csv', index = False)
Drug_Edge_Label.to_csv(r'./Preprocessed_data/Drug_Edge_Label.csv', index = False)


## data import
Drug_Node = pd.read_csv ('./Preprocessed_data/Drug_Node.csv')
Drug_Node_Feature = pd.read_csv ('./Preprocessed_data/Drug_Node_Feature.csv')
Drug_Edge = pd.read_csv ('./Preprocessed_data/Drug_Edge.csv')
Drug_Edge_Label = pd.read_csv ('./Preprocessed_data/Drug_Edge_Label.csv')

Drug_net = nx.Graph()     # Drug network
Drug_net.add_nodes_from(np.array(Drug_Node.iloc[0].tolist()))
temp5 = Drug_Edge.iloc[:,0:2].to_records(index=False)
Drug_net.add_edges_from(list(temp5))


### Polypharmacy Network
# Node: polypharmacy (14247)
# Node label: polypharmacy side effects (14247 x 1308)
# Node feature: combined target genes of two drugs (14248 x 3648)
# Edge: association btw polypharmacy including same drugs (14248 x 14248)
# Edge label: non


Poly_Node = Drug_Edge   # Node: polypharmacy (14247 x 2)
Poly_Node.to_csv(r'./Preprocessed_data/Poly_Node.csv', index = False)

Poly_Node_Label = Drug_Edge_Label     # Node label: polypharmacy side effects (14247 x 1308)
Poly_Node_Label.to_csv(r'./Preprocessed_data/Poly_Node_Label.csv', index = False)

Poly_Node_Feature = df()
Drug_Node_Feature.index = sum(Drug_Node.values.tolist(), [])
for i in Poly_Node.index:
    drug1 = Poly_Node.loc[i, 'STITCH 1']
    drug2 = Poly_Node.loc[i, 'STITCH 2']
    temp6 = Drug_Node_Feature.loc[drug1] + Drug_Node_Feature.loc[drug2]
    Poly_Node_Feature = Poly_Node_Feature.append(temp6, ignore_index=True)     # Node feature: combined target genes of two drugs (14248 x 3648)
Poly_Node_Feature.to_csv(r'./Preprocessed_data/Poly_Node_Feature.csv', index = False)



Poly_net = nx.Graph()     # Polypharmacy network
Poly_net.add_nodes_from(np.array(Poly_Node.index.tolist()))

G_inci = nx.incidence_matrix(Drug_net).todense()
temp7 = (G_inci.transpose())*(G_inci)
np.fill_diagonal(temp7, np.zeros((1,14247)))    # Edge: association btw polypharmacy including same drugs (14248 x 14248)
Poly_Edge = df(temp7)
Poly_Edge.to_csv(r'./Preprocessed_data/Poly_Edge.csv', index = False)




# getIndexes
def getIndexes(dfObj, value):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
    # Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos


Poly_Edge_list = df(getIndexes(Poly_Edge, 1), columns=['drug1','drug2']).astype('category')
Poly_Edge_list.to_csv(r'./Preprocessed_data/Poly_Edge_list.csv', index = False)
