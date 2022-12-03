# Final_Project_The_Trio

## *BEGNN*(BERT and GNN Together)
We are mainly focusing on the Text classication of the documents Using the BEGNN. 
so In this repository we do have different type of files of dataset,.py, README which we have done for the process of this paper [https://arxiv.org/abs/2105.05727]

## *Summary of the work:*
We employed the Bert-enhanced graph neural network to improve the text's representation capabilities. Here we are using BERT to extract the semantic features 
and GNN for structural features of the text. Then interacting and Aggregating these both the modules made the performance better on both of the datasets(R8 and ohsumed). 
We have compared our model results with other baseline models (GNN, GCN, BERT), and we are getting better accuracy with BERT+GNN.

## *Input*:
we are considering the documents from the two datasets mainly 
1)R8 datset
  It is a single-labelled document from Reuters dataset
2)ohsumed datset
  It consists of the medical abstracts of the MeSH categories.

## *Model*:
Here we use Bert and GNN models to process the built graphs and extracting the features from these models. 
which is BEGNN model in total

## *Output*:
We get our built model accuracies at the output and the results will be shown in results section
In this work, we propose BertGCN, a model that combines large scale pretraining and transductive learning for text classification. BertGCN constructs a  heterogeneous graph over the dataset and represents documents as nodes using BERT representations. By jointly training the BERT and GCN modules within BertGCN, the proposed model is able to leverage the advantages of both worlds: large-scale pretraining which takes the advantage of the massive amount of raw data and transductive learning which jointly learns representations for both training data and unlabeled test data by propagating label influence through graph convolution. Experiments show that BertGCN achieves SOTA performances on a wide range of text classification datasets. 


## Results:

|*Model* | *R8* | *R52* | *Ohsumed* |
| ------------ | ---- | ---- | ---- |
| TextGCN | 86.89 | 83.6 | 67.44 |
| Finetuned Bert | 97.44 | 86.45 | 97.99|
| BertGNN | 98.17 | 95.56 | 98.51 |

# The output files in the form of *Bert_datasetTerminalOutput_Log .txt* are the outputs of *finetune_bert.py*
   example output file *Bert_R8TerminalOutput_Log .txt* --for R8 dataset
# The output files in the form of *BertGNN_datasetTerminalOutput_Log.txt* are the outputs of *train_bert_plus_gnn.py*
   example output file *Bert_R8TerminalOutput_Log .txt* --for R8dataset


## Software/tools/data sets used in the project:

Tools used: 
cudatoolkit=10.2.243
dgl-cuda10.1=0.6
ignite=0.4.2
nltk=3.4.5
python=3.6.9
pytorch==1.5.0
scikit-learn=0.22
transformers=4.1.1



## Packages used: 
Numpy
Pandas
Networkx
using GNN to extract text characteristics

Create colab or visual studio environment for running all these files(.py)
1)install the softwares as menctioned above 
2)we need to install the required tools menctioned above using pip install --package

## Steps:
1)Run the building_graph.py --dataset for creating the graphs
 for example, consider the R8 dataset *python building_graph.py R8*
2)Run *python finetune_bert.py --dataset for getting the features from BERT
  for example, consider the R8 dataset *python finetune_bert.py --dataset R8*
3)Run *python train_bert_plus_gnn.py --dataset for extracting and aggregating the both features.
  for example, consider R8 dataset *python train_bert_plus_gnn.py --dataset R8  -m 0.7* , where m is the lambda value varies from 0 to 1(for better accuracy, lambda=0.7)



## References:
Bert-Enhanced Text Graph Neural Network for Classification
Published online 2021 Nov 18. doi: 10.3390/e23111536
PMCID: PMC8624482
PMID: 34828233



