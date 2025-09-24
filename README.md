## MKCD
#### Introduction
The project is an implementation of a multi-scale neighbor topology-guided transformer with Kolmogorov-Arnold network enhanced feature learning for circRNA and disease association prediction (MKCD). If you have any questions, please contact the email gtalhy@163.com.

#### Catalogs
**/mycode**: Contains main code.

**/mycodedata**: Contains the dataset used in our method.

**/mycodecode**: Contains the code implementation of MMDMA algorithm.

**/mycodedataloader.py**: Processes the disease, circRNA and miRNA similarities, associations, embeddings, and feature matrices.

**/mycodemodel.py**: Defines the model.

**/mycodemain.py**: Trains the model.

**/mycodeSF1.xlsx**: Contains the top 15 circRNA candidates predicted by MKCD for each disease.

**/mycode/Supporting_Information_SF2.docx**: Contains the supplementary experiments and analysis.

## Environment
The MKCD code has been implemented and tested in the following development environment:

Python == 3.11.9

Matplotlib == 3.7.2

PyTorch == 2.4.0

NumPy == 1.23.5

#### Dataset
circRNA_name.npy: Contains the names of 834 circRNAs.
disease_name.npy: Contains the names of 138 diseases.
miRNA_name.npy: Contains the names of 555 diseases.
circRNA_disease.npy: Contains the circRNA_disease associations.
circRNA_miRNA.npy: Contains the circRNA_miRNA interactions.
disease_miRNA.npy: Contains the disease_miRNA associations.
disease_disease.npy: Contains the disease similarities.
miRNA_miRNA.npy: Contains the miRNA similarities.

#### How to Run the Code
Train and test the model.
```
python /mycode/CNN_disease/Train_test_circ_disease.py
```
