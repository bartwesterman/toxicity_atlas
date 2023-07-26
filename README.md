## A real-world toxicity-atlas shows that adverse events of combination-therapies results in additive rather than synergistic interactions

### Written by Asli Küçükosmanoglu, Silvia Scoarta, Nicoleta Spinu

### Abstract

Combination-therapies are a promising approach for improving cancer treatment, but it is challenging to predict their resulting adverse events. To address this issue, we provide here a proof-of-concept study using 15 million patient records from the FDA Adverse Event Reporting System (FAERS). To better understand the complex and real-world adverse-event data, we visualized frequencies of adverse events of drugs or their combinations as heatmaps onto a 2D UMAP grid. A convolutional neural network (CNN) autoencoder was able to recognize these adverse event frequencies. We characterized drug interactions over all drug combinations provided in the database and related them to independent datasets (clinicaltrials.gov and the pharmacotherapeutic compass). Our study shows that adverse events of drug combinations are commonly additive rather than synergistic. Furthermore, we show that they occur in similar patterns for individual drugs or their combinations. These real-world insights might enable the implementation of new combination therapies in clinical practice. 

____________________________________________________________________________________________________________________________________________________
#### [Data and results](/data)
- 01_drugs_ids.csv: input file with the names of the drugs and their IDs
- 01_kompas_benchmark_data_drug_ids.csv: input file with the data collected from Pharmacotherapeutic compass database
- 01_md_data_init.csv: input file with the drug combinations data from FAERS database
- 01_sd_data_init.csv: input file with the single drug combinations data from FAERS database
- 01_snomed_ids.csv: input file with the information about the SNOMED IDs and MEDDRA related explanations
- 02_data_init_bliss_chi.csv: output file with the calculated BLISS and Chi-square ratios
- 03_data_init_bliss_chi_min6.csv: output file with the records that had minimum six cases reported per SNOMED ID
- 04_data_init_bliss_chi_bench.csv: output file with commonly identified records in FAERS and Pharmacotherapeutic compass databases
- 05_data_final.csv: output file with the complete information
#### [Synergy assessment (Python script)](/stats.py)
#### [CNN model (Python script)](/cnn_model.py)
____________________________________________________________________________________________________________________________________________________
