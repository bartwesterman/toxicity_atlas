## A real-world toxicity-atlas shows that adverse events of combination-therapies results in additive rather than synergistic interactions

Küçükosmanoglu, Scoarta et al.
____________________________________________________________________________________________________________________________________________________
#### [Data and results](/data)
- 01_drugs_ids.csv: input file with the names of the drugs and their IDs
- 01_kompas_benchmark_data_drug_ids.csv: input file with the data collected from Farmacotherapeutisch Kompas (FK) database
- 01_md_data_init.csv: input file with the drug combinations data from FAERS databse
- 01_sd_data_init.csv: input file with the single drug combinations data from FAERS database
- 01_snomed_ids.csv: input file with the information about the SNOMED IDs and MEDDRA related explanations
- 02_data_init_bliss_chi.csv: output file with the calculated BLISS and Chi-square ratios
- 03_data_init_bliss_chi_min6.csv: output file with the records that had minimum six cases reported per SNOMED ID
- 04_data_init_bliss_chi_bench.csv: output file with commonly identified records in FAERS and Farmacotherapeutisch Kompas databases
- 05_data_final.csv: output file with the complete information
#### [Synergy assessment (Python script)](/stats.py)
____________________________________________________________________________________________________________________________________________________

### Abstract

Combination therapies are expected to improve anti-cancer treatment. However, it is currently difficult to assess what adverse events can be expected from combination therapies, given that the data is sparse, non-independent and recorded in a real-world context. We provide here a proof-of-concept study to assess whether combination therapies lead to additive or synergistic adverse event interactions. For this, 15 million patient-records describing adverse events using the FDA Adverse Event Reporting System (FAERS) were used and these are representative of real-world data. First, we visualized adverse-event frequencies on a fixed 2D grid where locations were based on UMAP dimensionality reduction. Combinatorial drug effects could then be assessed using RGB images to show their individual and combined effects. We observed that adverse events of drug combinations predominantly result in additive interactions. Subsequently, we used 7,300 of these single-drug grids to train a convolutional neural network (CNN) autoencoder to automatically recognize patterns of adverse events. This neural network was able to decode patterns for monotherapies but also for unseen combination therapy profiles which was validated on trial data from ClinicalTrials.gov (165,328 records for 209 drug combinations). This showed that adverse events occur in conserved and recognizable patterns that match to drug targets. Finally, we determined the adverse-event interaction pattern of 180 drug-combinations known to be avoided in clinical practice. These benchmark cases showed additive interactions rather than  synergistic interactions for 97% of the cases. Together, our work provides a new framework to analyze complex, sparse and non-structured semantic adverse events data. Our analysis shows that drug interactions as they occur in the clinic commonly lead to additive rather than synergistic  adverse event drug interactions, indicating that the interaction-landscape of adverse events is relatively well-balanced. This insight can assist in the implementation of new combination therapies in clinical practice.
