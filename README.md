# ICRel
Identifying Cellular Relations

## Requirements
* Python 3
* MySQL server or SQLite


## Getting Started

First install requirements
```
pip install -r requirements.txt
git clone https://github.com/lasigeBioTM/MER.git
```
Change config/database.config accordingly

Download cl.owl from https://bioportal.bioontology.org/ontologies/CL

Download cytokine registry from http://www.immport.org/immport-open/public/reference/cytokineRegistry and convert it to csv
Move both files to data/

```
python database_schema.py
python generate_cell_names.py
python generate_cytokine_names_entrez.py
```

Copy data/cell.txt and data/cytokine.txt to MER/data/ and run ```./produce_data_files cell``` and ```./produce_data_files cytokine```

Change config/immuno.config or create a new file according to your specifications

```
python generate_corpus.py -c config/immuno.config
```

Create directories temp/ and results/

```
python run_ds -c config/train_immuno.config
```

## References: 

- A. Lamurias, J. Ferreira, L. Clarke, and F. Couto, “Generating a tolerogenic cell therapy knowledge graph from literature,” Frontiers in Immunology, vol. 8, no. 1656, pp. 1--23, 2017 (https://doi.org/10.3389/fimmu.2017.01656)

- A. Lamurias, L. Clarke, and F. Couto, “Extracting microRNA-gene relations from biomedical literature using distant supervision,” PLoS ONE, vol. 12, no. 3, pp. 1--20, 2017 (https://doi.org/10.1371/journal.pone.0171929)
