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

Change config/corpus/immuno.config or create a new file according to your specifications

```
python generate_corpus.py config/corpus/immuno.config
```

Create directories temp/ and results/

```
python run_ds -c config/train_immuno.config
```
