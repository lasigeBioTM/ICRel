import csv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from database_schema import Normalization

# process CSV file converted from the excel file downloaded from IMMport
# Download from here: http://www.immport.org/immport-open/public/reference/cytokineRegistry
# convert to csv
# save to data/cytokines.csv

#engine = create_engine('sqlite:///database.sqlite', echo=False)
with open("config/database.config", 'r') as f:
    for l in f:
        if l.startswith("username"):
            username = l.split("=")[-1].strip()
        elif l.startswith("password"):
            password = l.split("=")[-1].strip()
# engine = create_engine('sqlite:///database.sqlite', echo=False)
engine = create_engine('mysql+pymysql://{}:{}@localhost/immuno?charset=utf8mb4'.format(username, password), echo=False)
Session = sessionmaker(bind=engine)
Base = declarative_base()
session = Session()

cytokine_names = set()
entrez_index = {}
for cytokine in session.query(Normalization).filter(Normalization.entity_type == "cytokine"):
    session.delete(cytokine)

remappings = ["IFN", "TGF-beta", "IL-1", "IL1" , "IL-12", "IL12", "IFN1", "IL-27",
              "interleukin 27", "interleukin-1", "IFN-a", "IFNA1", "interleukin-12"]

cytokine_names.update(set(remappings))
do_not_add = ["ml", "killer", "KILLER", "light", "LIGHT", "MDC"]

# extra entries:
#IFN -> IFN1
normalization = Normalization(text="IFN", reference_name="IFN1", entity_type="cytokine",
                                                  reference_source="entrez")
session.add(normalization)

normalization = Normalization(text="IFN1", reference_name="IFN1", entity_type="cytokine",
                                                  reference_source="entrez")
session.add(normalization)

normalization = Normalization(text="IFNA1", reference_name="IFNA1", entity_type="cytokine",
                                                  reference_source="entrez")
session.add(normalization)

normalization = Normalization(text="IFN-a", reference_name="IFNA1", entity_type="cytokine",
                                                  reference_source="entrez")
session.add(normalization)

normalization = Normalization(text="TGF-beta", reference_name="TGFB1", entity_type="cytokine",
                                                  reference_source="entrez")
session.add(normalization)
normalization = Normalization(text="IL-1", reference_name="IL1", entity_type="cytokine",
                                                  reference_source="entrez")
session.add(normalization)

normalization = Normalization(text="IL1", reference_name="IL1", entity_type="cytokine",
                                                  reference_source="entrez")
session.add(normalization)

normalization = Normalization(text="interleukin-1", reference_name="IL1", entity_type="cytokine",
                                                  reference_source="entrez")
session.add(normalization)

normalization = Normalization(text="IL-12", reference_name="IL12", entity_type="cytokine",
                                                  reference_source="entrez")
session.add(normalization)

normalization = Normalization(text="interleukin-12", reference_name="IL12", entity_type="cytokine",
                                                  reference_source="entrez")
session.add(normalization)

normalization = Normalization(text="interleukin-23", reference_name="IL23", entity_type="cytokine",
                                                  reference_source="entrez")
session.add(normalization)


normalization = Normalization(text="IL-27", reference_name="IL17", entity_type="cytokine",
                                                  reference_source="entrez")
session.add(normalization)

normalization = Normalization(text="interleukin 27", reference_name="IL17", entity_type="cytokine",
                                                  reference_source="entrez")
session.add(normalization)


with open("data/cytokines.csv", 'r') as cytos:
    cytoreader = csv.reader(cytos, delimiter=',')
    next(cytoreader)
    for line in cytoreader:
        if line[0].strip():
            # print(line[0])
            names = set()
            try:
                reference_name = line[5].strip()


                #if not reference_name.startswith("CD"):
                names.add(line[1]) # reference name
                names.add(line[5])  # entrez symbol
                names.update(line[27].split(';'))  # protein ontology synonym
                names.add(line[2]) # HGNC ID
                names.add(line[3]) # entrez name

                entrez_aliases = line[6].split(';')
                names.update(entrez_aliases)
                entrez_additional = line[7].split(';')
                names.update(entrez_additional)
                names.add(line[8]) # MGI
                names.add(line[14]) # uniprot ID
                names.add(line[15]) # uniprot name
                names.update(line[16].split(';')) # unprot alt names
                names.add(line[20]) # MESH ID
                names.add(line[21]) # mesh name
                names.update(line[22].split(';')) # typographical variations
                names.update(line[24].split(';')) # IX synonym
                names.add(line[26]) # protein ontology name




            except IndexError:
                print(line[:2], len(line))
            reference_name = line[5].strip()
            cytokine_names.update([n for n in names if n not in do_not_add])
            for n in names: #TODO: deal with repeated entries

                if n.strip() and reference_name and n.strip() not in remappings:
                    entrez_index[n.strip()] = line[5].strip()
                    normalization = Normalization(text=n.strip(), reference_name=reference_name, entity_type="cytokine",
                                                  reference_source="entrez")
                    session.add(normalization)


session.commit()

# 276 entries
#print(names)
print(len(names))
with open("data/cytokine_names.txt", 'w') as cytofile:
    cytofile.write('\n'.join(cytokine_names))

with open("data/cytokine_entrez_index.txt", 'w') as entrezfile:
    for n in entrez_index:
        entrezfile.write("{}\t{}\n".format(n, entrez_index[n]))
