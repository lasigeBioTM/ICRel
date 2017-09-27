import ontospy
import rdflib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import inflect

from database_schema import Normalization
#names = set()
model = ontospy.Ontospy("./cl.owl")

#lymphocytes1 = model.getClass("http://purl.obolibrary.org/obo/CL_0000542")
#lymphocytes2 = model.getClass("http://purl.obolibrary.org/obo/CL_0000219")
cells = model.getClass("http://purl.obolibrary.org/obo/CL_0000000")
#for e in lymphocytes1.descendants():
#    names.add(e.bestLabel())

#for e in lymphocytes2.descendants():
#    names.add(e.bestLabel())

remappings = ["OVA", "monocyte", "Treg"]


synonyms = {}
for e in cells.descendants():
    #ames.add(e.bestLabel())
    synonyms[e.bestLabel()] = e.bestLabel()
    for syn in e.rdfgraph.objects(subject=e.uri,
                         predicate=rdflib.term.URIRef('http://www.geneontology.org/formats/oboInOwl#hasExactSynonym')):
        #print(e.bestLabel(), "exact", str(syn))
        #names.add(str(syn))
        if str(syn) not in remappings:
            synonyms[str(syn)] = e.bestLabel()

    # has_related_synonym
    for syn in e.rdfgraph.objects(subject=e.uri,
                         predicate=rdflib.term.URIRef('http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym')):
        #print(e.bestLabel(), "related", str(syn))
        #names.add(str(syn))
        if str(syn) not in remappings:
            synonyms[str(syn)] = e.bestLabel()

    for syn in e.rdfgraph.objects(subject=e.uri,
                         predicate=rdflib.term.URIRef('http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym')):
        #print(e.bestLabel(), "related", str(syn))
        #names.add(str(syn))
        if str(syn) not in remappings:
            synonyms[str(syn)] = e.bestLabel()

    for syn in e.rdfgraph.objects(subject=e.uri,
                         predicate=rdflib.term.URIRef('http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym')):
        #print(e.bestLabel(), "related", str(syn))
        #names.add(str(syn))
        if str(syn) not in remappings:
            synonyms[str(syn)] = e.bestLabel()

engine = inflect.engine()
#all_names = synonyms.keys()
for s in list(synonyms):
    plural = engine.plural_noun(s)
    if plural != s:
        synonyms[plural] = synonyms[s]

print(len(synonyms))

with open("cell_names.txt", 'w') as cellfile:
    cellfile.write('\n'.join(synonyms.keys()))

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

for cell in session.query(Normalization).filter(Normalization.entity_type == "cell"):
    session.delete(cell)

normalization = Normalization(text="monocyte", reference_name="monocyte", entity_type="cell",
                                                  reference_source="cellontology")
session.add(normalization)

normalization = Normalization(text="monocytes", reference_name="monocyte", entity_type="cell",
                                                  reference_source="cellontology")
session.add(normalization)

normalization = Normalization(text="Treg", reference_name="regulatory T cell", entity_type="cell",
                                                  reference_source="cellontology")
session.add(normalization)

normalization = Normalization(text="Tregs", reference_name="regulatory T cell", entity_type="cell",
                                                  reference_source="cellontology")
session.add(normalization)

with open("cell_reference.txt", 'w') as cellfile:
    for s in synonyms:
        cellfile.write('{}\t{}\n'.format(s, synonyms[s]))
        normalization = Normalization(text=s, reference_name=synonyms[s], entity_type="cell",
                                      reference_source="cellontology")
        session.add(normalization)
session.commit()
