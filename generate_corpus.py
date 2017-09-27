import requests
import logging
import xml.etree.ElementTree as ET
import codecs
import os
import sys
import time
import json
import pickle
from time import sleep
import subprocess
import multiprocessing as mp

from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from database_schema import Corpus, Document, Entity, Sentence, Token, Normalization

"""
Generate corpus based on a pubmed query
"""




def parse_pubmed_xml(xml, pmid):
    if xml.strip() == '':
        print("PMID not found", pmid)
        sys.exit()
    else:
        root = ET.fromstring(xml)
        title = root.find('.//ArticleTitle')
        if title is not None:
            title = title.text
        else:
            title = ""
        abstext = root.findall('.//AbstractText')
        if abstext is not None and len(abstext) > 0:
            abstext = [a.text for a in abstext]
            if all([abst is not None for abst in abstext]):
                abstext = '\n'.join(abstext)
            else:
                abstext = ""
        else:
            print("Abstract not found:", title, pmid)
            print(xml[:50])
            abstext = ""
            # print xml
            # sys.exit()
        articleid = root.findall('.//ArticleId')
        #for a in articleid:
        #    if a.get("IdType") == "pmc":
        #        self.pmcid = a.text[3:]
    return title, abstext

def get_pubmed_abs(pmid):
    logging.info("gettting {}".format(pmid))
    # conn = httplib.HTTPConnection("eutils.ncbi.nlm.nih.gov")
    # conn.request("GET", '/entrez/eutils/efetch.fcgi?db=pubmed&id={}&retmode=xml&rettype=xml'.format(pmid))
    payload = {"db": "pubmed", "id": pmid, "retmode": "xml", "rettype": "xml"}
    try:
        r = requests.get('http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', payload)
    except requests.exceptions.ConnectionError:
        r = requests.get('http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', payload)
    # logging.debug("Request Status: " + str(r.status_code))
    response = r.text
    # logging.info(response)
    title, abstract = parse_pubmed_xml(response.encode("utf8"), pmid)
    return title, abstract, str(r.status_code)

def get_pubmed_abstracts(searchterms, corpus_text_path, corpus_name, negative_pmids=[]):
    session = get_session()
    # searchterms = "+".join([t + "[mesh]" for t in terms])
    newcorpus = Corpus(name=corpus_name)
    for corpus in session.query(Corpus).filter(Corpus.name == corpus_name):
        session.delete(corpus)
    query = {"term": "{}+hasabstract[text]".format(searchterms),
             #"mindate": "2006",
             #"retstart": "7407",
             "retmax": "100000",
             "sort": "pub+date"} #max 100 000
    r = requests.get('http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi', query)
    logging.debug("Request Status: {}".format(str(r.status_code)))
    response = r.text
    print(r.url)
    root = ET.fromstring(response)
    pmids = []
    repeats = 0
    for pmid in root.find("IdList"):
        if pmids not in negative_pmids:
            pmids.append(pmid.text)
        else:
            print("repeated pmid: {}".format(pmid))
            repeats += 1
    print("repeated: {}".format(repeats))

    #with codecs.open(corpus_text_path, 'w', 'utf-8') as docfile:
    for i, pmid in enumerate(pmids):
        #doc = pubmed.PubmedDocument(pmid)
        title, abstract, status = get_pubmed_abs(pmid)
        # docfile.write(pmid + "\t" + abstract.replace("\n", " ") + '\n')
        doc = session.query(Document).filter(Document.pmid == int(pmid)).first()
        if not doc:  # create if it doesnt exist already
            doc = Document(pmid=int(pmid), title=title, abstract=abstract)
        newcorpus.documents.append(doc)
        doc.corpora.append(newcorpus)
        print("{}/{}".format(i, len(pmids)))
        sleep(0.4)
    session.add(newcorpus)
    session.commit()

def process_documents(corpus_name, nlp):
    session = get_session()
    corpus_id = session.query(Corpus).filter(Corpus.name == corpus_name).one().id
    docs = session.query(Document).filter(Document.corpora.any(id=corpus_id)).filter(Document.parsed == False).all()
    #print(docs)
    abstracts = [doc.abstract for doc in docs]
    pmids = [doc.pmid for doc in docs]
    #docs_generator = nlp.pipe(texts=abstracts, batch_size=10, n_threads=2)
    for i, doc in enumerate(docs):
    #for i, parsed_doc in enumerate(docs_generator):
        print(i, '/', len(abstracts))
        #print(doc.pmid, doc.title.encode("utf8"))
        parsed_doc = nlp(doc.abstract)
        doc = session.query(Document).filter(Document.corpora.any(id=corpus_id)) \
            .filter(Document.pmid == pmids[i]).one()
        for parsed_sentence in parsed_doc.sents:
            sentence = Sentence(offset=parsed_sentence.start_char, order=parsed_sentence.start,
                                section='A', text=parsed_sentence.text)

            doc.sentences.append(sentence)
            session.add(sentence)
            for word in parsed_sentence:
                #print(word.i, word.idx, word.text, word.lemma, word.lemma_, word.tag, word.tag_, word.pos, word.pos_)
                token = Token(start=word.idx-parsed_sentence.start_char, end=word.idx+len(word.text)-parsed_sentence.start_char,
                              order=word.i,
                              text=word.text, pos=word.tag_, lemma=word.lemma_)
                sentence.tokens.append(token)
                session.add(token)

        doc.parsed = True
        session.commit()
        #for word in results:
        #    print(word.text, word.lemma, word.lemma_, word.tag, word.tag_, word.pos, word.pos_)
    #session.commit()


def load_gold_relations(reltype):
    with codecs.open("seedev_relation.txt", 'r', "utf-8") as f:
        gold_relations = f.readlines()
    entities = {} # text -> types
    relations = {} # type#text -> type#text
    for r in gold_relations:
        values = r.strip().split("\t")
        if values[1] == reltype or reltype == "all":
            type1, entity1 = values[0].split("#")
            type2, entity2 = values[2].split("#")
            if entity1 not in entities:
                entities[entity1] = set()
            if entity2 not in entities:
                entities[entity2] = set()
            entities[entity1].add(type1)
            entities[entity1].add(type2)
            if values[0] not in relations:
                relations[values[0]] = set()
            relations[values[0]].add((values[2], values[1]))
    return entities, relations

def run_mer(text, lexicon, docid="0", docsection="A"):
    os.chdir("MER")
    #get_entities_args = ["./get_entities.sh", docid, docsection, text, lexicon]
    get_entities_args = ["./get_entities.sh", text, lexicon]
    #print(get_entities_args)
    p = subprocess.Popen(get_entities_args, stdout=subprocess.PIPE)
    output, err = p.communicate()
    os.chdir("..")
    #print(output)
    return output.decode()


def run_mer_externally(text, lexicons):
    """
    Call server to run MER using REST api
    :param text:
    :param lexicons: list of lexicon names
    :param docid:
    :param docsection:
    :return:
    """
    base_url = "http://cloud141.ncg.ingrid.pt/IBELight/ibelight.cgi"
    params = {"method":"getAnnotations",
              "becalm_key": "3deb66a13349fc7889549dfda065a3d8877ac04f",
              "text": text,
              "types": lexicons}
    request = requests.post(base_url, data=json.dumps(params))
    return request.text

def annotate_sentence(sentence_text, sentence_id, lexicon):
    #sentence_entities = run_mer_externally(sentence_text, [lexicon])
    sentence_entities = run_mer(sentence_text, lexicon)
    output = []
    if sentence_entities.strip():
        for l in sentence_entities.strip().split("\n"):
            #print(l)
            values = l.split('\t')
            char_start, char_end = int(values[0]), int(values[1])
            output.append((char_start, char_end, lexicon, values[2], sentence_id))
    return output



def annotate_documents(corpus_name, lexicons=["CHEMICAL", "CELL_LINE_AND_CELL_TYPE", "DISEASE", "PROTEIN", "MIRNA",
                                 "SUBCELLULAR_STRUCTURE", "TISSUE_AND_ORGAN"]):
    session = get_session()
    corpus = session.query(Corpus).filter(Corpus.name == corpus_name).one()
    print("annotating {}...".format(corpus))
    #procs = []
    #output = mp.Queue()
    results = []
    for lexicon in lexicons:
        for entity in session.query(Entity).filter(Entity.corpus_id == corpus.id)\
                .filter(Entity.ner == "mer_" + lexicon):
            session.delete(entity)
    session.commit()
    entities_added = 0
    all_docs = session.query(Document).filter(Document.corpora.any(name=corpus_name)).all()
    for i, doc in enumerate(all_docs):
        #if i < 10000:
        #    continue
        #if i == 100:
        #    sys.exit()
        logging.info("{}/{} {}".format(str(i), str(len(all_docs)), doc.pmid))
        for sent in session.query(Sentence).filter(Sentence.document_id == doc.pmid):
            for lexicon in lexicons:
                #annotate_sentence(sent, lexicon, output)
                #p = mp.Process(target=annotate_sentence, args=(sent.text, sent.id, lexicon, output))
                #procs.append(p)
                #p.start()
                #logging.info("annotating sentence with {}...".format(lexicon))
                sentence_output = annotate_sentence(sent.text, sent.id, lexicon)
                #logging.info(sent.text)
                #logging.info(sentence_output)
                #logging.info("done")
                #results += sentence_output
                #results = [output.get() for p in procs]

                #for p in procs:
                #    p.join()

                #print("adding entities...")
                for entity in sentence_output:
                    char_start, char_end, lexicon, text, sentence_id = entity
                    try:
                        #logging.info("searching for tokens...")
                        token_start_id = session.query(Token)\
                            .filter(Token.sentence_id == sentence_id) \
                            .filter(Token.start == char_start) \
                            .one().id
                        token_end_id = session.query(Token)\
                            .filter(Token.sentence_id == sentence_id) \
                            .filter(Token.end == char_end) \
                            .one().id
                        normalized = text
                        #logging.info("done")
                        """if lexicon == "cytokine":
                            q = session.query(Normalization).filter(Normalization.text == text) \
                                .filter(Normalization.entity_type == "cytokine").first()
                            if q:
                                normalized = q.reference_name
                        elif lexicon == "cell":
                            q = session.query(Normalization).filter(Normalization.text == text) \
                                .filter(Normalization.entity_type == "cell").first()
                            if q:
                                normalized = q.reference_name"""

                        entity = Entity(start=char_start, end=char_end,
                                        start_token_id=token_start_id,
                                        end_token_id=token_end_id,
                                        text=text, type=lexicon,
                                        normalized=normalized, ner="mer_" + lexicon,
                                        sentence_id=sentence_id, corpus_id=corpus.id)
                        session.add(entity)

                        #sent.entities.append(entity)
                        #corpus.entities.append(entity)
                        entities_added += 1
                    except NoResultFound:
                        logging.info("sent {}: {}".format(sentence_id, sent.text))
                        logging.info("skipped this entity: {}".format(entity))
                        #print(sentence_id)
                        #sentence = session.query(Sentence).filter(Sentence.id == sentence_id).one()

        session.commit()

        logging.info("added {} entities".format(entities_added))

    #session.commit()

def normalize_entities(goldstd, lexicons):
    session = get_session()
    for l in lexicons:
        print(l)
        if l == "cytokine":
            for entity in session.query(Entity).filter(Entity.ner == "mer_" + l):
                if goldstd in [c.name for c in entity.sentence.document.corpora]:
                    q = session.query(Normalization).filter(Normalization.text == entity.text) \
                        .filter(Normalization.entity_type == entity.type) \
                        .filter(Normalization.reference_source == "entrez").first()
                    if q:
                        # print(q)
                        normalized = q.reference_name
                        entity.normalized = normalized
                    else:
                        entity.normalized = entity.text
        elif l == "cell":
            for entity in session.query(Entity).filter(Entity.ner == "mer_" + l):
                if goldstd in [c.name for c in entity.sentence.document.corpora]:
                    q = session.query(Normalization).filter(Normalization.text == entity.text) \
                        .filter(Normalization.entity_type == entity.type)\
                        .filter(Normalization.reference_source == "cellontology").first()
                    if q:
                        # print(q)
                        normalized = q.reference_name
                        entity.normalized = normalized
                    else:
                        entity.normalized = entity.text
    session.commit()

def write_annotations_to_file(corpus_name):
    pass

def annotate_corpus_relations(corpus, model, corpuspath):

    logging.info("getting relations...")
    # entities, relations = load_gold_relations(reltype)
    logging.info("finding relations...")
    # print entities.keys()[:20]
    for did in corpus.documents:
        for sentence in corpus.documents[did].sentences:
            sentences_mirnas = []
            sentence_tfs = []
            #print sentence.entities.elist
            for entity in sentence.entities.elist[model]:
                if entity.type == "mirna":
                    sentences_mirnas.append(entity)
                elif entity.type == "protein":
                    sentence_tfs.append(entity)
            #for mirna in sentences_mirnas:
            #    for tf in sentence_tfs:
            #        ss = ssm.simui_go(mirna.best_go, tf.best_go)
            #        if ss > 0:
            #            print(ss, mirna.text, tf.text, mirna.best_go, tf.best_go)

    print("saving corpus...")
    corpus.save(corpuspath)

def get_session():
    with open("config/database.config", 'r') as f:
        for l in f:
            if l.startswith("username"):
                username = l.split("=")[-1].strip()
            elif l.startswith("password"):
                password = l.split("=")[-1].strip()
    #engine = create_engine('sqlite:///database.sqlite', echo=False)
    engine = create_engine('mysql+pymysql://{}:{}@localhost/immuno?charset=utf8mb4'.format(username, password), echo=False)
    Session = sessionmaker(bind=engine)
    #Base = declarative_base()
    session = Session()
    return session


def main():
    start_time = time.time()

    # try using configargparse for config files
    try:
        import configargparse
        parser = configargparse.ArgumentParser(description='')
    except ImportError:
        import argparse
        parser = argparse.ArgumentParser(description='')

    parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')
    parser.add_argument("--goldstd", default="", dest="goldstd", help="Gold standard to be used.")
    parser.add_argument("--corpus", dest="corpus_path", default="corpora/mirna-ds/abstracts.txt",
                        help="corpus path")
    parser.add_argument("--models", dest="models", help="model destination path, without extension")
    parser.add_argument("--entitytype", dest="etype", help="type of entities to be considered", default="all")
    parser.add_argument("--pairtype", dest="ptype", help="type of pairs to be considered", default="all")
    parser.add_argument("--doctype", dest="doctype", help="type of document to be considered", default="all")
    parser.add_argument("--pubmedquery", dest="pubmedquery", help="terms parameter of a pubmed query",
                        default='(("cystic fibrosis"[MeSH Terms] OR ("cystic"[All Fields] AND "fibrosis"[All Fields]) OR "cystic fibrosis"[All Fields])\
                       AND ("micrornas"[MeSH Terms] OR "micrornas"[All Fields] OR "mirna"[All Fields])) AND ("2011/09/04"[PDat] : "2016/09/01"[PDat])')
    parser.add_argument("-o", "--output", "--format", dest="output",
                        nargs=2, help="format path; output formats: xml, html, tsv, text, chemdner.")
    parser.add_argument("--get_pubmed", action='store_true')
    parser.add_argument("--parse", action='store_true')
    parser.add_argument("--annotate_entities", action='store_true')
    parser.add_argument("--normalize_entities", action='store_true')
    parser.add_argument("--lexicons", default=["CHEMICAL", "CELL_LINE_AND_CELL_TYPE", "DISEASE", "PROTEIN", "MIRNA",
                                 "SUBCELLULAR_STRUCTURE", "TISSUE_AND_ORGAN"], nargs='+')
    parser.add_argument("--log", action="store", dest="loglevel", default="WARNING", help="Log level")
    options = parser.parse_args()

    # set logger
    numeric_level = getattr(logging, options.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % options.loglevel)
    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    logging_format = '%(asctime)s %(levelname)s %(filename)s:%(lineno)s:%(funcName)s %(message)s'
    logging.basicConfig(level=numeric_level, format=logging_format)
    logging.getLogger().setLevel(numeric_level)
    logging.getLogger("requests.packages").setLevel(30)

    #negative_pmids = open("negative_pmids.txt", 'r').readlines()
    if options.get_pubmed:
        get_pubmed_abstracts(options.pubmedquery, options.corpus_path, options.goldstd)

    if options.parse:
        import spacy
        from spacy.symbols import ORTH, LEMMA, POS
        nlp = spacy.load('en_core_web_md')
        #nlp = spacy.load('en')
        print("processing corpus")
        process_documents(options.goldstd, nlp)

    if options.annotate_entities:
        annotate_documents(options.goldstd, options.lexicons)
        write_annotations_to_file(options.goldstd)
    #print session.query(Document).count()
    if options.normalize_entities:
        normalize_entities(options.goldstd, options.lexicons)

    # annotate
    #results = pickle.load(open("results/mirna_ds_entities.pickle", 'rb'))
    #results.load_corpus("mirna_ds")
    #corpus = results.corpus
    #annotate_corpus_relations(corpus, "combined", "corpora/mirna-ds/abstracts.txt_1.pickle")

if __name__ == "__main__":
    main()