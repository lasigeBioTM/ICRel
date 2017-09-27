import random
import sys
from time import sleep
import requests
import xml.etree.ElementTree as ET
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score, auc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from database_schema import Corpus, Document, Entity, Sentence, Token, Normalization, Pair

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


def write_results(fps, fns, tps, filename):
    docs = {}
    for f in tps:
        if f[0] not in docs:
            docs[f[0]] = []
        docs[f[0]].append(("TP:", f[1], f[2]))

    for f in fps:
        if f[0] not in docs:
            docs[f[0]] = []
        docs[f[0]].append(("FP:", f[1], f[2]))

    for f in fns:
        if f[0] not in docs:
            docs[f[0]] = []
        docs[f[0]].append(("FN:", f[1], f[2]))

    with open(filename, 'w') as f:
        for d in docs:
            f.write(str(d) + '\n')
            for l in docs[d]:
                f.write('\t'.join(l) + '\n')

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    # https://stackoverflow.com/a/312464/3605086
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_multiple_pmids(pmids, corpus_name):
    newcorpus = Corpus(name=corpus_name)
    result = []
    for i, c in enumerate(chunks(list(pmids), 200)):
        payload = {"db": "pubmed", "id": c, "retmode": "xml", "rettype": "xml"}
        try:
            r = requests.get('http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', payload)
        except requests.exceptions.ConnectionError:
            r = requests.get('http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', payload)
        # logging.debug("Request Status: " + str(r.status_code))
        response = r.text
        xml = response.encode("utf8")
        # logging.info(response)
        root = ET.fromstring(xml)
        articles = root.findall('.//PubmedArticle')
        #print(len(articles))
        for j, article in enumerate(articles):
            title = article.find('.//ArticleTitle')
            if title is not None:
                title = title.text
            else:
                title = ""
            abstext = article.findall('.//AbstractText')
            if abstext is not None and len(abstext) > 0:
                abstext = [a.text for a in abstext]
                if all([abst is not None for abst in abstext]):
                    abstext = '\n'.join(abstext)
                else:
                    abstext = ""
                pmid = article.find('.//PMID')
                result.append((pmid.text, title, abstext))
                doc = session.query(Document).filter(Document.pmid == int(pmid.text)).first()
                if not doc: # create if it doesnt exist already
                    doc = Document(pmid=int(pmid.text), title=title, abstract=abstext)
                newcorpus.documents.append(doc)
                doc.corpora.append(newcorpus)
                print("{}/{}".format(i*200+j, len(pmids)))
            else:
                print("Abstract not found:", title, pmid)
                print(xml[:50])
                abstext = ""
                # print xml
                # sys.exit()
    session.add(newcorpus)
    session.commit()
    return result



def main():
    """
    The purpose of this script is to process the immunoexpresso data
    It assumes the existance of immunoexpresso.relations, containing one relations per line with this format:
    source target relation_qualifier relation_source(redundant)
    and the file immunoexpresso.docs, with this format:
    cytokine cell relation_qualifier relation_source pmids(separated by ,)
    """
    base_file = "immunoexpresso.docs"
    relations_file = "immunoexpresso.relations"
    #base_file = "full_immunoexpresso_test.docs"
    #relations_file = "full_immunoexpresso_test.relations"
    #base_file = "full_tenpercent.docs"
    #relations_file = "full_tenpercent.relations"
    actions = ("add_documents", "add_relations", "evaluate_relations", "evaluate_docs", "evaluate_entities",
               "extend_corpus")
    if len(sys.argv) < 2 or sys.argv[1] not in actions:
        print("actions:", actions)
    if sys.argv[1] == "add_documents":
        # parse the file and add the pmids to the database to a corpus with the same name as the file
        all_pmids = set()
        for d in session.query(Document).filter(Document.corpora.any(name=base_file)):
            session.delete(d)
        for corpus in session.query(Corpus).filter(Corpus.name == base_file):
            session.delete(corpus)

        session.commit()

        with open(base_file, 'r') as f:
            for l in f:
                values = l.strip().split('\t')
                pmids = values[4].split(',')
                all_pmids.update(pmids)
        print("# of documents: {}".format(len(all_pmids)))
        articles = get_multiple_pmids(all_pmids, base_file)

    elif sys.argv[1] == "add_relations":
        # parse file and add relationships to the database
        # also write two files to be used as gold standard for NER and RE
        for relation in session.query(Pair).join(Document).filter(Document.corpora.any(name=base_file)):
            session.delete(relation)
        session.commit()
        corpus = session.query(Corpus).filter(Corpus.name == base_file).one()
        doc_entities = {}
        doc_pairs = {}
        all_pairs = set()
        with open(base_file, 'r') as f:
            for l in f:
                values = l.strip().split('\t')
                pmids = values[4].split(',')
                first = values[3]
                gene = values[0]
                cell = values[1]
                pairtype = values[2]
                # normalize entities: off by default
                if len(sys.argv) > 2 and sys.argv[2] == "normalize":
                    q = session.query(Normalization).filter(Normalization.text == gene) \
                        .filter(Normalization.entity_type == "cytokine").first()
                    if q:
                        # print(q)
                        gene = q.reference_name
                    else:
                        print("no normalization: {}".format(gene))
                    q = session.query(Normalization).filter(Normalization.text == cell) \
                        .filter(Normalization.entity_type == "cell").first()
                    if q:
                        # print(q)
                        cell = q.reference_name
                    else:
                        print("no normalization: {}".format(cell))

                if first == "cell":
                    pair = (cell, gene)
                else:
                    pair = (gene, cell)
                all_pairs.add(pair)
                # print(pair, pmids)
                for pmid in pmids:
                    if pmid not in doc_entities:
                        doc_entities[pmid] = set()
                    doc_entities[pmid].add((cell, "cell"))
                    doc_entities[pmid].add((gene, "cytokine"))
                document_cells = []
                document_cytokines = []
                # any instance of the cytokine in the documents
                for entity1 in session.query(Entity,Sentence).join(Sentence).filter(Sentence.document_id.in_(pmids))\
                    .filter(Entity.type == "cytokine").filter(Entity.normalized == gene).all():
                    #document_cytokines.append(entity)
                    pmid = entity1[0].sentence.document.pmid
                    #print(entity1[0].corpus_id, entity1[0].text, entity1[0].sentence_id, cell)
                    for entity2 in session.query(Entity,Sentence).join(Sentence).filter(Sentence.id == entity1[0].sentence_id) \
                            .filter(Entity.type == "cell").filter(Entity.normalized == cell).all():
                        # print(entity2)
                        if pmid not in doc_pairs:
                            doc_pairs[pmid] = set()
                        if first == "cell":
                            new_pair = Pair(entity1_id=entity2[0].id, entity2_id=entity1[0].id,
                                            source="smil", type="cytokine-cell",
                                            values=values[2], document_id=pmid)
                            doc_pairs[pmid].add((entity2[0].normalized, entity1[0].normalized, pairtype))

                        else:
                            new_pair = Pair(entity1_id=entity1[0].id, entity2_id=entity2[0].id,
                                            source="smil", type="cytokine-cell",
                                            values=values[2], document_id=pmid)
                            doc_pairs[pmid].add((entity2[0].normalized, entity1[0].normalized, pairtype))
                        # print("adding", entity1, entity2)
                        corpus.pairs.append(new_pair)
                        session.add(new_pair)
        session.commit()

        # write gold standard files
        with open(base_file + ".cytokine", 'w') as entities_file:  # file to write which entities appear on each document
            for d in doc_entities:
                #entities_file.write(d + '\n')
                for e in doc_entities[d]:
                    if e[1] == "cytokine":
                        entities_file.write("\t".join((d, e[0], e[1])) + '\n')
        with open(base_file + ".cell", 'w') as entities_file:  # file to write which entities appear on each document
            for d in doc_entities:
                #entities_file.write(d + '\n')
                for e in doc_entities[d]:
                    if e[1] == "cell":
                        entities_file.write("\t".join((d, e[0], e[1])) + '\n')

        positive_file = open(base_file + ".positive", 'w')
        negative_file = open(base_file + ".negative", 'w')

        with open(base_file + ".pairs", 'w') as pairs_file:  # file to write which pairs appear on each document
            # entity names are normalized
            for d in doc_pairs:
                #print(d)
                #pairs_file.write(str(d) + '\n')
                unique_pairs = set()
                for e in doc_pairs[d]:
                    #if (e[0], e[1]) not in unique_pairs:
                    pairs_file.write("\t".join((str(d), e[0], e[1])) + '\n')
                    #unique_pairs.add((e[0], e[1]))
                    # print(e[2])
                    if e[2] == "Positive" or e[2] == "Unknown":
                        positive_file.write("\t".join((str(d), e[0], e[1])) + '\n')
                    elif e[2] == "Negative" or e[2] == "Unknown":
                        negative_file.write("\t".join((str(d), e[0], e[1])) + '\n')

        positive_file.close()
        negative_file.close()
        with open(base_file + ".bags", 'w') as bags_file:
            for p in all_pairs:
                bags_file.write("\t".join(p) + '\n')

    elif sys.argv[1] == "evaluate_relations":
        # compare relations of immunoexpresso with results file
        results_file = sys.argv[2]
        threshold = 0
        n_docs = 0
        if len(sys.argv) > 3:
            threshold = float(sys.argv[3])
            if threshold > 1:
                n_docs = threshold
                threshold = 0
        gold_standard = set()
        #with open(base_file + ".pairs", 'r') as f:
        #    for l in f:
        #        values = l.strip().split("\t")
                #if values[4]
        #        gold_standard.add((values[1], values[2]))
        with open(base_file + ".pairs", 'r') as f:
            for l in f:
                values = l.strip().split('\t')
                #if values[3] == "cell":
                #    gold_standard.add((values[0], values[1]))
                #else:
                #    gold_standard.add((values[1], values[0]))
                gold_standard.add(('pairs', values[1], values[2]))

        results = set()
        with open(results_file, 'r') as f:
            next(f)
            for l in f:
                values = l.split("\t")
                if threshold > 0 and float(values[5]) < threshold:
                    continue
                elif n_docs > 0 and int(values[7]) < n_docs:
                    continue
                results.add(('pairs', values[0], values[1]))

        fps = results - gold_standard
        fns = gold_standard - results
        tps = results & gold_standard
        # print(len(fps), len(fns), len(tps))
        precision, recall, fmeasure = 0, 0, 0
        if len(tps) + len(fps) > 0:
            precision = len(tps) / (len(tps) + len(fps))
        if len(tps) + len(fns) > 0:
            recall = len(tps) / (len(tps) + len(fns))
        if recall + precision > 0:
            fmeasure = 2 * precision * recall / (recall + precision)
        # print("fps:", random.sample(fps, 10))
        # print("fns:", random.sample(fns, 10))
        # print("fps:", fps)
        # print("fns:", fns)
        print(threshold, precision, recall, fmeasure)
        write_results(fps, fns, tps, results_file + ".report")

    elif sys.argv[1] == "evaluate_docs":
        # compare relations obtained on each document with immunoexpresso
        results_file = sys.argv[2]
        threshold = 0
        if len(sys.argv) > 2:
            threshold = float(sys.argv[3])
        gold_standard = set()
        with open(base_file + ".pairs", 'r') as f:
            for l in f:
                values = l.strip().split("\t")
                gold_standard.add((str(values[0]), values[1], values[2]))

        #with open(base_file, 'r') as f:
        #    for l in f:
        #        values = l.strip().split("\t")

        #        for pmid in values[4].split(","):
        #            gold_standard.add((pmid, values[1], values[0]))

        results = set()
        y_true = []
        probs = []
        with open(results_file, 'r') as f:
            next(f)
            for l in f:
                if l.strip():
                    values = l.strip().split("\t")
                    probs.append(float(values[6]))
                    if (values[7], values[0], values[1]) in gold_standard:
                        y_true.append(1)
                    else:
                        y_true.append(0)
                    #for pmid in values[6].split(","):
                        #if values[3] == "cytokine":
                    #    score = float(values[5])
                    #    if score < 0.999:
                     #       continue
                    # print(values)
                    if float(values[6]) >= threshold:
                        results.add((values[7], values[0], values[1]))
                    #else:
                    #    results.add((pmid, values[1], values[0]))
        for g in gold_standard:
            if g not in results:
                y_true.append(1)
                probs.append(-1)
        fps = results - gold_standard
        fns = gold_standard - results
        tps = results & gold_standard
        #print(len(fps), len(fns), len(tps))
        precision, recall, fmeasure = 0, 0, 0
        if len(tps) + len(fps) > 0:
            precision = len(tps)/(len(tps) + len(fps))
        if len(tps) + len(fns) > 0:
            recall = len(tps) / (len(tps) + len(fns))
        if recall + precision > 0:
            fmeasure = 2*precision*recall/(recall+precision)
        #print("fps:", random.sample(fps, 10))
        #print("fns:", random.sample(fns, 10))
        #print("fps:", fps)
        #print("fns:", fns)
        print(threshold, precision, recall, fmeasure)
        write_results(fps, fns, tps, results_file + ".report")
        roc_score = roc_auc_score(y_true, probs)
        print("roc:", roc_score)
        average_precision = average_precision_score(y_true, probs)
        print("avg precision", average_precision)
        precision, recall, _ = precision_recall_curve(y_true, probs)

        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
            average_precision))
        plt.savefig("{}_PRcurve.png".format(sys.argv[2]))

        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        print("roc auc", roc_auc)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig("{}_ROC curve.png".format(sys.argv[2]))

    elif sys.argv[1] == "evaluate_entities": # evaluate against entities present in the relations
        results = set()
        corpus = session.query(Corpus).filter(Corpus.name == base_file).one()
        for e in session.query(Entity).filter(Entity.corpus_id == corpus.id):
            #if e.corpus_id == base_file:
            #    print(e)
            results.add((str(e.sentence.document_id), e.normalized, e.type))
            #results.add(str(e.sentence.document_id))
        gold_standard = set()
        with open(base_file + ".entities", 'r') as f:
            for l in f:
                values = l.strip().split('\t')
                gold_standard.add(tuple(values))
                #gold_standard.add(values[0])


        fps = results - gold_standard
        fns = gold_standard - results
        tps = results & gold_standard
        precision = len(tps) / (len(tps) + len(fps))
        recall = len(tps) / (len(tps) + len(fns))
        #print("fps:", random.sample(fps, 10))
        #print("fns:", random.sample(fns, 10))
        print(len(fps), len(fns), len(tps))
        print(precision, recall)
        write_results(fps, fns, tps, "immunoexpresso_ner.report")

    elif sys.argv[1] == "extend_corpus":
        # use documents from base_file and get all the entities related to those documents,
        # and write to base file again
        full_docs_file = "immunoexpresso.docs"
        all_pmids = set()
        with open(base_file, 'r') as f:
            for l in f:
                values = l.strip().split('\t')
                pmids = values[4].split(',')
                all_pmids.update(pmids)
        print("# of documents: {}".format(len(all_pmids)))
        relations = {}
        with open(full_docs_file, 'r') as f:
            for l in f:
                values = l.strip().split('\t')
                relation_pmids = values[4].split(',')
                relations['\t'.join(values[:4])] = [pmid for pmid in relation_pmids if pmid in all_pmids]
        with open('full_' + base_file, 'w') as f:
            for r in relations:
                if relations[r]: # prevent writing relations with no docs
                    f.write(r + '\t' + ','.join(relations[r]) + '\n')
        with open('full_' + relations_file, 'w') as f:
            for r in relations:
                if relations[r]: # prevent writing relations with no docs
                    f.write(r + '\t' + ','.join(relations[r]) + '\n')


if __name__ == "__main__":
    main()
