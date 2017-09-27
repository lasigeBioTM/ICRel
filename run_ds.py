import os
import time
import logging
from random import shuffle
import sys
from memory_profiler import profile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

#from config.corpus_paths import paths
from database_schema import Corpus, Document, Entity, Sentence
from multiinstance import MILClassifier

# read location of documents and annotations from configuration file or command line
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

def run_crossvalidation(datasets, folds, pairtype, kb, modelname, ner):
    found_relations = {} # relation string -> docs list
    print("retrieving corpus docs...")
    all_docs = []

    # use just one dataset for now
    all_docs = session.query(Document).filter(Document.corpora.any(name=datasets[0])).all()
    corpus = session.query(Corpus).filter(Corpus.name == datasets[0]).one()

    print("shuffling...")
    shuffle(all_docs)
    partition_size = int(len(all_docs)/folds)
    # each partition i is from i*partition_size to (i+1)*paritition_size
    for i in range(folds):
        print("CV fold {}/{}".format(i+1, folds))
        train_file_dir = "temp/cv{}_{}".format(i, folds)
        test_docs = all_docs[i*partition_size:(i+1)*partition_size]
        print(len(test_docs), [d.pmid for d in test_docs[:10]])
        train_docs = [doc for doc in all_docs if doc not in test_docs]
        print(len(train_docs), [d.pmid for d in train_docs[:10]])

        # train
        train_model = MILClassifier(None, pairtype, ner="all", modelname=modelname)
        train_model.corpus_id = corpus.id
        train_model.load_kb(kb)
        train_model.generateMILdata(test=False, docs=train_docs)
        if os.path.exists(train_file_dir):
            os.remove(train_file_dir)
        train_model.write_to_file(train_file_dir)
        #sys.exit()
        train_model.train(save=False)


        test_model = MILClassifier(None, pairtype, ner="all",
                                   modelname=modelname)
        test_model.corpus_id = corpus.id

        test_model.generateMILdata(test=True, docs=test_docs)
        #test_model.load_classifier()
        test_model.vectorizer = train_model.vectorizer
        test_model.classifier = train_model.classifier

        test_model.test()
        logging.info("getting predictions...")
        results = test_model.get_predictions(len(test_docs))


        write_results(results, pairtype, "results/{}_{}_cv{}_{}_{}".format(modelname, ",".join(datasets),
                                                                           i, folds, pairtype))

        for p in results:
            # merge results
            if p not in found_relations:
                found_relations[p] = []
            found_relations[p] += results[p]
            #print(relation, ','.join(found_relations[relation]))
            #for instance in results[p]:
                #sentence = session.query(Sentence).filter(Sentence.id == instance[0]["sentence_id"]).one().text
                #print(p, instance[0]["document_id"], instance[0]["sentence_id"], instance[2])
            #print()

    write_results(found_relations, pairtype,
                  "results/{}_{}_cv{}_{}".format(modelname, ",".join(datasets), folds, pairtype))

def write_results(results, pairtype, outputfilename):
    """
    Write distant supervison results to file
    :param results: output of get_predictions
    :param pairtype:
    :param outputfilename:
    :return:
    """
    results_file = open(outputfilename, 'w')
    details_file = open(outputfilename + ".details", 'w')
    docs_file = open(outputfilename + ".docs", 'w')
    results_file.write("cell\tcytokine\tvalue\torigin\ttarget\tscore\tdocs\tndocs\n")  # header
    details_file.write("cell <-> cytokine\tsentences\n")
    docs_file.write("cell\tcytokine\tvalue\torigin\ttarget\tmax_score\tscore\tdoc\tsentence\n")
    found_relations = {}
    for p in results:
        # print(results[p])
        pmids = [str(x[0]["document_id"]) for x in results[p]]
        sentences = [str(x[2]) for x in results[p]]
        # print(pmids)
        score = max([i[3] for i in results[p]])

        if results[p][0][0]["type"] == "cell":  # cell is first column
            # results_file.write("{}\t{}\tPositive\t{}\t{}\n".format(p[0], p[1], results[p][0][0]["type"], pmids))
            relation = "{}\t{}\t{}\t{}\t{}\t{}\t".format(p[0], p[1], pairtype, results[p][0][0]["type"],
                                                     results[p][0][1]["type"], score)
        else:
            # results_file.write("{}\t{}\tPositive\t{}\t{}\n".format(p[1], p[0], results[p][0][0]["type"], pmids))
            relation = "{}\t{}\t{}\t{}\t{}\t{}\t".format(p[1], p[0], pairtype, results[p][0][0]["type"],
                                                             results[p][0][1]["type"], score)
        sentence_ids = set()
        for x in results[p]:
            #logging.info((p, x))
            if x[0]["sentence_id"] not in sentence_ids:
                docs_file.write(relation + str(x[3]) + '\t' + str(x[0]["document_id"]) + '\t' + x[2] + '\n')
                sentence_ids.add(x[0]["sentence_id"])

        if relation not in found_relations:
            found_relations[relation] = [[], []]
        found_relations[relation][0] += pmids
        found_relations[relation][1] += sentences


    for r in found_relations:
        # print(found_relations[r])
        sentence_docs = set([x[0] + '\t' + x[1] for x in zip(found_relations[r][0], found_relations[r][1])])
        results_file.write(r + ','.join(found_relations[r][0]) + '\t' + str(len(found_relations[r][0])) + '\n')
        details_file.write(' <-> '.join(r.split('\t')[:2]) + "\t" + str(len(sentence_docs)) +
                           "\n" + '\n'.join(sentence_docs) + "\n\n")
    results_file.close()
    details_file.close()
    docs_file.close()

def main():
    start_time = time.time()
    # try using configargparse for config files
    try:
        import configargparse
        parser = configargparse.ArgumentParser(description='')
    except ImportError:
        import argparse
        parser = argparse.ArgumentParser(description='')
    # action is train DS classifier
    parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')
    parser.add_argument("--trainsets", default="", dest="trainsets", nargs="+",
                        help="Gold standard to be used for training.")
    parser.add_argument("--testsets", default="", dest="testsets", nargs="+",
                        help="Gold standard to be used for testing.")
    parser.add_argument("--modelname", dest="modelname", help="model destination path, without extension")
    parser.add_argument("--entitytype", dest="etype", help="type of entities to be considered", default="all")
    parser.add_argument("--pairtype", dest="ptype", help="type of pairs to be considered", default="all")
    parser.add_argument("--doctype", dest="doctype", help="type of document to be considered", default="all")
    parser.add_argument("--kb", dest="kb", help="knowledge base containing relations")
    parser.add_argument("-o", "--output", "--format", dest="output",
                        nargs=2, help="format path; output formats: xml, html, tsv, text, chemdner.")
    parser.add_argument("--prepare_data", action='store_true')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--cv", action='store', default=0, type=int)
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
    #logging.info("Processing action {0} on {1}".format(options.actions, options.goldstd))
    # assume corpus with entity annotations already exists
    # retrieve corpus


    # retrieve knowledge base
    # retrieve documents and entities of corpus
    #for doc in session.query(Document).join(Corpus).filter(Corpus.name == options.goldstd):
    #    continue

    #train_model.load_kb("corpora/transmir/transmir_relations.txt")

    train_file_dir = "temp/mil.train"
    if options.prepare_data and options.train and options.test:
        logging.info("training with {} and testing on {}".format(options.trainsets[0], options.testsets[0]))
        train_corpus = session.query(Corpus).filter(Corpus.name == options.trainsets[0]).one()
        test_corpus = session.query(Corpus).filter(Corpus.name == options.testsets[0]).one()

        train_docs = session.query(Document).filter(Document.corpora.any(name=options.trainsets[0])).all()
        test_docs = session.query(Document).filter(Document.corpora.any(name=options.testsets[0])).all()
        # train with just one gold standard


        print("test:", len(test_docs), [d.pmid for d in test_docs[:10]])
        print("train", len(train_docs), [d.pmid for d in train_docs[:10]])

        # train
        train_model = MILClassifier(None, options.ptype, ner="all", modelname=options.modelname)
        train_model.corpus_id = train_corpus.id
        train_model.load_kb(options.kb)
        train_model.generateMILdata(test=False, docs=train_docs[:5000])
        if os.path.exists(train_file_dir):
            os.remove(train_file_dir)
        train_model.write_to_file(train_file_dir)
        # sys.exit()
        train_model.train(save=True)
        # test
        # test_model = MILClassifier(None, pairtype, ner="all",
        #                           modelname=modelname)
        # test_model.load_kb("corpora/transmir/transmir_relations.txt")
        fold_results = {}
        # send doc my doc
        # base_test_model = MILClassifier(None, pairtype, ner="all",
        #                           modelname=modelname)
        # base_test_model.corpus_id = corpus.id
        # base_test_model.load_classifier()
        # base_test_model.vectorizer = train_model.vectorizer
        # base_test_model.classifier = train_model.classifier
        # test_model = train_model
        test_model = MILClassifier(None, options.ptype, ner="all",
                                   modelname=options.modelname)
        test_model.corpus_id = test_corpus.id
        # base_test_model.load_classifier()
        test_model.vectorizer = train_model.vectorizer
        test_model.classifier = train_model.classifier
        # test_model.load_kb(kb)
        # test_model = base_test_model
        test_model.generateMILdata(test=True, docs=test_docs)
        test_model.test()

        logging.info("getting predictions...")
        results = test_model.get_predictions(len(test_docs))


        write_results(results, options.ptype, "results/{}_{}_{}".format(options.modelname,
                                                                                     ",".join(options.testsets),
                                                                                      options.ptype))

    elif options.prepare_data:
        if os.path.isfile(train_file_dir):
            os.remove(train_file_dir)
        for goldstd in options.trainsets:
            print("processing", goldstd)
            train_model = MILClassifier(goldstd, options.ptype, ner="all")
            train_model.load_kb(options.kb)
            train_model.generateMILdata(test=False)
            train_model.write_to_file(train_file_dir)

    elif options.train:
        print("training...")
        train_model = MILClassifier(None, options.ptype, ner="all", modelname=options.modelname)
        train_model.load_from_file(train_file_dir)
        train_model.train()

    elif options.test:
        for goldstd in options.testsets:
            all_results = {}
            all_docs = session.query(Document).filter(Document.corpora.any(name=goldstd)).all()
            corpus = session.query(Corpus).filter(Corpus.name == goldstd).one()
            for test_doc in all_docs:
                test_model = MILClassifier(None, options.ptype, ner="all",
                                           modelname=options.modelname)
                test_model.corpus_id = corpus.id
                # test_model.load_kb("corpora/transmir/transmir_relations.txt")
                test_model.generateMILdata(test=True, docs=[test_doc])
                test_model.write_to_file(train_file_dir + ".test")
                test_model.load_classifier()
                # test_model.vectorizer = train_model.vectorizer
                # test_model.classifier = train_model.classifier

                test_model.test()
                results = test_model.get_predictions()
                for p in results:
                     if p not in all_results:
                         all_results[p] = []
                     all_results[p] += results[p]
            #results.path = options.results + "-" + options.test[i]
            #results.save(options.results + "-" + options.test[i] + ".pickle")
            write_results(all_results, options.ptype,
                          "results/{}_on_{}_{}.txt".format(options.modelname, goldstd, options.ptype))

    if options.cv > 0:
        run_crossvalidation(options.trainsets, folds=options.cv, pairtype=options.ptype, kb=options.kb,
                            modelname=options.modelname, ner="all")


    total_time = time.time() - start_time
    print
    "Total time: %ss" % total_time

if __name__ == "__main__":
    main()