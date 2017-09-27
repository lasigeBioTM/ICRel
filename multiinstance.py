from __future__ import unicode_literals
import os
import logging
import pickle
import sys
from collections import Counter
from memory_profiler import profile
#from profilehooks import profile
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database_schema import Corpus, Document, Entity, Sentence, Token


import gc
import numpy
import math
import misvm
#from nltk import Tree
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline

from subprocess import Popen, PIPE
import platform
import itertools
import codecs

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

class MILClassifier(object):
    def __init__(self, corpus_name, pairtype, modelname="mil_classifier.model", ner="goldstandard"):
        super(MILClassifier, self).__init__()
        self.modelname = modelname
        self.pairtype = pairtype
        if corpus_name:
            self.corpus_id = session.query(Corpus).filter(Corpus.name == corpus_name).one().id

        self.basedir = "./models/"

        self.pairs = {}  # (e1.normalized, e2.normalized) => (e1, e2)
        self.instances = {}  # bags of instances (e1.normalized, e2.normalized) -> all instances with these two entities
        self.labels = {} # (e1.normalized, e2.normalized) => label (-1/1)
        self.bag_labels = []  # ordered list of labels for each bag
        self.bag_pairs = []  # ordered list of pair labels (e1.normalized, e2.normalized)
        self.data = []  # ordered list of bags, each is a list of feature vectors
        self.predicted = []  # ordered list of predictions for each bag
        self.predicted_instances = [] # ordered list of predictions for each instance
        self.resultsfile = None
        self.examplesfile = None
        self.ner_model = ner
        self.relations = None
        #self.vectorizer = CountVectorizer(min_df=0.1, ngram_range=(1, 1), token_pattern=r'\b\w+\-\w+\b')
        #self.vectorizer = CountVectorizer(min_df=0.1, ngram_range=(1, 1), token_pattern=r'[^\s]+')
        #self.vectorizer = HashingVectorizer(ngram_range=(1, 1),
        #                                    token_pattern=r'\b\w+\-\w+\b', )

        #self.vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1), token_pattern=r'\b\w+\-\w+\b')
        self.vectorizer = TfidfVectorizer(min_df=0.01, token_pattern=r'[^\s]+', ngram_range=(1, 1))

        #self.classifier = misvm.MISVM(kernel='linear', C=100, max_iters=20)
        #self.classifier = misvm.sbMIL(kernel='linear', C=100)
        self.classifier = misvm.sMIL(kernel='linear', C=100)
        #self.classifier = misvm.MissSVM(kernel='linear', C=100) #, max_iters=20)
        #if generate:
        #    self.generateMILdata(test=test, pairtype=pairtype, relations=relations)


    def generateMILdata(self, test, docs=None):
        """
        Generate data for self.instances, self.labels, self.pairs dictionaries bag->data
        :param test: True if test mode, false if training
        :return:
        """
        # pairtypes = (config.relation_types[pairtype]["source_types"], config.relation_types[pairtype]["target_types"])
        # pairtypes = (config.event_types[pairtype]["source_types"], config.event_types[pairtype]["target_types"])
        logging.info("generating data...")
        pcount = 0
        truepcount = 0
        strue = 0
        sfalse = 0
        skipped = 0
        #for sentence in corpus.get_sentences(self.ner_model):
        #if docs is None:
        #    docs = session.query(Document).filter(Document.corpus_id == self.corpus_id)
        if docs is None:
            docs = session.query(Document).filter(Document.corpora.any(Corpus.id == self.corpus_id)).all()
        for i, doc in enumerate(docs):
            if i % 1000 == 0:
                logging.info("{}/{} {}".format(i, len(docs), doc.pmid))
            for sentence in session.query(Sentence).filter(Sentence.document_id == doc.pmid):
                # print(sentence.id)
            #for sentence in self.corpus.get_sentences(self.ner_model):
                # doc_entities = corpus.documents[did].get_entities("goldstandard")
                sids = []
                # print len(corpus.type_sentences[pairtype])
                # sentence_models = set([m for m in sentence.entities.elist])
                # print self.ner_model, sentence_models
                self.generate_sentence_data(sentence, test=test)
        # print(self.labels)
        truepcount = len([b for b in self.labels if self.labels[b] == 1])
        pcount = len(self.labels)
        logging.info("True/total relations:{}/{} ({})".format(truepcount, pcount,
                                                              str(1.0 * truepcount / (pcount + 1))))
        # print "total bags:", len(self.instances)

    def write_to_file(self, filepath):
        print("writing train data to {}".format(filepath))
        with codecs.open(filepath, 'a', 'utf-8') as f:
            for bag in self.instances: # one line per bag
                f.write('#'.join(bag) + "\t") # starts with bag pair joined by #
                for i in self.instances[bag]: # write each instance of the pair in the same line
                    f.write(i + "\t") # i consists of the tokens of the pair instance joined by a space and order of pair
                f.write(str(self.labels[bag]) + "\n") # write bag label at the end

    def load_from_file(self, filepath):
        print("loading train data from {}".format(filepath))
        with codecs.open(filepath, 'r', 'utf-8') as f:
            for l in f:
                values = l.split("\t")
                bag = tuple(values[0].split("#"))
                self.instances[bag] = []
                for i in values[1:-1]:
                    self.instances[bag].append(i)
                self.labels[bag] = int(values[-1])

    def load_kb(self, kb_path):
        logging.info("loading KB...")
        self.relations = set()
        with open(kb_path) as rfile:
            for l in rfile:
                values = l.strip().split('\t')
                if self.pairtype == "all" or (len(values) > 2 and self.pairtype == values[2]) or self.pairtype == "Unknown":
                    self.relations.add((values[1], values[0]))
        logging.info("done")

    def load_classifier(self):
        logging.info("loading classifier...")
        #self.classifier = joblib.load("{}/{}/{}.pkl".format(self.basedir, self.modelname, self.modelname))
        #self.vectorizer = joblib.load("{}/{}/{}_bow.pkl".format(self.basedir, self.modelname, self.modelname))
        with open("{}/{}/{}.pkl".format(self.basedir, self.modelname, self.modelname), 'rb') as modelfile:
            self.classifier = pickle.loads(modelfile.read())
        with open("{}/{}/{}_bow.pkl".format(self.basedir, self.modelname, self.modelname), 'rb') as modelfile:
            self.vectorizer = pickle.loads(modelfile.read())

    def generate_vectorizer(self):
        logging.info("Building vocabulary...")
        all_text = []
        # print self.vectorizer
        for pair in self.instances:
            for i in self.instances[pair]:
                all_text.append(i)
        #print(all_text[:5])
        x = self.vectorizer.fit_transform(all_text)
        # print x, self.vectorizer
        #vocab = self.vectorizer.get_feature_names()
        #print vocab
        logging.info([w for w in self.vectorizer.get_feature_names()][:10])

    def vectorize_text(self):
        for pair in self.instances:
            bag = []
            # print self.instances[pair]

            for i in self.instances[pair]:

                x = self.vectorizer.transform([i]).toarray()
                #print(len(x[0]))
                bag.append(x[0])
            #print(len(bag[0]), len(bag))
            self.data.append(bag)
            # print bag
            self.bag_labels.append(self.labels[pair])
            self.bag_pairs.append(pair)

    def train(self, save=True):
        self.generate_vectorizer()
        # self.vectorizer = pickle.load("{}/{}/{}_bow.pkl".format(self.basedir, self.modelname, self.modelname))
        self.vectorize_text()

        # print self.vectorizer
        # sys.exit()

        logging.info("Training with {} bags".format(str(len(self.labels))))
        logging.info("{} instances".format(str(sum([len(d) for d in self.data]))))
        #for d in self.data[:10]:
        #    logging.info(str(len(d)) + " instances")
            #logging.info([len(y) for y in d])
        #    for y in d:
                #logging.info(len([x for x in y if x != 0]))
        #        logging.info(y)

        #logging.info(self.bag_labels[:10])
        # for i, d in enumerate(self.data):
        #     if self.bag_labels[i] == 1:
        #         print self.bag_pairs[i], len(d), self.bag_labels[i]
        #         for pair in self.pairs[self.bag_pairs[i]]:
        #             print self.corpus.get_sentence(pair[0].sid).text
        #         print

        #gc.collect()
        self.classifier.fit(self.data, self.bag_labels)
        #gc.collect()
        if save:
            if not os.path.exists(self.basedir + self.modelname):
                os.makedirs(self.basedir + self.modelname)
            logging.info("Training complete, saving to {}/{}/{}.pkl".format(self.basedir, self.modelname, self.modelname))
            #joblib.dump(self.classifier, "{}/{}/{}.pkl".format(self.basedir, self.modelname, self.modelname))
            #joblib.dump(self.vectorizer, "{}/{}/{}_bow.pkl".format(self.basedir, self.modelname, self.modelname))
            s = pickle.dumps(self.classifier)
            with open("{}/{}/{}.pkl".format(self.basedir, self.modelname, self.modelname), 'wb') as modelfile:
                modelfile.write(s)
            s = pickle.dumps(self.vectorizer)
            with open("{}/{}/{}_bow.pkl".format(self.basedir, self.modelname, self.modelname), 'wb') as modelfile:
                modelfile.write(s)

    def test(self):
        if self.instances:
            self.vectorize_text()
            # print self.data
            #self.predicted, self.predicted_instances = self.classifier.predict(self.data, instancePrediction=True)
            self.predicted = self.classifier.predict(self.data)
            #self.predicted = self.classifier.predict(self.data)
            #self.predicted = [1]*len(self.data)
            logging.info(Counter([round(x, 1) for x in self.predicted]))
            #print(Counter([round(x, 1) for x in self.predicted_instances]))
            # print(self.predicted_instances)

    def annotate_sentences(self, sentences):
        """
        Generate self.data for a list of sentences and then self.test and return list of results for each sentence
        :param sentences: list of sentence objects
        :return:
        """
        for sentence in sentences:
            self.generate_sentence_data(sentence)
            # print "len pairs", self.pairs
        self.test()

    def generate_sentence_data(self, sentence, test=True):
        #pairtypes = (config.relation_types[self.pairtype]["source_types"], config.relation_types[self.pairtype]["target_types"])
        pairtypes = ("all", "all")
        sentence_entities = []
        if self.ner_model == "all":
            #offsets = set()
            #for elist in sentence.entities.elist:
            #    for entity in sentence.entities.elist[elist]:
            #        offset = (entity.dstart, entity.dend)
            #        if offset not in offsets:
            #            sentence_entities.append(entity)
            #            offsets.add(offset)
            sentence_entities = session.query(Entity).filter(Entity.sentence_id == sentence.id)\
                                .filter(Entity.corpus_id == self.corpus_id)\
                                .distinct(Entity.start, Entity.end).all()
            #sentence_entities = sentence.entities
        else:
            sentence_entities = session.query(Entity).distinct(Entity.start_token_id).filter(Entity.ner == self.ner_model)\
                .filter(Entity.sentence_id == sentence.id).all()
        #if len(sentence_entities) > 1:
        #    print(sentence_entities)
        # print self.ner_model, sentence_entities
        #logging.info(sentence_entities)
        for pair in itertools.combinations(sentence_entities, 2):
            if pair[0].type == pair[1].type:
                continue
            #if pair[0].token_start.order + 10 < pair[1].token_start.order:
                # skip entities with distance higher than 5
           #     continue
            if (pair[0].type in pairtypes[0] or pairtypes[0] == "all") and \
                    (pair[1].type in pairtypes[1] or pairtypes[1] == "all"): # and pair[0].normalized_score > 0 and pair[1].normalized_score > 0:

                if pair[0].type == "cytokine":
                    pair = (pair[1], pair[0])
                #if test:
                #    bag = (sentence.did, pair[0].normalized, pair[1].normalized)
                #else:
                #    bag = (pair[0].normalized, pair[1].normalized)
                bag = (pair[0].normalized, pair[1].normalized) #, str(sentence.document_id))
                #print(bag)
                #logging.info((sentence.document_id, sentence.id, pair))
                if bag not in self.instances:
                    # print "creating bag", bag
                    self.instances[bag] = []
                    self.labels[bag] = -1  # assume no relation until it's confirmed
                    self.pairs[bag] = []
                # print "adding pair", pair
                self.pairs[bag].append(pair)
                # if bag[1:] in relations:
                #print((pair[0].normalized, pair[1].normalized))
                #logging.info(list(self.relations)[:10])
                #logging.info((str(sentence.document_id), pair[0].normalized, pair[1].normalized))

                if not test and (pair[0].normalized, pair[1].normalized) in self.relations:
                    #print("true pair")
                    self.labels[bag] = 1

                #else:
                #    print("false pair")
                pair_features = self.get_pair_features(sentence, pair)
                # logging.info((bag, str(self.labels[bag]), pair_features))
                self.instances[bag].append(pair_features)

    def process_sentence(self, sentence):
        """
        return list of relations using sMIL results
        :param sentence:
        :return:
        """
        processed_pairs = []
        for i, pred in enumerate(self.predicted):
            if pred >= 0:
                score = 1.0 / (1.0 + math.exp(-pred))
                bag = self.bag_pairs[i]
                pairs = self.pairs[bag]
                for pair in pairs:
                    # print pair, sentence
                    if pair[0].sid == sentence.sid:
                        pair = sentence.add_relation(pair[0], pair[1], self.pairtype, relation=True)
                        processed_pairs.append(pair)
        return processed_pairs



    def get_predictions(self, ndocs):
        #results = ResultsRE(self.resultsfile)
        #document_pairs = {}
        predicted_pairs = {}
        for i, pred in enumerate(self.predicted):

            bag = self.bag_pairs[i]
            pairs = self.pairs[bag]
            score = 1.0 / (1.0 + math.exp(-pred))
            score = len(set([p[0].sentence.document_id for p in pairs]))/ndocs
            #logging.info(bag) #, [pair[0].sentence.text for pair in pairs])
            for pair in pairs:
                #did = bag[0]
                #did = pair[0].did

                #for p in pair:
                #    logging.info(p.id, p.text, p.normalized, p.sentence_id, p.type)
                pair_instance = ({"id": pair[0].id, "text": pair[0].text, "type":pair[0].type,
                                  "sentence_id": pair[0].sentence_id,
                                  "document_id": pair[0].sentence.document_id},
                                 {"id": pair[1].id, "text": pair[1].text, "type": pair[1].type,
                                  "sentence_id": pair[1].sentence_id,
                                  "document_id": pair[1].sentence.document_id},
                                 pair[0].sentence.text.replace('\n', ' '),
                                 score)
                pair_name = (pair[0].normalized, pair[1].normalized)
                if pair_name not in predicted_pairs:
                    predicted_pairs[pair_name] = []
                predicted_pairs[pair_name].append(pair_instance)
                # print()
        return predicted_pairs


    def get_pair_features(self, sentence, pair):
        start1, end1, start2, end2 = pair[0].token_start.order, pair[0].token_end.order + 1,\
                                     pair[1].token_start.order, pair[1].token_end.order + 1
        # adjust for sentence offset
        start1, end1, start2, end2 = start1 - sentence.order, end1 - sentence.order,\
                                     start2 - sentence.order, end2 - sentence.order
        #retrieve tokens corresponding to entities in this sentence
        sentence_entities_tokens = {}
        for e in sentence.entities:
            for i in range(e.token_start.order, e.token_end.order + 1): # assume no overlaps
                sentence_entities_tokens[i] = e.type

        # token_order1 = [t.order for t in pair[0].tokens]
        # token_order2 = [t.order for t in pair[1].tokens]
        token_order1 = range(start1, end1+1)
        token_order2 = range(start2, end2+1)
        order = "normal-order"
        entitytext = [pair[0].text, pair[1].text]
        if start1 > start2:
            order = "reverse-order"
            #start, end = pair[1].tokens[-1].order, pair[0].tokens[0].order
            start1, end1, start2, end2 = start2, end2, start1, end1
            entitytext = [pair[1].text, pair[0].text]
        before_features = []
        middle_features = []
        end_features = []
        feature_window = 3

        for i, t in enumerate(sentence.tokens[max(start1-feature_window, 0):start1]):
            if t.order in sentence_entities_tokens:
                before_features.append(str(i) + "-before-{}-entity".format(sentence_entities_tokens[t.order]))
            else:
                before_features.append(str(i) + "-before-" + t.lemma.strip().replace("-", ".") + "-" + t.pos)
                #before_features.append(str(i) + "-before-" + t.lemma + "-" + t.pos + "-" + t.tag)
                #before_features.append("before-" + t.pos)

        for i, t in enumerate(sentence.tokens[end1:min(end1+feature_window, start2)]):
            if t.order in sentence_entities_tokens:
                middle_features.append(str(i) + "-middle1-{}-entity".format(sentence_entities_tokens[t.order]))
            else:
                middle_features.append(str(i) + "-middle1-" + t.lemma.strip().replace("-", ".") + "-" + t.pos)
                #middle_features.append(str(i) + "-middle-" + t.lemma  + "-" + t.pos + "-" + t.tag)
                #middle_features.append("middle-" + t.pos)

        for i, t in enumerate(sentence.tokens[end1:max(end1, start2-feature_window)]):
            if t.order in sentence_entities_tokens:
                middle_features.append(str(i) + "-middle2-{}-entity".format(sentence_entities_tokens[t.order]))
            else:
                middle_features.append(str(i) + "-middle2-" + t.lemma.strip().replace("-", ".") + "-" + t.pos)
                #middle_features.append(str(i) + "-middle-" + t.lemma  + "-" + t.pos + "-" + t.tag)
                #middle_features.append("middle-" + t.pos)

        for i, t in enumerate(sentence.tokens[end2:end2+feature_window]):
            if t.order in sentence_entities_tokens:
                end_features.append(str(i) + "-end-{}-entity".format(sentence_entities_tokens[t.order]))
            else:
                end_features.append(str(i) + "-end-" + t.lemma.strip().replace("-", ".") + "-" + t.pos)
                #end_features.append(str(i) + "-end-" + t.lemma  + "-" + t.pos + "-" + t.tag)
                #end_features.append("end-" + t.pos)
        features = before_features + middle_features + end_features #  + [order]
        #features = [f.split("-")[2] + "-" + f.split("-")[1] for f in features if len(f.split("-")[2]) > 1]
        #features = [f.split("-")[2] for f in features if len(f.split("-")[2]) > 0]
        features = [f.split("-")[2] for f in features if sum(c.isalpha() for c in f.split("-")[2]) > 0]
        #for f in features:
        #    if f.count("-") < 3 and "order" not in f:
        #        logging.info(f)
        # print(sentence.text, pair[0].text, pair[1].text, features, start1, end1, start2, end2)

        return " ".join(features)
