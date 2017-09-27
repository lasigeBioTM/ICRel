from sqlalchemy import UniqueConstraint
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Table, Boolean, Text
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

class Normalization(Base):
    # Table to store normalized forms of entities
    __tablename__ = "normalization"
    id = Column(Integer, primary_key=True)
    text = Column(Text)
    reference_name = Column(Text)
    entity_type = Column(String(32), default="")
    reference_source = Column(String(32), default="")


class Pair(Base):
    __tablename__ = 'pairs'
    id = Column(Integer, primary_key=True)
    type = Column(String(32), default="")
    source = Column(String(32), default="")
    values = Column(String(32), default="")

    entity1_id = Column(Integer, ForeignKey('entities.id'))
    entity2_id = Column(Integer, ForeignKey('entities.id'))
    document_id =  Column(Integer, ForeignKey('documents.pmid'))
    corpus_id = Column(Integer, ForeignKey('corpora.id'))

    entity1 = relationship("Entity", foreign_keys=[entity1_id]) #, cascade="all, delete, delete-orphan", single_parent=True)
    entity2 = relationship("Entity", foreign_keys=[entity2_id]) #, cascade="all, delete, delete-orphan", single_parent=True)

    document = relationship("Document", foreign_keys=[document_id])
    corpus = relationship("Corpus", foreign_keys=[corpus_id])

class Entity(Base):
    __tablename__ = 'entities'
    id = Column(Integer, primary_key=True)
    start = Column(Integer)
    end = Column(Integer)
    ner = Column(String(32))
    type = Column(String(32), default="")
    text = Column(Text, default="")
    normalized = Column(Text, default="")

    sentence_id = Column(Integer, ForeignKey('sentences.id'))
    corpus_id = Column(Integer, ForeignKey('corpora.id'))
    start_token_id = Column(Integer, ForeignKey('tokens.id'))
    end_token_id = Column(Integer, ForeignKey('tokens.id'))
    __table_args__ = (UniqueConstraint('start', 'end', 'type', 'ner', 'sentence_id', 'corpus_id', name='_entity_annotation'),
                      )
    sentence = relationship("Sentence", back_populates="entities")
    token_start = relationship("Token", foreign_keys=[start_token_id])
    token_end = relationship("Token", foreign_keys=[end_token_id])
    corpus = relationship("Corpus", back_populates="entities")
    def __repr__(self):
        return "<Entity(start='%s', end='%s', text='%s')>" % (
            self.start, self.end, self.text)

class Token(Base):
    __tablename__ = 'tokens'
    id = Column(Integer, primary_key=True)
    start = Column(Integer) #sentence offset
    end = Column(Integer)
    text = Column(Text, default="")
    order = Column(Integer)
    pos = Column(String(32), default="")
    lemma = Column(Text, default="")
    sentence_id = Column(Integer, ForeignKey('sentences.id'))

    sentence = relationship("Sentence", back_populates="tokens")
    #entities_start = relationship("Entity", back_populates="token_start",
    #                      cascade="all, delete, delete-orphan", foreign_keys="")
    #entities_end = relationship("Entity", back_populates="token_end",
    #                              cascade="all, delete, delete-orphan")
    def __repr__(self):
        return "<Token(start='%s', end='%s', text='%s')>" % (
            self.start, self.end, self.text)

class Sentence(Base):
    __tablename__ = 'sentences'
    id = Column(Integer, primary_key=True)
    offset = Column(Integer)
    section = Column(String(10))
    text = Column(Text, default="")
    order = Column(Integer)
    document_id = Column(Integer, ForeignKey('documents.pmid'))

    document = relationship("Document", back_populates="sentences")
    tokens = relationship("Token", order_by=Token.order, back_populates="sentence",
                             cascade="all, delete, delete-orphan")
    entities = relationship("Entity", order_by=Entity.start, back_populates="sentence",
                            cascade="all, delete, delete-orphan")

CorpusDocument = Table('CorpusDocument', Base.metadata,
                       Column('id', Integer, primary_key=True),
                       Column('document_id', Integer, ForeignKey("documents.pmid")),
                       Column('corpus_id', Integer, ForeignKey("corpora.id"))
                       )

class Document(Base):
    __tablename__ = 'documents'
    pmid = Column(Integer, primary_key=True)
    title = Column(Text)
    abstract = Column(Text)
    parsed = Column(Boolean, default=0)
    #corpus_id = Column(Integer, ForeignKey('corpora.id'))

    corpora = relationship("Corpus", secondary=CorpusDocument, back_populates="documents")
    #corpus = relationship("Corpus", back_populates="documents")
    sentences = relationship("Sentence", order_by=Sentence.order, back_populates="document",
                             cascade="all, delete, delete-orphan")

    pairs = relationship("Pair", back_populates="document",
                             cascade="all, delete, delete-orphan")
    def __repr__(self):
        return "<Document(id='%s', title='%s', abstract='%s')>" % (
        self.pmid, self.title, self.abstract)

class Corpus(Base):
    __tablename__ = 'corpora'
    id = Column(Integer, primary_key=True)
    name = Column(String(32))
    description = Column(String(32))
    documents = relationship("Document", secondary=CorpusDocument, back_populates="corpora")
    #documents = relationship("Document", order_by=Document.pmid, back_populates="corpus",
    #                         cascade="all, delete, delete-orphan")
    entities = relationship("Entity", back_populates="corpus")
    pairs = relationship("Pair", back_populates="corpus")
    def __repr__(self):
        return "<Corpus(name='%s')>" % self.name

if __name__ == "__main__":
    with open("config/database.config", 'r') as f:
        for l in f:
            if l.startswith("username"):
                username = l.split("=")[-1].strip()
            elif l.startswith("password"):
                password = l.split("=")[-1].strip()
    #engine = create_engine('sqlite:///database.sqlite', echo=False)
    engine = create_engine('mysql+pymysql://{}:{}@localhost/immuno?charset=utf8mb4'.format(username, password), echo=True)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)