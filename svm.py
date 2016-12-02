from xml_to_html import *
from sklearn import svm
summary_triplet, doc_triplet,file = triplet_creation()
doc_name, common_triplet = common_triplets(summary_triplet,doc_triplet,file)

X = [file,doc_name,len(doc_name)]
y = [common_triplets()]
clf = svm.SVC()
clf.fit(X, y)


def triplet_creation():
    for dirname, dirnames, filenames in os.walk('/home/surabhi/PycharmProjects/nlp_project/NLP/test_data'):

        for filename in filenames:
            file = os.path.join(dirname, filename)
            print file
            with open(file) as f:
                entity_names = []
                sample = f.read()
                chunked_sentences = []
                sentences = nltk.sent_tokenize(sample)
                tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
                tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
                for sentence in tagged_sentences:
                    chunkParser = nltk.RegexpParser(chunkgram)
                    chunked = chunkParser.parse(sentence)
                    chunked_sentences.append(chunked)
                for tree in chunked_sentences:
                    entity_names.extend(extract_entity_names(tree))
                summary_sentences = []
                for sentence in sentences:
                    for entity in entity_names:
                        doc_triplet.append(entity)
                        # if entity in sentence and entity in summary_triplet:
                        # # if entity in sentence:
                        #     if sentence not in summary_sentences:
                        #         summary_sentences.append(sentence)

        print len(summary_sentences)
        for i in summary_sentences:
            print i
            print ("\n")

        print clf.predict(file,doc_name,len(doc_name),summary_triplet)