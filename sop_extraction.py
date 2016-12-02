import nltk
import os
# nltk.download('maxent_treebank_pos_tagger')
chunkgram =  r"""Chunk:
{<DT>?<JJ>?<N.*>+<NN*>*<V.*>*<N.*>*}
"""
x = []

def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'Chunk':
            entity_names.append(' '.join([child[0] for child in t]))

        else:
            for child in t:

                entity_names.extend(extract_entity_names(child))

    return entity_names

with open('/home/surabhi/PycharmProjects/nlp_project/NLP/1_data.txt') as f:
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
    print entity_names

x = ['an analysis', 'the semantic interpretation', 'intensional verbs', 'seek', 'direct objects', 'either individual', 'type', 'readings', 'the quantifier case', 'quantifying-in rules', 'This simple account follows', 'use', 'logical deduction', 'linear logic', 'the relationship', 'syntactic structures', 'meanings', 'analysis resembles', 'categorial approaches', 'important ways', 'type flexibility', 'categorial semantics', 'a precise connection', 'a result', 'derivations', 'certain readings', 'sentences', 'intensional verbs', 'direct objects', 'categorial accounts', 'the syntax-semantics interface', 'The analysis forms', 'a part', 'ongoing work', 'semantic interpretation', 'the framework', 'Lexical-Functional Grammar']
summary_sentences = []
for sentence in sentences:
    for entity in entity_names:
        if entity in sentence and entity in x:
        #if entity in sentence:
            if sentence not in summary_sentences:
                summary_sentences.append(sentence)

print len(summary_sentences)
for i in summary_sentences:
    print i
    print ("\n")

