import numpy as np
import json
import nltk
from nltk.corpus import stopwords
import pickle


def get_embedding(tokens, embedding_dict, embedding_dim=300):
    embedding = []
    # stop_words = set(stopwords.words('english'))
    for token in tokens:
        if token not in stop_words:
            if token in embedding_dict:
                embedding.append(embedding_dict[token])
            else:
                embedding.append(np.zeros(embedding_dim)) # 300 is the dimension of GloVe embedding
    return embedding


questions = []
# read qanta_dev_question.json
with open('qanta_dev_question.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        questions.append(data)

# print len of questions
print(f"Loaded {len(questions)} questions")


# remove stopwords
stop_words = set(stopwords.words('english'))

embedding_dict = {}

# read glove.840B.300d.txt
# with open('glove.840B.300d.txt', 'r') as f:
#     for line in f:
#         values = line.split(' ')
#         word = values[0]
#         vector = np.asarray(values[1:], "float32")
#         embedding_dict[word] = vector
        
# save embedding_dict to disk using pickle
# with open('embedding_dict.pickle', 'wb') as handle:
#     pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load embedding_dict from disk using pickle
with open('embedding_dict.pickle', 'rb') as handle:
    embedding_dict = pickle.load(handle)
# # print len of embedding_dict
print(f"Loaded {len(embedding_dict)} word vectors")


# read graph data
graph = {}
with open('qanta_dev_graph.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        graph[data['id']]= data

print(f'Loaded {len(graph)} graphs')


counter = 0 # debug counter
for q in questions:
    if q['qanta_id'] not in graph:
        id = q['qanta_id']
        # print(f"Graph of question {id} not found, skipped")
        continue

    answer = q['page']
    g = graph[q['qanta_id']]
    # # tokenize question with nltk
    # tokens = nltk.word_tokenize(q['text'])
    # # print(tokens)
    # # convert tokens to embedding
    # embedding = get_embedding(tokens, embedding_dict)
    # # average embedding
    # avg_embedding = np.mean(embedding, axis=0)
    # print(avg_embedding.shape)

    # get list of question entities
    q_ets = g['q_et']
    # get list of positive entities
    pos_ets = g['pos_et']
    # get list of negative entities
    neg_ets = g['neg_ets']

    print(len(q_ets), len(pos_ets), len(neg_ets))
    # print name of postive entities
    # for et in pos_ets:
    #     print(et)
    
    counter += 1
    # break

# print(graph[0].keys())

# print(graph[0]['id'])
# print(graph[0]['q_et'])
# print(graph[0]['q_et'][0])

# print(graph[0]['pos_et'])
# print(graph[0]['neg_ets'][0].keys())

# print(len(graph[0]['q_et']))
# print(graph[0]['q_et'][0])
# print(graph[0]['q_et'][1])
# print(graph[0]['q_et'][2])

