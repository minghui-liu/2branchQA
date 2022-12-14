import pickle
import json

# read triples.json
# with open('triples.json', 'rb') as f:
#     triples = json.load(f)

# print(len(triples))

# # save triples.json as triples.pickle
# # with open('triples.pickle', 'wb') as f:
# #     pickle.dump(triples, f)

# read triples_train pickle
with open('data/qblink_dev', 'rb') as f:
    data = pickle.load(f)

print(type(data))
print(len(data))
print(len(data[0]))
print(data[0])