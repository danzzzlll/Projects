import generation
import pickle

data = open('./data/master_margo.txt').read()

result = generation.generate(data)

f = open('pickle.bin', 'wb')
pickle.dump(result, f)

print(result)
#nf = open('pickle.bin', 'rb')
#print(pickle.load(nf))