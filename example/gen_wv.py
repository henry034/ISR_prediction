import numpy as np
from bert_serving.client import BertClient

def main():
    bc = BertClient()
    with open('./data/71_final.txt', 'r') as f:
        content = []
        for line in f:
            line = line.strip()
            content.append(line)
    vec = bc.encode(content)
    np.save('./data/71_final.npy',vec)

if __name__ == '__main__':
    main()
