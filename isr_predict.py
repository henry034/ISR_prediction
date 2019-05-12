import numpy as np
import os
import pickle
from bert_serving.client import BertClient
from model import *

def load_file(path):
    with open(path, 'r') as f:
        list_data = []
        idx = 0
        for line in f:
            element = line.strip().split('\t')
            pph = element[0]
            isr = float(element[1].split(':')[-1])
            isr_utt = float(element[2].split(':')[-1])
            isr_raw = float(element[3].split(':')[-1])
            s_fname = element[4]
            s_id = int(element[5])

            pph_data = {
                'pph': pph,
                'isr': isr,
            }
            if s_id != idx:
                if idx != 0:
                    list_data.append(sentence_data)
                sentence_data = {
                    'sentence': '',
                    'pph_data': [],
                    's_id': s_id,
                    's_fname': s_fname,
                    'isr_utt': isr_utt,
                    'isr_raw': isr_raw
                }
                idx = s_id
            sentence_data['sentence'] += pph
            sentence_data['pph_data'].append(pph_data)
        list_data.append(sentence_data)
    return list_data
            

def load_data():
    trn_path = './data/with_utt/training_pph_bert.txt'
    test_path = './data/with_utt/test_pph_bert.txt'
    valid_path = './data/with_utt/valid_pph_bert.txt'
    data = {}
    data['trn'] = load_file(trn_path)
    load_vec(data['trn'])
    data['test'] = load_file(test_path)
    load_vec(data['test'])
    data['valid'] = load_file(valid_path)
    load_vec(data['valid'])

    return data


def load_vec(data, vec='BERT'):
    if vec == 'BERT':
        bc = BertClient()
        sentence = []
        for i in data:
            sentence.append(i['sentence'])
        vec = bc.encode(sentence)
       
        for v_i, sen_i in zip(vec,data):
            sen_len = len(sen_i['sentence'])
            v_i = np.delete(v_i, 0, axis=0)
            v_i = np.delete(v_i, sen_len+1, axis=0)
            idx = 0
            for pph_i in sen_i['pph_data']:
                l = len(pph_i['pph'])
                pph_i['pph_vec'] = v_i[idx:idx+l]
                idx += l

def prepare_data(data):
    def gen_from_data(data, length, dim):
        cnt = 0
        for i in data:
            for j in i['pph_data']:
                cnt += 1
        # pph character vector
        feature = np.zeros((cnt, length, dim))
        cnt = 0
        for i in data:
            for j in i['pph_data']:
                l = j['pph_vec'].shape[0]
                feature[cnt][:l] = j['pph_vec']
                cnt += 1
        
        # utt isr
        feature_utt = np.zeros(cnt)
        cnt = 0
        for i in data:
            for j in i['pph_data']:
                feature_utt[cnt] = i['isr_utt']
                cnt +=1

        
        # pph local isr
        label = np.zeros(cnt)
        cnt = 0 
        for i in data:
            for j in i['pph_data']:
                label[cnt] = j['isr']
                cnt += 1 
        return {'feature':feature, 'feature_utt': feature_utt, 'label':label}
    def get_seq_shape(data):
        cnt = 0
        length, dim = data[1]['pph_data'][0]['pph_vec'].shape
        for i in data:
            for j in i['pph_data']:
                cnt += 1
                length = max(length, j['pph_vec'].shape[0])
        return length, dim
    l1, d1 = get_seq_shape(data['trn'])
    l2, d2 = get_seq_shape(data['valid'])
    l3, d3 = get_seq_shape(data['test'])
    l = max(l1, l2, l3)
    dim = d1
    assert(d1==d2)
    assert(d2==d3)
    dataset = {
        'trn':gen_from_data(data['trn'], l, dim), 
        'valid':gen_from_data(data['valid'], l, dim), 
        'test':gen_from_data(data['test'], l, dim)
    }
    return dataset

def next_batch(dataset, batch):
    cnt = dataset['feature'].shape[0]
    shuffle = np.random.permutation(dataset['feature'].shape[0])
    start = 0
    x = dataset['feature'][shuffle]
    x1 = dataset['feature_utt'][shuffle]
    y = dataset['label'][shuffle]
    while(start+batch <= cnt):
        yield (
            x[start:start+batch],
            x1[start:start+batch],
            y[start:start+batch]
        )
        start += batch
    yield( x[start:], x1[start:], y[start:])

def next_batch_nos(dataset, batch):
    cnt = dataset['feature'].shape[0]
    start = 0
    x = dataset['feature']
    x1 = dataset['feature_utt']
    y = dataset['label']
    while(start+batch <= cnt):
        yield (
            x[start:start+batch],
            x1[start:start+batch],
            y[start:start+batch]
        )
        start += batch
    yield( x[start:], x1[start:], y[start:])

def gen_fd(x, x1, y, g):
    fd = {}
    fd[g.inputs] = x
    fd[g.inputs_utt] = x1
    fd[g.outputs] = y
    return fd

def calc_tre(data, preds, ans):
    idx = 0
    tre = np.zeros(len(data))
    for cnt_s, i in enumerate(data):
        preds_err_var = np.zeros(len(data))
        preds_var = np.zeros(len(data))
        for cnt, j in enumerate(i['pph_data']):
            preds_err_var[cnt] = preds[idx]-ans[idx]
            preds_var[cnt] = ans[idx]
            idx += 1
        err_var = np.var(preds_err_var)
        var = np.var(preds_var)
        tre[cnt_s] = err_var/var
    tre_total = np.mean(tre)
    return tre_total
def run_tre(sess, g, data, dataset):
    batch = 200
    preds = None
    for x,x1,y in next_batch_nos(dataset, batch):
        fd = gen_fd(x,x1,y,g)
        _, test_loss, test_preds = sess.run(
                            [g.train_op, 
                             g.loss, 
                             g.preds],
                            feed_dict = fd
            )
        test_preds = np.reshape(test_preds, (test_preds.shape[0]))
        if preds is None:
            preds = test_preds
        else:
            preds = np.concatenate((preds, test_preds))
  
    tre = calc_tre(data, preds, dataset['label'])
    return tre


def main():
    # Load Data from raw or pkl
    data_pre_path = './data/with_utt/data_utt.pkl'
    if os.path.exists(data_pre_path):
        with open(data_pre_path, 'rb') as f:
            data = pickle.load(f)
        print('### Data loaded from {} ###'.format(data_pre_path))
    else:
        data = load_data()
        with open(data_pre_path, 'wb') as f:
            pickle.dump(data, f)
        print('### Data loaded ###')
    dataset = prepare_data(data)
    max_l = dataset['trn']['feature'].shape[1]

    # Graph
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
    config = tf.ConfigProto(gpu_options = gpu_options)
    g_trn = Graph(max_l)
    g_valid = Graph(max_l, is_train=False)
    g_test = Graph(max_l, is_train=False)

    sess_trn = tf.Session(config = config, graph=g_trn.graph)
    sess_valid = tf.Session(config = config, graph=g_valid.graph)
    sess_test = tf.Session(config = config, graph=g_test.graph)

    writer_trn = tf.summary.FileWriter('./log/trn', g_trn.graph)
    writer_valid = tf.summary.FileWriter('./log/valid', g_valid.graph)
    writer_test = tf.summary.FileWriter('./log/test', g_test.graph)

    with g_trn.graph.as_default():
        init_trn = tf.global_variables_initializer()
        save_trn = tf.train.Saver()
    
    with g_valid.graph.as_default():
        init_valid = tf.global_variables_initializer()
        save_valid = tf.train.Saver()

    with g_test.graph.as_default():
        init_test = tf.global_variables_initializer()
        save_test = tf.train.Saver()

    # Start 
    epochs = 20
    batch = 200
    sess_trn.run(init_trn)
    for i in range(epochs):
        for x, x1, y in next_batch(dataset['trn'], batch):
            fd = gen_fd(x, x1, y,g_trn)
            _, trn_loss, summary_trn = sess_trn.run(
                                [g_trn.train_op, g_trn.loss, 
                                 g_trn.summary_op],
                                feed_dict = fd
                )
        ckpt = './log/model_epoch_{:02d}.ckpt'.format(i)
        save_trn.save(sess_trn, ckpt)
        
        sess_valid.run(init_valid)
        save_valid.restore(sess_valid, ckpt)
        for x,x1,y in next_batch(dataset['valid'], batch):
            fd = gen_fd(x,x1,y,g_valid)
            _, valid_loss, summary_valid = sess_valid.run(
                                [g_valid.train_op, 
                                 g_valid.loss, 
                                 g_valid.summary_op],
                                feed_dict = fd
                )
    
        sess_test.run(init_test)
        save_test.restore(sess_test, ckpt)
        for x,x1,y in next_batch(dataset['test'], batch):
            fd = gen_fd(x,x1,y,g_test)
            _, test_loss, summary_test = sess_test.run(
                                [g_test.train_op, 
                                 g_test.loss, 
                                 g_test.summary_op],
                                feed_dict = fd
                )

        print('Epoch {:3}'.format(i),
              '\tTrain loss: {:>6.5f}'.format(trn_loss),
              '\tValid loss: {:>6.5f}'.format(valid_loss),
              '\tTest loss: {:>6.5f}'.format(test_loss))
        writer_trn.add_summary(summary_trn,i)
        writer_valid.add_summary(summary_valid,i)
        writer_test.add_summary(summary_test,i)

        # run tre
        sess_valid.run(init_valid)
        save_valid.restore(sess_valid, ckpt)
        tre_valid = run_tre(sess_valid, g_valid, data['valid'], dataset['valid'])
        sess_test.run(init_test)
        save_test.restore(sess_test, ckpt)
        tre_test = run_tre(sess_test, g_test, data['test'], dataset['test'])
        print('\t\tValid TRE: {:>6.5f}'.format(tre_valid), 
              '\t\tTest TRE: {:>6.5f}'.format(tre_test))

    sess_trn.close()
    sess_valid.close()
    sess_test.close()

if __name__ == '__main__':
    main()
