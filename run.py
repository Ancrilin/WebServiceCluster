import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import time
import argparse
from tqdm import tqdm
from logger import Logger
import os
from model.Doc2vec import Doc2vec
from util.util import Preprocessor
from util.visualization import draw_curve

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(1)
torch.manual_seed(1)                                                 # 设置随机种子用来保证模型初始化的参数是一致
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True

def main(args):
    logger.info('device {}'.format(device))
    config = vars(args)  # 返回参数字典
    logger.info(config)
    def train_doc2vec(name, text):
        logger.info('train doc2vec...')
        doc2vec = Doc2vec(vector_size=args.doc_vec_size)
        doc2vec = doc2vec(name, text, args.doc_n_epoch)
        doc_vec = []
        for n in tqdm(name):
            doc_vec.append(doc2vec.docvecs[n])
        doc_vec = np.array(doc_vec)
        logger.info('save doc2vec vector in {}'.format(os.path.join(args.output_dir, args.doc_vec_savepath)))
        np.save(os.path.join(args.output_dir, args.doc_vec_savepath), doc2vec)
        logger.info('save doc2vec model in {}'.format(os.path.join(args.output_dir, 'dov2vec.model')))
        doc2vec.save(os.path.join(args.output_dir, 'save/dov2vec.model'))
        logger.info('doc_vec shape {}'.format(np.shape(doc_vec)))
        return doc_vec

    def train_sine(config, model, dataset):
        start_time = time.time()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  # 优化器 Adam优化算法

        total_batch = 0  # 记录进行到多少batch
        data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
        last_improve = 0
        n_sample = len(data_loader)
        curve_loss = []
        t_epoch = 0
        for epoch in range(config.epochs):
            total_loss = 0
            for batch in tqdm(data_loader):
                xi, xj, xk = batch
                # print('xi', xj, 'xk', xi, xj, xk)
                model.zero_grad()
                loss = model(xi, xj, xk)
                regularizer_loss = config.alpha * model.regularization_weight()  # alpha为控制正则化参数 mymodel
                loss += regularizer_loss
                loss.backward()
                optimizer.step()
                total_loss += loss
                if total_batch % 10000 == 0:
                    logger.info('total_batch {} batch_loss {}'.format(total_batch, loss / config.batch_size))
                total_batch += 1
            logger.info('Epoch [{}/{}] loss {}'.format(epoch + 1, config.epochs, total_loss / n_sample))
            curve_loss.append(total_loss / n_sample)
            t_epoch += 1
        draw_curve(curve_loss, t_epoch, 'SINE_loss', args.output_dir)
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'save/sine.pkl'))

    def train_kmeans(dataset):
        from model.Kmeanspp import kmeans
        model = kmeans(n_cluster=args.n_cluster, max_iter=args.max_iter, tol=args.tol)
        model.fit(dataset)
        cluster_label = model.label
        return cluster_label

    if args.train:
        preprocessor = Preprocessor(config['min_time'])
        logger.info('load dataset...')
        all_name, all_process_cuts, all_labels, all_tags, word_map, tag_map, label_map = preprocessor.read_data(
            os.path.join(args.dataset, args.doc_datafile))
        if args.reset or not os.path.exists(os.path.join(args.output_dir, args.doc_vec_savepath)):
            doc_vec = train_doc2vec(all_name, all_process_cuts)
        else:
            doc_vec = np.load(os.path.join(args.output_dir, args.doc_vec_savepath))

        if args.reset or not os.path.exists(os.path.join(args.output_dir, args.graph_vec_savepath)):
            from model.SINE import Config, SINE
            from util.processor_dataset import get_dataset
            from util.dataset import myDataset
            triplets, vocab = get_dataset(os.path.join(args.dataset, args.graph_datafile))
            config = Config()
            config.device = device
            config.epochs = args.graph_n_epoch
            config.batch_size = args.batch_size
            config.n_vocab = len(vocab)
            config.embed = args.graph_vec_size
            config.dim = args.dim
            dataset = myDataset(triplets, config)
            sine = SINE(config).to(device)
            logger.info(sine)
            train_sine(config, sine, dataset)
            graph_vec = []
            for n in all_name:
                graph_vec.append(sine.get_embedding(n))
            graph_vec = np.array(graph_vec)
            np.save(os.path.join(args.output_dir, args.graph_vec_savepath), graph_vec)
        else:
            graph_vec = np.load(os.path.join(args.output_dir, args.graph_vec_savepath))
        logger.info('graph vector shape {}'.format(np.shape(graph_vec)))
        vector = np.concatenate((doc_vec, graph_vec), axis=1) if args.graph \
            else doc_vec
        real_label = []
        for l in all_labels:
            real_label.append(label_map[l])
        cluster_label = train_kmeans(vector)
        from util.metric import get_score
        recall, precison = get_score(cluster_label, real_label)
        logger.info('recall {} precision {}'.format(recall, precison))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--delta',
                        default=0.5,
                        type=float)
    parser.add_argument('--alpha',
                        default=0.1,
                        type=float)
    parser.add_argument('--tol',
                        default=1e-4,
                        type=float)
    parser.add_argument('--dim',
                        default=512,
                        type=int)
    parser.add_argument('--n_cluster',
                        default=20,
                        type=int)
    parser.add_argument('--max_iter',
                        default=300,
                        type=int)
    parser.add_argument('--batch_size',
                        default=32,
                        type=int)
    parser.add_argument('--graph_lr',
                        default=1e-3,
                        type=float)
    parser.add_argument('--doc_n_epoch',
                        default=4,
                        type=int)
    parser.add_argument('--graph_n_epoch',
                        default=8,
                        type=int)
    parser.add_argument('--dataset',
                        required=True,
                        choices={'data'})
    parser.add_argument('--doc_vec_size',
                        default=500,
                        type=int)
    parser.add_argument('--graph_vec_size',
                        default=200,
                        type=int)
    parser.add_argument('--min_time',
                        default=1,
                        type=int)
    parser.add_argument('--doc_datafile',
                        type=str,
                        default='20ClassesRawData_API_cleanTag.csv')
    parser.add_argument('--graph_datafile',
                        type=str,
                        default='WebNet_df.csv')
    parser.add_argument('--doc_vec_savepath',
                        default='doc_vec.npy',
                        type=str)
    parser.add_argument('--graph_vec_savepath',
                        default='graph_vec.npy',
                        type=str)
    parser.add_argument('--output_dir',
                        default='output',
                        type=str)
    parser.add_argument('--reset',
                        action='store_true')
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)
    logger = Logger(os.path.join(args.output_dir, 'train.log'))
    main(args)