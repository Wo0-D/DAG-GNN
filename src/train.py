''''
Main function for traininng DAG-GNN

'''

from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

# import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import math

# import numpy as np
from utils import *
from modules import *

parser = argparse.ArgumentParser()

# -----------data parameters ------         数据参数
# configurations
parser.add_argument('--data_type', type=str, default='synthetic',
                    choices=['synthetic', 'discrete', 'real'],
                    help='choosing which experiment to do.')                # 选择做什么实验：合成、离散、真实
parser.add_argument('--data_filename', type=str, default='alarm',
                    help='data file name containing the discrete files.')   # 包含离散文件的数据文件名
parser.add_argument('--data_dir', type=str, default='data/',
                    help='data file name containing the discrete files.')   # 包含离散文件的数据文件路径
parser.add_argument('--data_sample_size', type=int, default=5000,
                    help='the number of samples of data')                   # 数据的样本数量
parser.add_argument('--data_variable_size', type=int, default=10,
                    help='the number of variables in synthetic generated data') # 生成合成数据中的变量数量
parser.add_argument('--graph_type', type=str, default='erdos-renyi',
                    help='the type of DAG graph by generation method')      # 生成DAG图的方法类型
parser.add_argument('--graph_degree', type=int, default=2,
                    help='the number of degree in generated DAG graph')     # 生成的DAG图的度数
parser.add_argument('--graph_sem_type', type=str, default='linear-gauss',
                    help='the structure equation model (SEM) parameter type')   # SEM的参数类型
parser.add_argument('--graph_linear_type', type=str, default='nonlinear_2',     # 合成数据类型：
                    help='the synthetic data type: linear -> linear SEM, nonlinear_1 -> x=Acos(x+1)+z, nonlinear_2 -> x=2sin(A(x+0.5))+A(x+0.5)+z')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')              # 要推断的边缘类型数量
parser.add_argument('--x_dims', type=int, default=1,  # changed here
                    help='The number of input dimensions: default 1.')      # 输入的维数
parser.add_argument('--z_dims', type=int, default=1,                        # 潜在变量维数：默认值与变量大小相同
                    help='The number of latent variable dimensions: default the same as variable size.')

# -----------training hyperparameters           训练超参数
parser.add_argument('--optimizer', type=str, default='Adam',
                    help='the choice of optimizer used')                    # 选用的优化器
parser.add_argument('--graph_threshold', type=float, default=0.3,  # 0.3 is good, 0.2 is error prune
                    help='threshold for learned adjacency matrix binarization')     # 学习邻接矩阵二值化的阈值
parser.add_argument('--tau_A', type=float, default=0.0,
                    help='coefficient for L-1 norm of A.')                  # A 的 L-1 准则系数
parser.add_argument('--lambda_A', type=float, default=0.,
                    help='coefficient for DAG constraint h(A).')            # DAG 约束条件 h(A) 的系数
parser.add_argument('--c_A', type=float, default=1,
                    help='coefficient for absolute value h(A).')            # h(A) 绝对值 的系数
parser.add_argument('--use_A_connect_loss', type=int, default=0,
                    help='flag to use A connect loss')                      # 选择 A 连接损失的标志
parser.add_argument('--use_A_positiver_loss', type=int, default=0,
                    help='flag to enforce A must have positive values')     # 强制 A 未正值的标志

parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')                         # 禁用 CUDA 训练
parser.add_argument('--seed', type=int, default=42, help='Random seed.')    # 随即种子
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')                      # 训练的 epochs 数
parser.add_argument('--batch-size', type=int, default=100,
                    # note: should be divisible by sample size, otherwise throw an error
                    help='Number of samples per batch.')                    # 每次 batch 的样本数
parser.add_argument('--lr', type=float, default=3e-3,  # basline rate = 1e-3
                    help='Initial learning rate.')                          # 初始 lr
parser.add_argument('--encoder-hidden', type=int, default=64,
                    help='Number of hidden units.')                         # encoder 中的隐藏单元的数量
parser.add_argument('--decoder-hidden', type=int, default=64,
                    help='Number of hidden units.')                         # decoder 中的隐藏单元的数量
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')                 # Gumbel softmax 的温度
parser.add_argument('--k_max_iter', type=int, default=1e2,
                    help='the max iteration number for searching lambda and c')     # 搜索 lambda 和 c 的最大迭代次数

parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp, or sem).')       # 路径 encoder 模型的类型（mlp，或 SEM）
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, or sim).')            # decoder 模型的类型（mlp，或 SEM）
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')                    # 禁用因子图模型
parser.add_argument('--suffix', type=str, default='_springs5',
                    help='Suffix for training data (e.g. "_charged".')      # 训练数据的后缀（例如"_charged）
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')            # encoder 的 dropout 率
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')            # decoder 的 dropout 率
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')      # 训练好模型的保存位置，留空不保存
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')               # 如果要 finetunning，从哪里加载训练好的模型，留空从头开始培训

parser.add_argument('--h_tol', type=float, default=1e-8,
                    help='the tolerance of error of h(A) to zero')          # h(A) 的误差对零的容许度
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')   # 重复使用教师资源前需要预测的若干步骤
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')      # 经过多少个 epochs 之后，LR 的衰变系数为伽马
parser.add_argument('--gamma', type=float, default=1.0,
                    help='LR decay factor.')                                # LR 衰变系数
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')    # 在 decoder 中跳过最初的边类型
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')                                # 输出参数
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')     # 在训练前向传播时使用离散样本
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')                  # 是否使用稀疏先验
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')    # 是否使用动态重计算图来测试

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dynamic_graph:
    print("Testing with dynamically re-computed graph.")

# Save model and meta-data. Always saves in a new sub-folder.               # 保存模型和元数据，总是在一个新的文件夹保存
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    # safe_name = save_folder.text.replace('/', '_')
    os.makedirs(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

# ================================================
# get data: experiments = {synthetic SEM, ALARM}
# ================================================
train_loader, valid_loader, test_loader, ground_truth_G = load_data(args, args.batch_size, args.suffix)

# ===================================
# load modules
# ===================================
# Generate off-diagonal interaction graph
off_diag = np.ones([args.data_variable_size, args.data_variable_size]) - np.eye(args.data_variable_size)

rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
rel_rec = torch.DoubleTensor(rel_rec)
rel_send = torch.DoubleTensor(rel_send)

# add adjacency matrix A
num_nodes = args.data_variable_size
adj_A = np.zeros((num_nodes, num_nodes))

if args.encoder == 'mlp':
    encoder = MLPEncoder(args.data_variable_size * args.x_dims, args.x_dims, args.encoder_hidden,
                         int(args.z_dims), adj_A,
                         batch_size=args.batch_size,
                         do_prob=args.encoder_dropout, factor=args.factor).double()
elif args.encoder == 'sem':
    encoder = SEMEncoder(args.data_variable_size * args.x_dims, args.encoder_hidden,
                         int(args.z_dims), adj_A,
                         batch_size=args.batch_size,
                         do_prob=args.encoder_dropout, factor=args.factor).double()

if args.decoder == 'mlp':
    decoder = MLPDecoder(args.data_variable_size * args.x_dims,
                         args.z_dims, args.x_dims, encoder,
                         data_variable_size=args.data_variable_size,
                         batch_size=args.batch_size,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout).double()
elif args.decoder == 'sem':
    decoder = SEMDecoder(args.data_variable_size * args.x_dims,
                         args.z_dims, 2, encoder,
                         data_variable_size=args.data_variable_size,
                         batch_size=args.batch_size,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout).double()

if args.load_folder:
    encoder_file = os.path.join(args.load_folder, 'encoder.pt')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(args.load_folder, 'decoder.pt')
    decoder.load_state_dict(torch.load(decoder_file))

    args.save_folder = False

# ===================================
# set up training parameters
# ===================================

if args.optimizer == 'Adam':
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
elif args.optimizer == 'LBFGS':
    optimizer = optim.LBFGS(list(encoder.parameters()) + list(decoder.parameters()),
                            lr=args.lr)
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()),
                          lr=args.lr)

scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

# Linear indices of an upper triangular mx, used for acc calculation        # 上三角 mx 的线性指数，用于 acc 计算
triu_indices = get_triu_offdiag_indices(args.data_variable_size)
tril_indices = get_tril_offdiag_indices(args.data_variable_size)

if args.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.DoubleTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)

    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    encoder.cuda()
    decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)


# compute constraint h(A) value
def _h_A(A, m):
    expm_A = matrix_poly(A * A, m)
    h_A = torch.trace(expm_A) - m
    return h_A


prox_plus = torch.nn.Threshold(0., 0.)


def stau(w, tau):
    w1 = prox_plus(torch.abs(w) - tau)
    return torch.sign(w) * w1


def update_optimizer(optimizer, original_lr, c_A):
    """ related LR to c_A, whenever c_A gets big, reduce LR proportionally """
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr


# ===================================
# training:
# ===================================

def train(epoch, best_val_loss, ground_truth_G, lambda_A, c_A, optimizer):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    shd_trian = []

    encoder.train()
    decoder.train()
    scheduler.step()

    # update optimizer
    optimizer, lr = update_optimizer(optimizer, args.lr, c_A)

    for batch_idx, (data, relations) in enumerate(train_loader):

        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data).double(), Variable(relations).double()

        # reshape data
        relations = relations.unsqueeze(2)

        optimizer.zero_grad()

        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data, rel_rec,
                                                                                          rel_send)  # logits is of size: [num_sims, z_dims]
        edges = logits

        dec_x, output, adj_A_tilt_decoder = decoder(data, edges, args.data_variable_size * args.x_dims, rel_rec,
                                                    rel_send, origin_A, adj_A_tilt_encoder, Wa)

        if torch.sum(output != output):
            print('nan error\n')

        target = data
        preds = output
        variance = 0.

        # reconstruction accuracy loss                              # 重构准确度 损失
        loss_nll = nll_gaussian(preds, target, variance)

        # KL loss                                                   # KL 散度损失
        loss_kl = kl_gaussian_sem(logits)

        # ELBO loss:                                                # ELBO 置信下界 损失
        loss = loss_kl + loss_nll

        # add A loss                                                # A 损失
        one_adj_A = origin_A  # torch.mean(adj_A_tilt_decoder, dim =0)
        sparse_loss = args.tau_A * torch.sum(torch.abs(one_adj_A))

        # other loss term                                           # 其他损失项
        if args.use_A_connect_loss:
            connect_gap = A_connect_loss(one_adj_A, args.graph_threshold, z_gap)
            loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

        if args.use_A_positiver_loss:
            positive_gap = A_positive_loss(one_adj_A, z_positive)
            loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

        # compute h(A)                                              # 无环约束 h(A)
        h_A = _h_A(origin_A, args.data_variable_size)
        loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(
            origin_A * origin_A) + sparse_loss  # +  0.01 * torch.sum(variance * variance)

        loss.backward()
        loss = optimizer.step()

        myA.data = stau(myA.data, args.tau_A * lr)

        if torch.sum(origin_A != origin_A):
            print('nan error\n')

        # compute metrics
        graph = origin_A.data.clone().numpy()
        graph[np.abs(graph) < args.graph_threshold] = 0

        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))

        mse_train.append(F.mse_loss(preds, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        shd_trian.append(shd)

    print(h_A.item())
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []

    print('Epoch: {:04d}'.format(epoch),
          'nll_train: {:.10f}'.format(np.mean(nll_train)),
          'kl_train: {:.10f}'.format(np.mean(kl_train)),
          'ELBO_loss: {:.10f}'.format(np.mean(kl_train) + np.mean(nll_train)),
          'mse_train: {:.10f}'.format(np.mean(mse_train)),
          'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(nll_val) < best_val_loss:
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'ELBO_loss: {:.10f}'.format(np.mean(kl_train) + np.mean(nll_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()

    if 'graph' not in vars():
        print('error on assign')

    return np.mean(np.mean(kl_train) + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A


# ===================================
# main
# ===================================

t_total = time.time()
best_ELBO_loss = np.inf
best_NLL_loss = np.inf
best_MSE_loss = np.inf
best_epoch = 0
best_ELBO_graph = []
best_NLL_graph = []
best_MSE_graph = []
# optimizer step on hyparameters
c_A = args.c_A
lambda_A = args.lambda_A
h_A_new = torch.tensor(1.)
h_tol = args.h_tol
k_max_iter = int(args.k_max_iter)
h_A_old = np.inf

try:
    for step_k in range(k_max_iter):
        while c_A < 1e+20:
            for epoch in range(args.epochs):
                ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = train(epoch, best_ELBO_loss, ground_truth_G, lambda_A,
                                                                       c_A, optimizer)
                if ELBO_loss < best_ELBO_loss:          # 更新全局最优解
                    best_ELBO_loss = ELBO_loss
                    best_epoch = epoch
                    best_ELBO_graph = graph

                if NLL_loss < best_NLL_loss:
                    best_NLL_loss = NLL_loss
                    best_epoch = epoch
                    best_NLL_graph = graph

                if MSE_loss < best_MSE_loss:
                    best_MSE_loss = MSE_loss
                    best_epoch = epoch
                    best_MSE_graph = graph

            print("Optimization Finished!")
            print("Best Epoch: {:04d}".format(best_epoch))
            if ELBO_loss > 2 * best_ELBO_loss:          # 过拟合？
                break

            # update parameters
            A_new = origin_A.data.clone()
            h_A_new = _h_A(A_new, args.data_variable_size)
            if h_A_new.item() > 0.25 * h_A_old:
                c_A *= 10
            else:
                break

            # update parameters
            # h_A, adj_A are computed in loss anyway, so no need to store  计算 loss 时已经计算过，因此不用存储
        h_A_old = h_A_new.item()
        lambda_A += c_A * h_A_new.item()

        if h_A_new.item() <= h_tol:
            break

    if args.save_folder:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        log.flush()

    # test()
    print(best_ELBO_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
    print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    print(best_NLL_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_NLL_graph))
    print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    print(best_MSE_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
    print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph = origin_A.data.clone().numpy()
    graph[np.abs(graph) < 0.1] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.1, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph[np.abs(graph) < 0.2] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.2, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph[np.abs(graph) < 0.3] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.3, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)


except KeyboardInterrupt:       # 键盘强制结束运行 CTRL+C
    # print the best anyway
    print(best_ELBO_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
    print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    print(best_NLL_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_NLL_graph))
    print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    print(best_MSE_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
    print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph = origin_A.data.clone().numpy()
    graph[np.abs(graph) < 0.1] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.1, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph[np.abs(graph) < 0.2] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.2, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph[np.abs(graph) < 0.3] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.3, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

f = open('trueG', 'w')
matG = np.matrix(nx.to_numpy_array(ground_truth_G))
for line in matG:
    np.savetxt(f, line, fmt='%.5f')
f.closed

f1 = open('predG', 'w')
matG1 = np.matrix(origin_A.data.clone().numpy())
for line in matG1:
    np.savetxt(f1, line, fmt='%.5f')
f1.closed

if log is not None:
    print(save_folder)
    log.close()
