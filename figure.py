import matplotlib
from matplotlib import rcParams
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import seaborn as sns
import math
import xlrd
import h5py
from sklearn.manifold import TSNE
import os

#####################  set parameters  ####################

N             = 39                     # number of node
omega_s       = 100 * math.pi          # synchronous angular frequency
baseMVA       = 10**8                  # power reference value
if N == 14:
    M         = 6800                   # mass moments of inertia, 6800 for 14, 12000 for 118
elif N == 39:
    M         = 50000
elif N == 118:
    M         = 12000
alpha         = 0.1                    # damping
theta         = math.pi                # range of theta_0
omega         = 20                     # range of omega_0
step = 0.05                      # time step to solve ODE
max_t = 120                      # maximum time to sove ODE
t = np.arange(0, max_t, step)    # time stream to solve ODE
data_number = 1000               # samping number
exp_num = 125

early_stop = False
interval = False

relative = False

normalize = False

standard = False
mode = 1

move = False
WSZ = 11
oversample = False
n_critical = 5000 # thresholds to change loss function
weight = False
n = 0 # current epochs

if interval == True:
    if N == 14:
        timelength = 100
    elif N == 39:
        timelength = 50
    elif N == 118:
        timelength = 100

else:
    if N == 14:
        timelength = 400
    elif N == 39:
        timelength = 100          # 原始数据的时间长度
    elif N == 118:
        timelength = 100
net = 'RGCN-TCN'
data_set = 'one'
adj_mode      = 2                       # 邻接矩阵模式：1、adj=Y
                                        #              2、adj=diag(P)+Y
                                        #              3、adj=P'+Y',P'=P·(1+ω_0/ω_s),Y'=Y_ij·sin(θ_i-θ_j)
chosedlength  =  99                     # length used to train
TEST_SIZE     =  0.2                    # train:val_test = 6:2:2
CHANNEL       =  1                      # only use omega data to train

if net == 'GCN' or 'RGCN' or 'RGCN-TCN' or 'RGCN-TCN_2':
    learning_rate = 1e-3                    # Learning rate for Adam
elif net == 'GAT' or 'RGAT':
    learning_rate = 5e-3
BATCH_SIZE    = 256                     # Batch size
epochs        = 5000                    # Number of training epochs
patience      = 200                       # Patience for early stopping

def dmove(t, y, sets):
    """
    定义ODE
    """
    X = np.zeros((N * 2))
    # 0-13 is the dtheta, 14-27 is the domega
    for i in range(N):
        X[i] = y[i + N]
        a = 0
        for j in range(N):
            a += sets[i + 1, j] * math.sin(y[j] - y[i])
        X[i + N] = -alpha * y[i + N] + sets[0, i] + a
    return X

def load_para(N, M, baseMVA, omega_s, net, adj_mode):
    """
    从.xlsx文件中导出参数及初始条件
    """
    # parameter = xlrd.open_workbook('/home/duguyuan/Documents/Swing_in_Grid/IEEE/case%s/parameter/parameter.xlsx' %(N))
    # parameter = xlrd.open_workbook('/public/home/spy2018/swing/parameter/parameter%s.xlsx' %(N))
    parameter = xlrd.open_workbook('./data/parameter.xlsx')
    # 功率矩阵
    P_sheet1 = parameter.sheet_by_index(0)
    nrows = P_sheet1.nrows
    ncols = P_sheet1.ncols
    P = np.zeros((N))
    for i in range(nrows):
        for j in range(ncols):
            P[i] = P_sheet1.cell_value(i, j)
    P = P * baseMVA
    P = [i - np.sum(P)/N for i in P]  # 功率补偿
    P = np.array([i/(M*omega_s) for i in P])
    # 导纳矩阵
    Y_sheet1 = parameter.sheet_by_index(1)
    nrows = Y_sheet1.nrows
    ncols = Y_sheet1.ncols
    Y = np.zeros((N, N))
    YY = np.zeros((N, N))
    for i in range(nrows):
        for j in range(ncols):
            Y[i, j] = Y_sheet1.cell_value(i, j)
            if Y[i, j] != 0:
                YY[i, j] = 1
    Y = np.array([i*baseMVA/(M*omega_s) for i in Y])
    # 参数合并
    PY = np.vstack((P, Y))
    PY = PY / 16

    Y /= 16
    P /= 16
    if net == 'GCN' or 'RGCN' or 'RGCN-TCN':
        if adj_mode == 1:
            Y = YY + np.eye(N)
        elif adj_mode == 2:
            Y = Y + np.diag(abs(np.squeeze(P)))
            Y = Y / np.amax(Y)
        else:
            Y = Y / np.amax(Y)
        # A = sparse.csr_matrix(Y)
    elif net == 'GAT':
        A = sparse.csr_matrix(YY)
    print('原始数据导入完毕')
    return Y, PY

def power_topology():
    N = 14
    parameter = xlrd.open_workbook('F:/Swing/parameter/case%s/parameter2.xlsx' %(N))
    Y_sheet1 = parameter.sheet_by_index(1)
    nrows = Y_sheet1.nrows
    ncols = Y_sheet1.ncols
    Y = np.zeros((N, N))
    for i in range(nrows):
        for j in range(ncols):
            Y[i, j] = Y_sheet1.cell_value(i, j)

    G = nx.Graph()  # 建立一个空的无向图G
    for i in range(N):
        G.add_node(i+1)
    for i in range(N):
        for j in range(N):
            if Y[i, j] != 0:
                G.add_edge(i+1, j+1)  # 添加节点

    print("nodes:", G.nodes())      #输出全部的节点： [1, 2, 3]
    print("edges:", G.edges())      #输出全部的边：[(2, 3)]
    print("number of edges:", G.number_of_edges())   #输出边的数量：1
    # plt.figure(figsize=(4/2.54, 4/2.54))
    plt.figure(figsize=(3/2.54, 3/2.54), dpi=300)
    pos = nx.shell_layout(G)
    colors = ['k']
    for i in range(20-1):
        colors.append('k')
    options = {
        "node_color": 'deepskyblue',
        "edge_color": colors,
        # "edge_cmap": plt.cm.Blues,
        "width": 0.5,
        "node_size": 5,
        "with_labels": False
    }
    nx.draw(
        G,
        pos,
        **options
    )
    plt.savefig('F:/Swing/14.eps', format='eps')
    plt.savefig('F:/Swing/14.png')
    plt.show()

def bar():
    size = 4
    x = np.arange(size)
    a = np.random.random(size)
    b = np.random.random(size)
    c = np.random.random(size)

    # ## N-1
    # a = np.array([0.9954359, 1-144/35349, 1-34/3651, 0.9995490209109387])
    # b = np.array([0.9927179, 1-108/35349, 1-176/3651, 0.9977010201341789])
    # c = np.array([0.99379486, 1-229/35349, 1-13/3651, 0.9993773090130521])

    # ## N-2
    # a = np.array([0.93106616, 1-10204/(10204+291248), 1-15336/(15336+53712), 0.9604149479163263])
    # b = np.array([0.9156653, 1-7038/(7038+294414), 1-24208/(24208+44840), 0.9299638352793982])
    # c = np.array([0.9351174, 1-9975/(9975+291477), 1-14064/(14064+54984), 0.9601777455288496])

    ## N-3
    a = np.array([0.984983, 1-370/(370+43018), 1-516/(516+15096), 0.9972146775411879])
    b = np.array([0.95742375, 1-961/(961+42427), 1-1551/(1551+14061), 0.9872436232576554])
    c = np.array([0.9860169, 1-426/(426+42962), 1-399/(399+15213), 0.9969618738647474])

    total_width, n = 0.6, 3
    width = total_width / n
    x = x - (total_width - width) / 2

    labels = ['ACC', 'TNR', 'TPR', 'AUC']
    config = {
        "font.family":'Times New Roman',
        "font.size": 10/(4/3),
        "mathtext.fontset":'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    fig = plt.figure(figsize=(8/2.54, 6/2.54), dpi=600)
    ax1 = fig.subplots()
    left = 0.15
    right = 0.2
    bottom = 0.15
    top = 0.15
    h = 1 - bottom - top
    w = 1 - left - right
    fig.subplots_adjust(left=left,
                        bottom=bottom,
                        right=left + w,
                        top=bottom + h)
    plt.rc('axes', lw=0.5)
    plt.bar(x,             a, width=width, label=r'$\rm B^{I}$',
        # color='r',
        ec='k', ls='-', lw=1
    )
    plt.bar(x + width,     b, width=width, label=r'$\rm B^{II}$',
        # color='dimgrey',
        ec='k', ls='-', lw=1, tick_label=labels
    )
    plt.bar(x + 2 * width, c, width=width, label=r'$\rm B^{III}$',
        # color='darkgrey',
        ec='k', ls='-', lw=1
    )
    plt.ylim((0.9, 1))
    ax1.tick_params(direction='in')
    legend = plt.legend(loc='upper right', bbox_to_anchor=(1.29, 1.033), fancybox=False, edgecolor='k', facecolor='none')
    legend.get_frame().set_linewidth(0.5)
    # plt.subplots_adjust(right=0.8)
    # plt.savefig('F:/Swing/comparsion_coupling_bar.eps', format='eps', bbox_inches='tight')
    plt.savefig('F:/Swing/comparison_coupling_bar.eps', format='eps')
    plt.savefig('F:/Swing/comparison_coupling_bar.png')
    plt.show()

def draw_matrix(con_mat):
    # trans_mat = np.array([[62, 16, 32 ,9, 36],
    #                     [16, 16, 13, 8, 7],
    #                     [28, 16, 61, 8, 18],
    #                     [16, 2, 10, 40, 48],
    #                     [52, 11, 49, 8, 39]], dtype=int)
    trans_mat = con_mat
    trans_prob_mat = (trans_mat.T/np.sum(trans_mat, 1)).T

    if True:
        label = ["Patt {}".format(i) for i in range(1, trans_mat.shape[0]+1)]
        df = pd.DataFrame(trans_prob_mat, index=label, columns=label)

        # Plot
        plt.figure(figsize=(7.5, 6.3))
        ax = sns.heatmap(df, xticklabels=df.corr().columns, 
                        yticklabels=df.corr().columns, cmap='magma',
                        linewidths=6, annot=True)
        
        # Decorations
        plt.xticks(fontsize=16,family='Times New Roman')
        plt.yticks(fontsize=16,family='Times New Roman')
        
        plt.tight_layout()
        plt.savefig('./result/result/3/matrix.png')
        plt.show()

def draw_matrix_1():
    A_1, PY = load_para(
        N=N, M=M, baseMVA=baseMVA, omega_s=omega_s, net=net, adj_mode=1
    )
    Y, PY = load_para(
        N=N, M=M, baseMVA=baseMVA, omega_s=omega_s, net=net, adj_mode=3
    )
    A_3, PY = load_para(
        N=N, M=M, baseMVA=baseMVA, omega_s=omega_s, net=net, adj_mode=2
    )
    a = 60
    if N == 14:
        length = 4000 # load data 4000*14
    elif N == 39:
        length = 1180 # load data 1000*39
    elif N == 118:
        length = 441 # load data 441*118
    f = h5py.File('./data_h5/2/gen.h5', 'r')
    # 调试信息
    print(f"f['gen'].shape: {f['gen'].shape}")
    print(f"f['gen'].dtype: {f['gen'].dtype}")
    one_omega = np.zeros((N))
    one_theta = np.zeros((N))
    for i in range(10):
        one_theta[i] = f['gen'][()][a ,i ,timelength]
        one_omega[i] = f['gen'][()][a, i ,timelength]
    f.close()
    del f
    A_2 = np.abs(np.sin(
            np.repeat(
                a=np.expand_dims(one_theta, axis=1),
                repeats=N,
                axis=1
            )-
            np.repeat(
                a=np.expand_dims(one_theta, axis=0),
                repeats=N,
                axis=0)
        )) * Y

    # A_2 = A_2 / np.amax(A_2)

    config = {
        "font.family":'Times New Roman',
        "font.size": 10/(4/3),
        "mathtext.fontset":'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8/2.54, 3/2.54), dpi=600)
    plt.rc('axes', lw=0.5)
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=15)

    left = 0.15
    bottom = 0.15
    top = 0.25
    h = 1 - bottom - top
    w = h / 8 * 3
    w_i = (1-2*left-3*w)/2
    fig.subplots_adjust(left=left,
                        bottom=bottom,
                        right=1-left,
                        top=bottom + h,
                        wspace=w_i)
    # plt.subplot(1, 3, 1)
    ax1.set_title(r'$\rm B^{I}$', y=1.)
    h1 = ax1.imshow(A_1, cmap='magma')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # plt.subplot(1, 3, 3)
    h2 = ax2.imshow(A_2, cmap='magma')
    ax2.set_title(r'$\rm B^{II}$', y=1.)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # plt.subplot(1, 3, 2)
    h3 = ax3.imshow(A_3, cmap='magma')
    ax3.set_title(r'$\rm B^{III}$', y=1.)
    # ax3.set_xticks([1, 10, 37], ['2', '11', '38'])
    # ax3.set_yticks([1, 10, 37], ['2', '11', '38'])
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # plt.axis('off')

    #colorbar 左 下 宽 高
    l = 1 - left + 0.01
    b = bottom
    w = 0.01
    h = h
    rect = [l, b, w, h]
    cbar_ax = fig.add_axes(rect)
    cb = plt.colorbar(h3, cax=cbar_ax)
    cb.ax.tick_params(direction='out', width=0.5, length=2)
    plt.savefig('./data_train/1-2/comparison_coupling_matrix_39.png')
    plt.savefig('./data_train/1-2/comparison_coupling_matrix_39.eps', format='eps')
    plt.show()

def draw_basin_one_14():
    length = 4000
    timelength = 400

    config = {
        "font.family":'Times New Roman',
        "font.size": 10/(4/3),
        "mathtext.fontset":'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    a = np.array([ 0.03588299, -0.06105624, -0.20931958, -0.16787398, -0.13939053, -0.25496969,
                    -0.23562615, -0.23731321, -0.26914548, -0.2758164,  -0.27045174, -0.27830865,
                    -0.28178378, -0.30140728, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
                ]) # IEEE-14的同步状态
    plt.figure(num=1, figsize=(16/2.54, 10/2.54), dpi=600)
    # plt.rc('font', family='Times New Roman', size=10/(4/3)*2)
    for i in range(14):
        f = h5py.File(r'F:\Swing\result\IEEE\case14\omega==20\change_one_node_long\4000_no_interval\%s.h5' % (i+1), 'r')
        S = f['Y'][()]
        Y = []
        for j in S:
            if j == 0:
                Y.append('g')
            elif j == 1:
                Y.append('k')
        plt.subplot(2, 7, i + 1)

        if i == 0:
            plt.xticks([])
            # plt.yticks(ticks=[-20, 0, 20], label=[-20, 0, 20])
        elif i == 7:
            plt.xticks(ticks=[-math.pi + a[i]], labels=[r'$-\pi+\delta(0)$'])
        # elif i == 10:
        #     tick = [-math.pi + a[i], math.pi + a[i]]
        #     label = [r'$-\pi+\delta_s$', r'$\pi+\delta_s$']
        #     plt.xticks(ticks=tick, labels=label)
        #     plt.yticks([])
        elif i == 13:
            plt.xticks(ticks=[math.pi + a[i]], labels=[r'$\pi+\delta(0)$'])
            plt.yticks([])
        else:
            plt.xticks([])
            plt.yticks([])

        plt.scatter(f['data_theta'][()][:, i*timelength], f['data_omega'][()][:, i*timelength], c=Y, s=0.5, marker='.', alpha=1)
        f.close()
        del f

        # plt.xlabel('%s' % (i+1))
    # plt.suptitle('Basin of Attraction')
    # plt.savefig('F:/Swing/basin_14.png', bbox_inches='tight')
    plt.savefig('F:/Swing/basin_14.png')
    plt.savefig('F:/Swing/basin_14.eps', format='eps')
    plt.show()

def draw_basin_one_118(node):
    
    data_type = 'IEEE'
    length = 441
    interval = False
    timelength = 100
    a = np.array([0.17335432, 0.18560887, 0.19125773, 0.2615371,  0.27018466, 0.22044063,
                  0.21260333, 0.36285742, 0.50028504, 0.64580355, 0.21431102, 0.20675767,
                  0.18669174, 0.19251879, 0.18404262, 0.19997606, 0.23382426, 0.19042649,
                  0.17914191, 0.19316219, 0.21993003, 0.26552547, 0.35850824, 0.3572315,
                  0.49161998, 0.52422305, 0.25460602, 0.22435134, 0.20839068, 0.32643439,
                  0.21143266, 0.24452551, 0.16952152, 0.17967459, 0.17089032, 0.17081699,
                  0.18861586, 0.29390144, 0.10463983, 0.07450999, 0.05610344, 0.05883873,
                  0.1796129,  0.2278303,  0.26148097, 0.31588945, 0.36075892, 0.34520813,
                  0.36442366, 0.31876906, 0.26162657, 0.24236868, 0.22238391, 0.23771163,
                  0.23557995, 0.23666676, 0.26321362, 0.24524271, 0.35598702, 0.43006953,
                  0.44572315, 0.43877892, 0.42042098, 0.45411405, 0.51486606, 0.53811889,
                  0.47577238, 0.50981939, 0.54575048, 0.39733138, 0.38880327, 0.36098612,
                  0.38550631, 0.37679343, 0.39980408, 0.37411307, 0.46879804, 0.46593663,
                  0.478486,   0.54360431, 0.52166227, 0.50234625, 0.54235983, 0.61531696,
                  0.65940195, 0.63489756, 0.6411435,  0.7433833,  0.83779115, 0.62744912,
                  0.59299285, 0.55542018, 0.5185552,  0.49679022, 0.48533945, 0.49562239,
                  0.51148517, 0.50034227, 0.48159233, 0.49303546, 0.50130629, 0.53506043,
                  0.42724782, 0.37353092, 0.3538011,  0.3492747,  0.30281171, 0.33390845,
                  0.32674094, 0.31523114, 0.3426475,  0.26777392, 0.23324874, 0.23806276,
                  0.23793752, 0.50236469, 0.17634976, 0.37862358, 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0.
                ])  # IEEE-118的同步状态
    path = 'F:/Swing/result/' + str(data_type) + '/case118' + '/omega=20' + '/change_one_node_long/' + str(length)
    if not interval:
        path += '_no_interval'
    path += '/' + str(node) + '.h5'
    f = h5py.File(path, 'r')
    data_theta = f['data_theta'][()]
    data_omega = f['data_omega'][()]
    output = f['Y'][()]
    f.close()
    del f
    """
    绘画单个节点的basin
    """
    # plt.figure(figsize=(3, 6))
    # s = output.T.ravel()
    # SS = []
    # for i in range((l+1)**2):
    #     if s[i]==0:
    #         SS.append('yellow')
    #     if s[i]==2:
    #         SS.append('k')
    #     if s[i]==1:
    #         SS.append('r')
    # x = np.linspace(-theta, theta, l+1)
    # y = np.linspace(-omega, omega, l+1)
    # xx, yy = np.meshgrid(x, y)
    # X = xx.flatten()
    # Y = yy.flatten()
    # plt.scatter(X, Y, c=SS, s=5)
    # plt.title('basin of node_%s, s=%s' %(node, str(np.sum(abs(output)))))
    config = {
        "font.family": 'Times New Roman',
        "font.size": 10/(4/3),
        "mathtext.fontset":'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    plt.figure(figsize=(6/2.54, 5/2.54), dpi=600)
    SS = []
    for i in range(length):
        if output[i] == 0:
            SS.append('w')
        elif output[i] == 1:
            SS.append('k')
    plt.axes([0.15, 0.13*1.2, 0.7, 0.7*1.2])
    plt.scatter(data_theta[:,(node-1)*timelength],data_omega[:,(node-1)*timelength], c=SS, s=5)
    # plt.title('basin of node_%s, s=%s' % (node, length-np.sum(output)))
    xtick = [-math.pi, -math.pi/2, 0, math.pi/2, math.pi]
    xlabel = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']
    plt.xticks(ticks=xtick, labels=xlabel)
    plt.xlabel(r'$\delta - \delta(0)$', labelpad=1)
    plt.ylabel(r'$\omega - \omega_s$',  labelpad=1)

    # if np.sum(output) > 0:
    #     if os.path.exists('F:/Swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s_no_interval/figure/basin_%s.png' %(N, omega, (l+1)**2, node)):
    #         plt.savefig('F:/Swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s_no_interval/figure/basin_%s.png' %(N, omega, (l+1)**2, node), dpi=300, bbox_inches='tight')
    #     else:
    #         pass
    # else:
    #     pass
    # if omega==100:
    #     plt.savefig('/home/duguyuan/Pictures/basin_%s_%s_large.png' %(node_initial,N))
    # else:
    #     plt.savefig('/home/duguyuan/Pictures/basin_%s_%s.png' %(node_initial,N))
    
    # plt.savefig('F:/Swing/basin_118_1.png', bbox_inches='tight')
    plt.savefig('F:/Swing/explain_basinStability_1.png')
    plt.savefig('F:/Swing/explain_basinStability_1.eps', format='eps')
    plt.show()

def draw_training_curve(HISTORY, N, exp_num):

    epochs          = range(0, len(HISTORY[0, :])+1)
    acc_values      = HISTORY[0, :]
    val_acc_values  = HISTORY[1, :]
    loss_values     = HISTORY[2, :]
    val_loss_values = HISTORY[3, :]
    acc_values = np.hstack(([0.604], acc_values))
    val_acc_values = np.hstack(([0.604], val_acc_values))
    loss_values = np.hstack(([5.04], loss_values))
    val_loss_values = np.hstack(([5.04], val_loss_values))
    config = {
        "font.family":'Times New Roman',
        "font.size": 10/(4/3),
        "mathtext.fontset":'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    # fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54), dpi=300)
    fig = plt.figure(figsize=(10/2.54, 6/2.54), dpi=300)
    plt.rc('axes', lw=0.5)
    ax1 = fig.subplots()
    left = 0.15
    bottom = 0.15
    plt.subplots_adjust(
        left=left, bottom=bottom, right=1-2*bottom+left, top=1-bottom
    )
    ax2 = ax1.twinx()
    # ax1.grid()
    ax1.plot(
        epochs, acc_values,     'r',
        # linestyle='-',
        linewidth=.5,
        label='ACC'
    )
    print(type(acc_values))
    ax1.plot(
        epochs, val_acc_values, 'm',
        # linestyle='-.',
        linewidth=.5,
        label='Val_ACC'
    )
    ax2.plot(
        epochs, loss_values,    'b',
        # linestyle='--',
        linewidth=.5,
        label='Loss'
    )
    ax2.plot(
        epochs, val_loss_values,'c',
        # linestyle=':', 
        linewidth=.5,
        label='Val_Loss'
    )
    ax1.set_ylim([0.5, 1])
    ax2.set_ylim([0, 5])
    ax1.set_xlim([0, 201])
    # plt.title('Training and Validation accuracy(test_size=%s,chosedlength=%s)' % (TEST_SIZE, chosedlength))
    ax1.tick_params(direction='in', width=0.5, length=2)
    ax2.tick_params(direction='in', width=0.5, length=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('ACC')
    ax2.set_ylabel('Loss')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fancybox=False, facecolor='none', edgecolor='k', loc='best')

    # 设置图例边框宽度
    legend = ax1.get_legend()
    legend.get_frame().set_linewidth(0.5)

    # from brokenaxes import brokenaxes
    # fig = plt.figure(figsize=(8/2.54, 6/2.54), dpi=300)
    # plt.rc('axes', lw=0.5)
    # ax1 = fig.subplots()
    # ax1.spines['left'].set_visible(False)
    # # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    # ax1.spines['bottom'].set_visible(False)
    # # fig, ax1 = plt.subplots(figsize=(8/2.54, 4/2.54), dpi=300)
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    # ax1.tick_params(direction='in', width=0.5, length=2)
    # left = 0.15
    # bottom = 0.15
    # plt.subplots_adjust(
    #     left=left, bottom=bottom, right=1-2*bottom+left, top=1-bottom
    # )
    # bax = brokenaxes(
    #     ylims=((0, 0.2), (0.8, 1)),
    #     # ylims=((1, 500), (600, 10000)),
    #     # xscale='log', yscale='log',
    #     hspace=(1-2*bottom)/1
    # )
    # bax.plot(
    #     epochs, acc_values,     'r',
    #     # linestyle='-',
    #     linewidth=.5,
    #     label='acc'
    # )
    # bax.plot(
    #     epochs, val_acc_values, 'm',
    #     # linestyle='-.',
    #     linewidth=.5,
    #     label='val_acc'
    # )
    # bax.plot(
    #     epochs, loss_values,    'b',
    #     # linestyle='--',
    #     linewidth=.5,
    #     label='loss'
    # )
    # bax.plot(
    #     epochs, val_loss_values,'c',
    #     # linestyle=':',
    #     linewidth=.5,
    #     label='val_loss'
    # )
    # # bax.set_ylim([0, 1])
    # # # ax2.set_ylim([0, 1])
    # # bax.set_xlim([-10, 310])
    # # plt.title('Training and Validation accuracy(test_size=%s,chosedlength=%s)' % (TEST_SIZE, chosedlength))
    # bax.tick_params(direction='in', width=0.5, length=2)
    # bax.set_xlabel('Epochs')
    # bax.set_ylabel('Acc')
    # bax.despine = True

    # ax2 = ax1.twinx()
    # ax2.spines['left'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)
    # # ax1.grid()
    # ax2.tick_params(direction='in', width=0.5, length=2)
    # ax2.set_ylabel('Loss')

    # legend = bax.legend(fancybox=False, facecolor='none', edgecolor='k')
    # legend.get_frame().set_linewidth(0.5)

    plt.savefig('./result_fre/result_fre/1/acc_%s_RGCN-TCN_%s.png' % (N, exp_num))
    plt.savefig('./result_fre/result_fre/1/acc_%s_RGCN-TCN_%s.eps' % (N, exp_num), format='eps')
    plt.show()

def draw_matrix(con_mat):
    
    con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
    con_mat_norm = np.around(con_mat_norm, decimals=2)
    # con_mat_norm = con_mat
    config = {
        "font.family":'Times New Roman',
        "font.size": 10/(4/3),
        "mathtext.fontset":'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    fig, ax = plt.subplots(figsize=(4/2.54, 4/2.54), dpi=300)
    left = 0.15
    bottom = 0.15
    plt.subplots_adjust(
        left=left, bottom=bottom, right=1-2*bottom+left, top=1-bottom
    )
    l = 1-2*bottom+left + 0.02
    b = bottom
    w = 0.03
    h = 1 - 2*bottom
    rect = [l, b, w, h]
    cbar_ax = fig.add_axes(rect)
    sns.heatmap(
        con_mat_norm, annot=True, ax=ax,
        # square=True,
        cmap='Blues', cbar_ax=cbar_ax,
        xticklabels=True)
    # plt.ylim(0, 2)
    # plt.title('Matrix(test_size=%s,chosedlength=%s)' % (TEST_SIZE, chosedlength))
    # plt.xlabel('Predicted labels', labelpad=4)
    # plt.ylabel('True labels', labelpad=4)
    plt.savefig('./result/result/1/matrix_%s_RGCN-TCN_%s.png' % (N, exp_num))
    plt.savefig('./result/result/1/matrix_%s_RGCN-TCN_%s.eps' % (N, exp_num), format='eps')
    plt.show()

def draw_roc_auc(fpr, tpr, auc):
        fig, ax = plt.subplots(figsize=(4/2.54, 4/2.54), dpi=300)
        plt.rc('axes', lw=0.5)
        left = 0.15
        bottom = 0.15
        plt.subplots_adjust(
            left=left, bottom=bottom, right=1-2*bottom+left, top=1-bottom
        )
        
        plt.grid()  # 生成网格
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, linewidth=2, label='AUC = {:.3f}'.format(auc))
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        # plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.tick_params(top=True, right=True, direction='in')
        legend = plt.legend(loc='best', fancybox=False, facecolor='w', edgecolor='k', framealpha=1)
        legend.get_frame().set_linewidth(0.5)
        plt.savefig('./result/1/roc_%s_RGCN-TCN_%s.png' % (N, exp_num))
        plt.savefig('./result/1/roc_%s_RGCN-TCN_%s.eps' % (N, exp_num), format='eps')
        plt.show()

def draw_training():

    exp_num = 12
    N = 118
    if True:
        path = './result_fre/result_fre/1/histroy_61_freq'
        f = h5py.File(path + '.h5', 'r')
        HISTORY = f['train_history'][()]
        test_loss = f['test_loss'][()]
        test_acc = f['test_accuracy'][()]
        con_mat = f['test_matrix'][()]
        auc = f['test_AUC'][()]
        fpr = f['test_fpr'][()]
        tpr = f['test_tpr'][()]
        f.close()
        del f, path
        print('model test loss: ', test_loss)
        print('model test accuracy: ', test_acc)
        print('max  val_acc appear at:%s' % (np.argmax(HISTORY[1, :])+1))
        print('max val_loss appear at:%s' % (np.argmax(-HISTORY[3, :])+1))
        draw_training_curve(HISTORY, N, exp_num)
        del HISTORY
        print(con_mat)
        draw_matrix(con_mat=con_mat)
        del con_mat
        print(auc)
        print(fpr)
        print(tpr)
        # # draw_roc_auc(fpr=fpr, tpr=tpr, auc=auc)
        # del fpr, tpr


def check_position(lst):
    for index, value in enumerate(lst):
        if value == 1:
            return index
    return None


def IEEE2TDS():

    exp_num = 120
    early_stop = True
    interval = True
    relative = False
    normalize = False
    standard = False
    mode = 1
    move = False
    WSZ = 11
    net = 'RGCN-TCN'
    chosedlength = 99
    path = 'F:/Swing/result/experiment/' + str(N) + '/' + str(data_set) + '/' + str(net) + '/' + str(exp_num)
    path += '/TDS' + str(chosedlength)
    if interval:
        path += '_interval'
    if relative:
        path += '_relative'
    if normalize:
        path += '_norm'
    if standard:
        path += '_std_' + str(mode)
    if move:
        path += 'move'
    if early_stop:
        path += '_es'
    path = './result/result/4/histroy_51_both_2'
    f = h5py.File(path + '.h5', 'r')
    test_loss = f['test_loss'][()]
    test_acc = f['test_accuracy'][()]
    con_mat = f['test_matrix'][()]
    # fpr = f['test_fpr'][()]
    # tpr = f['test_tpr'][()]
    # auc = f['test_AUC'][()]
    predict = f['pre'][()]
    origin = f['origin'][()]
    print(len(predict))
    print(len(origin))
    print(type(predict), type(origin))
    f.close()
    del f
    print('model test loss: ', test_loss)
    print('model test accuracy: ', test_acc)
    print(con_mat)
    # draw_matrix(con_mat=con_mat)
    # print(auc)
    # draw_roc_auc(fpr=fpr, tpr=tpr, auc=auc)
    
    """
    draw prediction result versus true data
    """
    plt.figure(figsize=(16/2.54, 4/2.54), dpi=300)
    plt.rc('font', family='Times New Roman', size=10/(4/3))
    plt.rc('axes', lw=0.5)
    left = 0.04
    bottom = 0.015
    w = 0.035
    h = 0.26
    h_i = (1-2*bottom-3*h)/2
    w_i = (1-2*left-10*w)/9
    plt.subplots_adjust(left=left,
                        bottom=bottom,
                        right=1-left,
                        top=1-bottom,
                        wspace=w_i,
                        hspace=h_i)
    length = 177
    data = 4
    x = np.linspace(-math.pi, math.pi, 13)
    y = np.linspace(-20, 20, 13)
    [Y, X] = np.meshgrid(y, x)
    x = X.reshape(1, -1)
    y = Y.reshape(1, -1)
    # for node in range(30, N + 1):
    if True:
        node = 30
        # f = h5py.File('./data_h5/3/label_freq.h5' , 'r')
        # # f = h5py.File('/public/home/spy2018/swing/result/TDS/case%s/omega==%s/change_one_node_long/%s/%s.h5' % (N, omega, length, node), 'r')
        # Y_test = f['label_freq'][()]
        # Y_test = np.array(Y_test)
        # f.close()
        Y_test = origin
        print(origin[0])
        print(predict[0])
        # del f
        y_test = []
        for i in Y_test:
            if check_position(i) == 0:
                y_test.append('g')
            elif check_position(i) == 1:
                y_test.append('y')
            elif check_position(i) == 1:
                y_test.append('b')
            elif check_position(i) == 1:
                y_test.append('r')
        print(len(y_test))
        # if data < 3:
        #     Y_pre = predict[(node-1)*length:node*length]
        # else:
        #     Y_pre = predict[(node-30)*length:(node-29)*length]
        Y_pre = np.rint(predict)
        y_pre = []
        for i in Y_pre:
            if check_position(i) == 0:
                y_pre.append('g')
            elif check_position(i) == 1:
                y_pre.append('y')
            elif check_position(i) == 1:
                y_pre.append('b')
            elif check_position(i) == 1:
                y_pre.append('r')
        print(len(y_pre))
        y_err = []
        for i in range(len(y_pre)):
            if y_pre[i] != y_test[i]:
                y_err.append('r')
            else:
                y_err.append('w')
        
        plt.subplot(1, 3, 1)
        plt.tick_params(width=0.5, length=2)
        # plt.axes([left+(i-30)*(w+w_i), bottom+2*(h+h_i), w, h])
        # plt.xticks(fontsize=5, rotation=90)
        plt.xticks([])
        if node == 30:
            plt.yticks(ticks=[20], labels=[20])
        else:
            plt.yticks([])
        plt.scatter(x, y, c=y_test[0:169], s=.2)
        # plt.title('True data', fontsize=10)

        plt.subplot(1, 3, 2)
        plt.tick_params(width=0.5, length=2)
        # plt.axes([left+(i-30)*(w+w_i), bottom+h+h_i, w, h])
        # plt.xticks(fontsize=5, rotation=90)
        plt.xticks([])
        if node == 30:
            plt.yticks(ticks=[0], labels=[0])
        else:
            plt.yticks([])
        plt.scatter(x, y, c=y_pre[0:169], s=.2)
        # plt.title('%.3f' % (test_acc), fontsize=10)

        plt.subplot(1, 3, 3)
        plt.tick_params(width=0.5, length=2)
        # plt.axes([left+(i-30)*(w+w_i), bottom, w, h])
        # plt.xticks(fontsize=5, rotation=90)
        plt.xticks([])
        if node == 30:
            plt.yticks(ticks=[-20], labels=[-20])
        else:
            plt.yticks([])
        plt.scatter(x, y, c=y_err[0:169], s=.2)
        # plt.title('%.3f' % (test_loss), fontsize=10)
    plt.savefig('./result/result/1/IEEE-TDS.png' )
    plt.savefig('./result/result/1/IEEE-TDS.eps', format='eps')
    plt.show()

def draw_timestep_effect():

    # ACC_GCN  = np.array([0.950185, 0.9713, 0.9875, 0.99518634, 0.9892461,  0.99384665, 0.9824629, 0.9970551])
    # LOSS_GCN = np.array([0.699564,      0,      0, 0.42895108, 0.18586446, 0.09346658, 1.0383884, 0.12074033])

    # ACC_CNN  = np.array([0.9302073121070862, 0.9585414528846741,  0.9769230484962463,  0.9868131875991821,   0.9474900364875793,  0.9863636493682861])
    # LOSS_CNN = np.array([0.1586904376745224, 0.10175808519124985, 0.06163538619875908, 0.03517157956957817,  0.12375964969396591, 0.03663807362318039])

    ## 2021.11.10
    l         = np.array([     25,      50,      75,     100,     125,     150,     175,     200,      10,      20,      30,      40,      50])
    ACC_RGCN  = np.array([0.98069, 0.98321, 0.98834, 0.99471, 0.99512, 0.99500, 0.99498, 0.99488, 0.98134, 0.98969, 0.99419, 0.99543, 0.99509])
    LOSS_RGCN = np.array([0.10797, 0.10828, 0.07088, 0.05266, 0.04217, 0.03152, 0.03320, 0.01981, 0.10142, 0.07839, 0.04562, 0.03884, 0.02600])
    FPR_RGCN  = np.array([     87,      91,      74,      61,      48,      42,      32,      33,     104,     91,      45,       41,      32])
    FNR_RGCN  = np.array([    120,      89,      51,      36,      19,      18,      24,      27,      96,     41,      28,        8,      21])
    AUC_RGCN  = np.array([0.99735, 0.99821, 0.99887, 0.99914, 0.99956, 0.99946, 0.99940, 0.99943, 0.99773, 0.99844, 0.99919, 0.99897, 0.99969])

    delta = 1

    if delta == 0.05:

        length = np.array([50, 100, 150, 200])
        config = {
            "font.family":'Times New Roman',
            "font.size": 10/(4/3),
            "mathtext.fontset":'stix',
            "font.serif": ['SimSun'],
        }
        rcParams.update(config)
        fig = plt.figure(figsize=(4/2.54, 6/2.54), dpi=300)
        plt.rc('axes', lw=0.5)
        ax1 = fig.subplots()
        left = 0.15
        bottom = 0.15
        plt.subplots_adjust(
            left=left, bottom=bottom, right=1-2*bottom+left, top=1-bottom
        )
        ax2 = ax1.twinx()
        ax1.plot(length, ACC_GCN[:4], label='Acc')
        ax1.plot(length, LOSS_GCN[:4], label='Loss')
        ax1.tick_params(direction='in', width=0.5, length=2)
        legend = ax1.legend(fancybox=False, facecolor='none', edgecolor='k')
        legend.get_frame().set_linewidth(0.5)
        ax1.set_xlabel(r'Number of $l$')
        ax1.set_ylabel('Acc')
        ax1.set_xticks([50, 100, 150, 200])
        ax1.set_xticklabels([50, 100, 150, 200])
        ax2.tick_params(direction='in', width=0.5, length=2)
        ax2.set_ylabel('Loss')
        plt.suptitle(r'Time step $\Delta$t=0.05s')
        plt.savefig('F:/Swing/length_effect_005.png')
        plt.savefig('F:/Swing/length_effect_005.eps', format='eps')
        plt.show()

    elif delta == 0.2:

        length = np.array([20, 50])
        config = {
            "font.family":'Times New Roman',
            "font.size": 10/(4/3),
            "mathtext.fontset":'stix',
            "font.serif": ['SimSun'],
        }
        rcParams.update(config)
        fig = plt.figure(figsize=(4/2.54, 6/2.54), dpi=300)
        plt.rc('axes', lw=0.5)
        ax1 = fig.subplots()
        left = 0.15
        bottom = 0.15
        plt.subplots_adjust(
            left=left, bottom=bottom, right=1-2*bottom+left, top=1-bottom
        )
        ax2 = ax1.twinx()
        ax1.plot(length, ACC_GCN[4:], label='Acc')
        ax1.plot(length, LOSS_GCN[4:], label='Loss')
        ax1.tick_params(direction='in', width=0.5, length=2)
        legend = ax1.legend(fancybox=False, facecolor='none', edgecolor='k')
        legend.get_frame().set_linewidth(0.5)
        ax1.set_xlabel(r'Number of $l$')
        ax1.set_ylabel('Acc')
        ax1.set_xticks([20, 50])
        ax1.set_xticklabels([20, 50])
        ax2.tick_params(direction='in', width=0.5, length=2)
        ax2.set_ylabel('Loss')
        plt.suptitle(r'Time step $\Delta$t=0.2s')
        plt.savefig('F:/Swing/length_effect_020.png')
        plt.savefig('F:/Swing/length_effect_020.eps', format='eps')
        plt.show()
    
    else:

        config = {
            "font.family":'Times New Roman',
            "font.size": 10/(4/3),
            "mathtext.fontset":'stix',
            "font.serif": ['SimSun'],
        }
        rcParams.update(config)
        fig = plt.figure(figsize=(8/2.54, 4/2.54), dpi=300)
        plt.rc('axes', lw=0.5)
        ax1 = fig.subplots()
        left = 0.15
        bottom = 0.15
        plt.subplots_adjust(
            left=left, bottom=bottom, right=1-2*bottom+left, top=1-bottom
        )
        plt.grid(color='k', linestyle=':', linewidth=0.5)
        length = np.array([25, 50, 75, 100, 125, 150, 175, 200])
        ax1.plot(length + 1, ACC_RGCN[:8] * 100,
            color='k', linewidth=0.5, linestyle='-',
            marker='s', markersize=1, markerfacecolor='k',
            label=r'$\Delta t=0.05s$'
        )
        ax1.plot(101, ACC_RGCN[3] * 100,
            marker='s', markersize=2, markerfacecolor='b'
        )
        # length = np.array([10, 20, 30, 40, 50])
        # ax1.plot(length*0.2, ACC_RGCN[8:] * 100,
        #     color='r', linewidth=0.5, linestyle='-.',
        #     marker='o', markersize=1, markerfacecolor='r',
        #     label=r'$\Delta t=0.2s$'
        # )
        plt.ylim((98, 100))
        ax1.tick_params(direction='in', width=0.5, length=2)
        # legend = ax1.legend(fancybox=False, facecolor='none', edgecolor='k', loc='lower right')
        # legend.get_frame().set_linewidth(0.5)
        ax1.set_xlabel(r'$l$', labelpad=0)
        ax1.set_ylabel('Acc (%)')
        # ax1.set_xticks([20, 50])
        # ax1.set_xticklabels([20, 50])
        plt.savefig('F:/Swing/length_effect.png')
        plt.savefig('F:/Swing/length_effect.eps', format='eps')
        plt.show()

def hidden_layer():
    interval = True
    mode = 'colorbar'
    # path = 'F:/Swing/result/experiment/' + str(N) + '/' + str(data_set) + '/' + str(net) + '/' + str(exp_num)
    # path = '/media/duguyuan/new/Swing/result/experiment/' + str(N) + '/' + str(data_set) + '/' + str(net) + '/' + str(exp_num)
    path = '/public/home/spy2018/swing/result/experiment/' + str(N) + '/' + str(data_set) + '/' + str(net) + '/' + str(exp_num)
    path += '/history' + str(chosedlength)
    if interval:
        path += '_interval'
    if relative:
        path += '_relative'
    if normalize:
        path += '_norm'
    if standard:
        path += '_std_' + str(mode)
    if move:
        path += 'move'
    if early_stop:
        path += '_es'

    f = h5py.File(path + '_middle_output.h5', 'r')

    output_7_train = f['/middle/7/train'][()]
    print(output_7_train.shape)
    Y_train = f['/true/train'][()]
    Y_predict_train = f['/output/train'][()]
    Y_predict_train_int = np.rint(Y_predict_train)

    output_7_TDS = f['/middle/7/TDS'][()]
    print(output_7_TDS.shape)
    Y_TDS = f['/true/TDS'][()]
    Y_predict_TDS = f['/output/TDS'][()]
    Y_predict_TDS_int = np.rint(Y_predict_TDS)
    
    f.close()
    del f

    color_train = []
    for i in Y_train:
        if i == 0:
            color_train.append('g')
        else:
            color_train.append('k')
    color_train_pre = []
    for i in Y_predict_train_int:
        if i == 0:
            color_train_pre.append('g')
        else:
            color_train_pre.append('k')
    color_train_error = []
    for i in range(Y_train.shape[0]):
        if Y_train[i] != Y_predict_train_int[i]:
            color_train_error.append('r')
        else:
            color_train_error.append('w')
    
    color_TDS = []
    for i in Y_TDS:
        if i == 0:
            color_TDS.append('g')
        else:
            color_TDS.append('k')
    color_TDS_pre = []
    for i in Y_predict_TDS_int:
        if i == 0:
            color_TDS_pre.append('g')
        else:
            color_TDS_pre.append('k')
    color_TDS_error = []
    for i in range(Y_TDS.shape[0]):
        if Y_TDS[i] != Y_predict_TDS_int[i]:
            color_TDS_error.append('r')
        else:
            color_TDS_error.append('w')

    '''
    2×4
    '''
    if mode == 'true-pre-error':
        ## true-pre-error

        output = np.concatenate((output_7_train, output_7_TDS), axis=0)
        # output = np.reshape(output, (output.shape[0], 39*16))
        del output_7_train, output_7_TDS, Y_train, Y_predict_train_int, Y_TDS, Y_predict_TDS_int
        tsne = TSNE(n_components = 2, init='pca', random_state=0)
        tsne = tsne.fit_transform(output)
        a = 34893    
        plt.figure(figsize=(16/2.54, 8/2.54))

        plt.subplot(2, 4, 1)
        plt.xlim(-80, 80)
        plt.ylim(-80, 80)
        plt.scatter(tsne[:a, 0], tsne[:a, 1], s=1, c=color_train)
        plt.subplot(2, 4, 2)
        plt.xlim(-80, 80)
        plt.ylim(-80, 80)
        plt.scatter(tsne[:a, 0], tsne[:a, 1], s=1, c=color_train_pre)
        plt.subplot(2, 4, 3)
        plt.xlim(-80, 80)
        plt.ylim(-80, 80)
        plt.scatter(tsne[:a, 0], tsne[:a, 1], s=1, cmap='RdYlBu', c=Y_predict_train)
        plt.subplot(2, 4, 4)
        plt.xlim(-80, 80)
        plt.ylim(-80, 80)
        plt.scatter(tsne[:a, 0], tsne[:a, 1], s=1, c=color_train_error)

        plt.subplot(2, 4, 5)
        plt.xlim(-80, 80)
        plt.ylim(-80, 80)
        plt.scatter(tsne[a:, 0], tsne[a:, 1], s=1, c=color_TDS)
        plt.subplot(2, 4, 6)
        plt.xlim(-80, 80)
        plt.ylim(-80, 80)
        plt.scatter(tsne[a:, 0], tsne[a:, 1], s=1, c=color_TDS_pre)
        plt.subplot(2, 4, 7)
        plt.xlim(-80, 80)
        plt.ylim(-80, 80)
        plt.scatter(tsne[a:, 0], tsne[a:, 1], s=1, cmap='RdYlBu', c=Y_predict_TDS)
        plt.subplot(2, 4, 8)
        plt.xlim(-80, 80)
        plt.ylim(-80, 80)
        plt.scatter(tsne[a:, 0], tsne[a:, 1], s=1, c=color_TDS_error)
        
        plt.show()
    
    elif mode == 'colorbar':
        '''
        colorbar
        '''
        output = np.concatenate((output_7_train, output_7_TDS), axis=0)
        del output_7_train, output_7_TDS, Y_train, Y_predict_train_int, Y_TDS, Y_predict_TDS_int
        tsne = TSNE(n_components = 2, init='pca', random_state=0)
        tsne = tsne.fit_transform(output)
        a = 34893
        x = tsne[:a, 0]
        x_a = np.array([-80, -80, -5, -25, -25, 15,  15, 80,  80, 125/2, 20])
        y = tsne[:a, 1]
        y_a = np.array([ 80, -80, 80, -80, -70, 80, -80, 80, -80,   -40, 65])
        z = np.squeeze(Y_predict_train)
        z_a = np.array([  0,   0,  0,   0,   0,  1,   1,  1,   1,     1,  1])

        del_index = []
        for i in range(x.shape[-1]):
            if z[i] < 0.5:
                if x[i] >= 10:
                    del_index.append(i)
                elif x[i] >= 8 and y[i] >= -18 and y[i] < -40:
                    del_index.append(i)
                elif x[i] >= 0 and y[i] >= 45:
                    del_index.append(i)
                elif x[i] >= 5 and y[i] <= 21:
                    del_index.append(i)
                else:
                    pass
            else:
                if x[i] <= 12:
                    del_index.append(i)
                elif x[i] >= 0 and x[i] < 10 and y[i] >= 18 and y[i] < 25:
                    del_index.append(i)
                else:
                    pass
        x_col = np.delete(x, del_index, axis=0)
        y_col = np.delete(y, del_index, axis=0)
        z_col = np.delete(z, del_index, axis=0)
        x_col = np.concatenate((x_col, x_a), axis=0)
        y_col = np.concatenate((y_col, y_a), axis=0)
        z_col = np.concatenate((z_col, z_a), axis=0)

        # x_col = np.concatenate((x, x_a), axis=0)
        # y_col = np.concatenate((y, y_a), axis=0)
        # z_col = np.concatenate((z, z_a), axis=0)

        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        from matplotlib import cm
        # clist = ['lightgrey','firebrick','lime']
        # newcmap = ListedColormap(["coral", "w", "skyblue"])
        viridis_big = cm.get_cmap('RdYlBu', 512)
        newcmap = ListedColormap(viridis_big(np.linspace(0.3, 0.7, 256)))
        config = {
            "font.family":'Times New Roman',
            "font.size": 20/(4/3),
            "mathtext.fontset":'stix',
            "font.serif": ['SimSun'],
        }
        rcParams.update(config)
        ## train
        fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54), dpi=300)
        plt.rc('axes', lw=0.5)
        left = 0.15
        bottom = 0.15
        plt.subplots_adjust(
            left=left, right=1-left, bottom=bottom, top=1-bottom
        )
        cntr2 = ax.tricontourf(x_col, y_col, z_col, levels=5, cmap=newcmap)
        # cntr3 = ax.tricontourf([10, 80, 10, 80], [-80, -80, -15, -15], [1., 1., 1., 1.], levels=1, cmap=newcmap)
        # cntr4 = ax.tricontourf([12.5, 80, 12.5, 80], [-80, -80, 80, 80], [1., 1., 1., 1.], levels=1, cmap=newcmap)
        # cntr5 = ax.tricontourf([-25, -10, -25, -10], [-75, -50, -25, -50], [0., 0., 0., 0.], levels=1, cmap=newcmap)
        cntr1 = ax.scatter(x, y, s=0.1, cmap='RdYlBu', c=z)
        ax.set_xticks([-75, -50, -25, 0, 25, 50, 75])
        ax.set_xticklabels([-75, -50, -25, 0, 25, 50, 75])
        ax.set_yticks([-75, -50, -25, 0, 25, 50, 75])
        ax.set_yticklabels([-75, -50, -25, 0, 25, 50, 75])
        # colorbar
        l = 1 - left + 0.02
        b = bottom
        w = 0.035
        h = 1 - 2*bottom
        rect = [l, b, w, h]
        cbar_ax = fig.add_axes(rect)
        cb = plt.colorbar(cntr1, cax=cbar_ax)
        cb.ax.tick_params(direction='in', width=0.5, length=2)
        cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cb.set_ticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set(xlim=(-80, 80), ylim=(-80, 80))
        ax.tick_params(direction='in')
        plt.savefig('/public/home/spy2018/swing/hidden_layer_train.png')
        plt.savefig('/public/home/spy2018/swing/hidden_layer_train.eps', format='eps')

        ## TDS
        x_TDS = tsne[a:, 0]
        x_a = np.array([-80, -80, -5, -25, -25, 15,  15, 80,  80, 125/2, 20])
        y_TDS = tsne[a:, 1]
        y_a = np.array([ 80, -80, 80, -80, -70, 80, -80, 80, -80,   -40, 65])
        z_TDS = np.squeeze(Y_predict_TDS)
        z_a = np.array([  0,   0,  0,   0,   0,  1,   1,  1,   1,     1,  1])
        fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54), dpi=300)
        plt.rc('axes', lw=0.5)

        left = 0.15
        bottom = 0.15
        plt.subplots_adjust(
            left=left, right=1-left, bottom=bottom, top=1-bottom
        )
        cntr2 = ax.tricontourf(x_col, y_col, z_col, levels=5, cmap=newcmap)
        cntr1 = ax.scatter(x_TDS, y_TDS, s=0.1, cmap='RdYlBu', c=z_TDS)
        ax.set_xticks([-75, -50, -25, 0, 25, 50, 75])
        ax.set_xticklabels([-75, -50, -25, 0, 25, 50, 75])
        ax.set_yticks([-75, -50, -25, 0, 25, 50, 75])
        ax.set_yticklabels([-75, -50, -25, 0, 25, 50, 75])
        # colorbar
        l = 1 - left + 0.02
        b = bottom
        w = 0.035
        h = 1 - 2*bottom
        rect = [l, b, w, h]
        cbar_ax = fig.add_axes(rect)
        cb = plt.colorbar(cntr1, cax=cbar_ax)
        cb.ax.tick_params(direction='in', width=0.5, length=2)
        # tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar上的刻度值个数
        # cb.locator = tick_locator
        # cb.update_ticks()
        cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cb.set_ticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set(xlim=(-80, 80), ylim=(-80, 80))
        ax.tick_params(direction='in')
        plt.savefig('/public/home/spy2018/swing/hidden_layer_TDS.png')
        plt.savefig('/public/home/spy2018/swing/hidden_layer_TDS.eps', format='eps')
    
    elif mode == 'true':
        '''
        groud truth
        '''
        output = np.concatenate((output_7_train, output_7_TDS), axis=0)
        del output_7_train, output_7_TDS, Y_train, Y_predict_train_int, Y_TDS, Y_predict_TDS_int
        tsne = TSNE(n_components = 2, init='pca', random_state=0)
        tsne = tsne.fit_transform(output)
        a = 34893
        config = {
            "font.family":'Times New Roman',
            "font.size": 20/(4/3),
            "mathtext.fontset":'stix',
            "font.serif": ['SimSun'],
        }
        rcParams.update(config)

        x = tsne[:a, 0]
        y = tsne[:a, 1]
        # z = np.squeeze(Y_train)
        ## train
        fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54), dpi=300)
        plt.rc('axes', lw=0.5)
        left = 0.15
        bottom = 0.15
        plt.subplots_adjust(
            left=left, right=1-left, bottom=bottom, top=1-bottom
        )
        cntr1 = ax.scatter(x, y, s=0.1, cmap='RdYlBu', c=color_train)
        ax.set_xticks([-75, -50, -25, 0, 25, 50, 75])
        ax.set_xticklabels([-75, -50, -25, 0, 25, 50, 75])
        ax.set_yticks([-75, -50, -25, 0, 25, 50, 75])
        ax.set_yticklabels([-75, -50, -25, 0, 25, 50, 75])
        # colorbar
        l = 1 - left + 0.02
        b = bottom
        w = 0.035
        h = 1 - 2*bottom
        rect = [l, b, w, h]
        cbar_ax = fig.add_axes(rect)
        cb = plt.colorbar(cntr1, cax=cbar_ax)
        cb.ax.tick_params(direction='in', width=0.5, length=2)
        cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cb.set_ticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set(xlim=(-80, 80), ylim=(-80, 80))
        ax.tick_params(direction='in')
        plt.savefig('/public/home/spy2018/swing/hidden_layer_train_gt.png')
        plt.savefig('/public/home/spy2018/swing/hidden_layer_train_gt.eps', format='eps')

        ## TDS
        x_TDS = tsne[a:, 0]
        y_TDS = tsne[a:, 1]
        # z_TDS = np.squeeze(Y_TDS)
        fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54), dpi=300)
        plt.rc('axes', lw=0.5)

        left = 0.15
        bottom = 0.15
        plt.subplots_adjust(
            left=left, right=1-left, bottom=bottom, top=1-bottom
        )
        # cntr2 = ax.tricontourf(np.concatenate((x, x_a), axis=0), np.concatenate((y, y_a), axis=0), np.concatenate((z, z_a), axis=0), levels=5, cmap=newcmap)
        cntr1 = ax.scatter(x_TDS, y_TDS, s=0.1, cmap='RdYlBu', c=color_TDS)
        ax.set_xticks([-75, -50, -25, 0, 25, 50, 75])
        ax.set_xticklabels([-75, -50, -25, 0, 25, 50, 75])
        ax.set_yticks([-75, -50, -25, 0, 25, 50, 75])
        ax.set_yticklabels([-75, -50, -25, 0, 25, 50, 75])
        # colorbar
        l = 1 - left + 0.02
        b = bottom
        w = 0.035
        h = 1 - 2*bottom
        rect = [l, b, w, h]
        cbar_ax = fig.add_axes(rect)
        cb = plt.colorbar(cntr1, cax=cbar_ax)
        cb.ax.tick_params(direction='in', width=0.5, length=2)
        cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cb.set_ticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set(xlim=(-80, 80), ylim=(-80, 80))
        ax.tick_params(direction='in')
        plt.savefig('/public/home/spy2018/swing/hidden_layer_TDS_gt.png')
        plt.savefig('/public/home/spy2018/swing/hidden_layer_TDS_gt.eps', format='eps')

    else:

        pass

    # plt.show()

    # a = 34893
    # plt.figure()
    # plt.scatter(tsne[a:, 0], tsne[a:, 1], s=1, cmap='RdYlBu', c=Y_predict_TDS)
    # plt.xlim(-120, 120)
    # plt.ylim(-120, 120)
    # plt.show()

    # plt.figure()
    # for i in range(N):
    #     plt.subplot(3, 13, i + 1)
    #     plt.xlim(-120, 120)
    #     plt.ylim(-120, 120)
    #     plt.scatter(tsne[i*441:(i+1)*441, 0], tsne[i*441:(i+1)*441, 1], cmap='RdYlBu', c=Y_predict_train[i*441:(i+1)*441], s=1)
    # # plt.subplot(4, 1, 1)
    # # plt.xlim(-120, 120)
    # # plt.ylim(-120, 120)
    # # plt.scatter(tsne[:, 0], tsne[:, 1], s=1)
    # plt.show()

def draw_curve():

    i = 28
    f = h5py.File('F:/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/1000_all/%s.h5' % (N, omega, i+1), 'r')
    data_theta = f['data_theta'][()]
    data_omega = f['data_omega'][()]
    S = f['Y'][()]
    f.close()
    del f
    print(data_theta.shape)
    print(S)
    config = {
        "font.family":'Times New Roman',
        "font.size": 10/(4/3),
        "mathtext.fontset":'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    fig = plt.figure(figsize=(8/2.54, 4/2.54), dpi=300)
    plt.rc('axes', lw=0.5)
    ax1 = fig.subplots()
    left = 0.15
    bottom = 0.15
    plt.subplots_adjust(
        left=left, bottom=bottom, right=1-2*bottom+left, top=1-bottom
    )
    plt.grid()
    ax1.plot(t, (np.squeeze(data_omega)[:38, :] / 100 / math.pi).T,
        color='r'
    )
    ax1.plot(t, (np.squeeze(data_omega)[N - 1, :] / 100 / math.pi).T,
        color='r'
        # ,label='synchronous'
        ,label='nonsynchronous'
    )
    ax1.plot(5 * np.ones((31)), np.arange(-0.15, 0.16, 0.01),
        '--', color='k',
        linewidth=0.5
    )
    ax1.tick_params(direction='in', width=0.5, length=2)
    ax1.set_xlabel('Time (s)', labelpad=0)
    ax1.set_ylabel('Rotor Speed (p.u.)')
    plt.xlim((0, 120))
    ax1.set_xticks([0, 5, 60, 120])
    ax1.set_xticklabels([0, 5, 60, 120])
    # plt.ylim((-0.05, 0.05))
    # ax1.set_yticks([-0.05, 0, 0.05])
    # ax1.set_yticklabels([0.95, 1, 1.05])
    plt.ylim((-0.15, 0.15))
    ax1.set_yticks([-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
    ax1.set_yticklabels([0.85, 0.90, 0.95, 1, 1.05, 1.10, 1.15])
    legend = ax1.legend(fancybox=False, facecolor='none', edgecolor='k', loc='lower right')
    legend.get_frame().set_linewidth(0.5)
    plt.savefig('F:/Swing/nonsyn.png')
    plt.savefig('F:/Swing/nonsyn.eps', format='eps')
    plt.show()

    # i = 30
    # # a = 9 1
    # a = 399
    # f = h5py.File('/run/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/1000_all/%s.h5' % (N, omega, i+1), 'r')
    # # f = h5py.File('F:/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/1000_all/%s.h5' % (N, omega, i+1), 'r')
    # # data_theta = f['data_theta'][()]
    # data_omega = f['data_omega'][()]
    # S = f['Y'][()]
    # f.close()
    # del f
    # print(data_omega.shape)
    # print(S[a])
    # if S[a] == 0:
    #     leng = 'syn'
    #     c = 'g'
    # else:
    #     leng = 'nonsyn'
    #     c = 'r'

    # config = {
    #     "font.family":'Times New Roman',
    #     "font.size": 10/(4/3),
    #     "mathtext.fontset":'stix',
    #     "font.serif": ['SimSun'],
    # }
    # rcParams.update(config)
    # fig = plt.figure(figsize=(8/2.54, 4/2.54), dpi=300)
    # plt.rc('axes', lw=0.5)
    # ax1 = fig.subplots()
    # left = 0.15
    # bottom = 0.15
    # plt.subplots_adjust(
    #     left=left, bottom=bottom, right=1-2*bottom+left, top=1-bottom
    # )
    # plt.grid()
    # ax1.plot(t, (np.squeeze(data_omega[a, :, :])[:38, :] / 100 / math.pi).T,
    #     linewidth=0.5,
    #     color=c
    # )
    # ax1.plot(t, (np.squeeze(data_omega[a, :, :])[N - 1, :] / 100 / math.pi).T,
    #     linewidth=0.5,
    #     color=c
    #     ,label=leng
    # )
    # ax1.plot(5 * np.ones((31)), np.arange(-0.15, 0.16, 0.01),
    #     '--', color='k',
    #     linewidth=0.5
    # )
    # ax1.tick_params(direction='in', width=0.5, length=2)
    # ax1.set_xlabel('Time (s)', labelpad=0)
    # ax1.set_ylabel('Rotor Speed (p.u.)')
    # plt.xlim((0, 120))
    # ax1.set_xticks([0, 5, 60, 120])
    # ax1.set_xticklabels([0, 5, 60, 120])
    # if S[a] == 0:
    #     plt.ylim((-0.05, 0.05))
    #     ax1.set_yticks([-0.05, 0, 0.05])
    #     ax1.set_yticklabels([0.95, 1, 1.05])
    # else:
    #     plt.ylim((-0.15, 0.15))
    #     ax1.set_yticks([-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
    #     ax1.set_yticklabels([0.85, 0.90, 0.95, 1, 1.05, 1.10, 1.15])
    # legend = ax1.legend(fancybox=False, facecolor='none', edgecolor='k', loc='lower right')
    # legend.get_frame().set_linewidth(0.5)
    # plt.savefig('/home/duguyuan/Pictures/%s.png' % (leng))
    # plt.savefig('/home/duguyuan/Pictures/%s.eps' % (leng), format='eps')
    # plt.show()

def cal_se():
    PY, initial = load_para(N=N, M=M, baseMVA=baseMVA, omega_s=omega_s)
    a = np.array([-0.24219997, -0.16992011, -0.21896319, -0.22769395, -0.20274313, -0.18877805,
                -0.23072831, -0.24088105, -0.25411382, -0.14792818, -0.16214242, -0.16401846,
                -0.16169114, -0.1933527,  -0.20324505, -0.17720979, -0.19711253, -0.21354782,
                -0.08796499, -0.11204258, -0.13237097, -0.04721098, -0.05117464, -0.1747437,
                -0.14210796, -0.16254737, -0.20094919, -0.09408921, -0.04086045, -0.12485783,
                -0.021106,   -0.01778558,  0.00184892, -0.02056255,  0.04571267,  0.10145837,
                -0.01671788,  0.08897803, -0.26130884, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0.]) # IEEE-39的同步状态
    
    a[N + 0] += 10
    total = 0
    for i in range(10):
        start = time.perf_counter()
        result = solve_ivp(fun=lambda t, y: dmove(t, y, PY), t_span=(0.0, max_t),  y0=a, method='RK45', t_eval=t)
        end = time.perf_counter()
        total += end - start
    print('训练时长：%ss' %(total / 10))

def draw_spendtime():

    l = np.array([25,     50,     75,     100,    125,    150,    175,    200,    225,    250,    275,    300,    325])
    T = np.array([0.1338, 0.2717, 0.4180, 0.5187, 0.6246, 0.7792, 0.8620, 1.0328, 1.1164, 1.2800, 1.3359, 1.5159, 1.6134])
    config = {
        "font.family":'Times New Roman',
        "font.size": 10/(4/3),
        "mathtext.fontset":'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    fig = plt.figure(figsize=(8 / 2.54, 4 / 2.54), dpi=300)
    plt.rc('axes', lw=0.5)
    ax1 = fig.subplots()
    left = 0.15
    bottom = 0.15
    plt.subplots_adjust(
        left=left, bottom=bottom, right=1-2*bottom+left, top=1-bottom
    )
    plt.grid(color='k', linestyle=':', linewidth=0.5)
    # ax1.plot(l, T)
    ax1.plot(l[:8]+1, T[:8],
        color='k', linewidth=0.5,
        marker='s', markersize=1, markerfacecolor='k'
        ,label=r'$\Delta t=0.05s$'
    )
    ax1.plot(101, T[3],
        marker='s', markersize=2, markerfacecolor='b'
    )
    ax1.tick_params(direction='in', width=0.5, length=2)
    ax1.set_ylabel(r'$T$ (s)')
    ax1.set_xlabel(r'$l$'
        ,labelpad=0
    )
    # plt.xlim((0, 325))
    # plt.ylim((0, 1.75))
    plt.ylim((0, 1.2))
    # legend = ax1.legend(fancybox=False, facecolor='none', edgecolor='k', loc='lower right')
    # legend.get_frame().set_linewidth(0.5)
    # ax1.set_xticks([0, 5, 60, 120])
    # ax1.set_xticklabels([0, 5, 60, 120])
    # plt.savefig('F:/Swing/time.png')
    # plt.savefig('F:/Swing/time.eps', format='eps')
    plt.show()

# power_topology()
# bar()
# draw_matrix_1()
# draw_basin_one_14()
# draw_basin_one_118(node=89)
draw_training()
# IEEE2TDS()
# draw_timestep_effect()
# hidden_layer()
# draw_curve()
# cal_se()
# draw_spendtime()
