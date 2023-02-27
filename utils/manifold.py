import numpy as np
import jittor as jt
jt.dirty_fix_pytorch_runtime_error()
from numpy.linalg import solve
import torch

def get_inter(nearnN, w_c, generated_f=None, feature_list = None, dtype=np.float32, neighbor=False):
    list_len = jt.array([feature_list.shape[0]])


    generated_f = generated_f.view(-1, 7168).cpu().numpy().astype(np.float32)

    b = jt.code([1, nearnN], 
            "int32", [jt.array(feature_list),jt.array(generated_f), list_len], 
    cpu_header="#include <algorithm>",
    cpu_src="""
            using namespace std;
            auto n=out_shape0, k=out_shape1;
            int N=@in2(0);
                        
            // 使用openmp实现自动并行化
            // 存储k近邻的距离和下标
            vector<pair<float,int>> id(N);
            #pragma omp parallel for
            for (int j=0; j<N; j++) {
                auto dis = 0.0;
                for (int d=0; d<7186; d++)
                {
                    auto dx = @in1(0,d)-@in0(j,d);
                    dis = dis +dx*dx;
                }
                id[j] = {dis, j};
            }
            // 使用c++算法库的nth_element排序
            nth_element(id.begin(), 
                id.begin()+k, id.end());
            // 将下标输出到计图的变量中
            for (int j=0; j<k; j++)
                @out(0,j) = id[j].second;
                """
            )
    idx_sort = b[0].numpy()
    A_0 = [feature_list[idx_sort[0],:]]

    A_m = A_0
    for i in range(1,nearnN):
        A_m = np.concatenate((A_m,[feature_list[idx_sort[i],:]]), axis=0)
            
    A_0 = np.array(A_0)
    A_m= np.array(A_m).T

    A_m0 = np.concatenate((A_m[:,1:]-A_0.T, np.ones((1,nearnN-1))*10), axis=0)
    A = np.dot(A_m0.T, A_m0)
    b = np.zeros((1, generated_f.shape[1]+1))

    b[0,0:generated_f.shape[1]] = generated_f-A_0

    B = np.dot(A_m0.T, b.T)

    x = solve(A, B)

    xx = np.zeros((nearnN,1))
    xx[0,0] = 1 - x.sum()
    xx[1:,0] = x[:,0]

    pred_f = np.dot(A_m, xx).T
    vec_mu = pred_f * w_c + (1-w_c)* generated_f
    A_ms = A_m.T 

    # print(pred_f[:8])
    # print(vec_mu[:8])
    # print(generated_f[:8])
    vec_mu = torch.from_numpy(vec_mu.astype(np.float32))
    vec_mu = vec_mu.view(-1, 14, 512).cuda()
    if neighbor:
        return vec_mu, A_ms
    else:
        return vec_mu

