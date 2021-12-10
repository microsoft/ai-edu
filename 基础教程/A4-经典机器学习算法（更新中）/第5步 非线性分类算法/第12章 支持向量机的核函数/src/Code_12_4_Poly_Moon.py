
from Code_12_4_Poly_Circle import *

def poly_svc(X, Y, d, r):
    model = SVC(kernel='poly', degree=d, coef0=r)
    model.fit(X,Y)

    #print("权重:",np.dot(model.dual_coef_, model.support_vectors_))
    #print("支持向量个数:",model.n_support_)
    #print("支持向量索引:",model.support_)
    #print("支持向量:",np.round(model.support_vectors_,3))
    #print("支持向量ay:",np.round(model.dual_coef_,3))
    score = model.score(X, Y)
    #print("准确率:", score)

    return model, score


def classification(X_raw, Y, degree, coef0):
    ss = StandardScaler()
    X = ss.fit_transform(X_raw)

    fig = plt.figure()
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False

    scope = [-2.5,2.5,100,-2.5,2.5,100]    

    for i in range(2):
        for j in range(4):
            start = time.time()
            # 故意循环100次来检查分类运行时间
            for k in range(100):
                idx = i * 4 + j
                d = degree[idx]
                r = coef0[idx]
                model, score = poly_svc(X, Y, d, r)
            end = time.time()
            print("time=", end-start)
            ax = plt.subplot(2,4,idx+1)
            set_ax(ax, scope)
            title = str.format("degree={0},coef0={1}, 准确率={2}", d, r, score)            
            #print(title)
            ax.set_title(title)
            show_predication_result(ax, model, X, Y, scope, style='contour')

    plt.show()

if __name__=="__main__":

    X_raw, Y = load_data("Data_12_moon_100.csv")
    degree = [2,3,4,5,6,7,8,9]
    coef0 = [1,1,1,1,1,1,1,1]
    classification(X_raw, Y, degree, coef0)
    