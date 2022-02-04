
import numpy as np

# 创建随机矩阵
M1 = [[0, 0], [0.5, 0.5]]
M2 = [[0.3, 0.7], [0.7, 0.3]]
M3 = [[0.5, 0.5], [0.6, 0.4]]
M4 = [[0, 1], [0, 1]]
M = [M1, M2, M3, M4]
print(M)

# 生成路径
path = [2]
for i in range(1, 4):
    paths = []
    for _, r in enumerate(path):
        temp = np.transpose(r)
        paths.append(np.append(temp, 1))
        paths.append(np.append(temp, 2))
    path = paths.copy()

path = [np.append(r, 2) for _, r in enumerate(path)]
print(path)

# 计算概率

pr = []
for _, row in enumerate(path):
    p = 1
    for i in range(len(row) - 1):
        a = row[i]
        b = row[i + 1]
        p *= M[i][a - 1][b - 1]
    pr.append((row.tolist(), p))
pr = sorted(pr, key=lambda x: x[1], reverse=True)
print(pr)

# 打印结果
print("以start=2为起点stop=2为终点的所有路径的状态序列y的概率为：")
for path, p in pr:
    print("    路径为：" + "->".join([str(x) for x in path]), end=" ")
    print("概率为：" + str(p))
print("概率[" + str(pr[0][1]) + "]最大的状态序列为:",
      "->".join([str(x) for x in pr[0][0]]))