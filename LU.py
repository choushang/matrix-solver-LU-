# python version 3.10.6

# 载入工具包
import numpy as np
import pandas as pd
import sys


def check_up(a, b):
    # 判断矩阵是否有解，以及是否有唯一解——矩阵秩判断
    A = a
    df_A = pd.DataFrame(a)
    df_b = pd.DataFrame(b)
    df = pd.concat([df_A, df_b], axis=1)  # 表格拼接

    M = np.array(df)
    r = np.linalg.matrix_rank(a)  # 求秩
    r_m = np.linalg.matrix_rank(M)  # 求增广矩阵的秩
    n = a.shape[0]

    if r != n or r != r_m or r_m != n:
        print('矩阵不存在唯一解，矩阵无法LU分解')
        sys.exit(0)
        pass
    else:
        print(a)
        print(b)
        print('行列式的值为{}'.format(np.linalg.det(a)))  # 返回矩阵的行列式)
        k = np.linalg.det(a)
        if k == 0:
            print('矩阵奇异，无法LU分解')
            sys.exit(0)
            pass
        pass
    return A


def find_max(a, j):
    # 寻找矩阵a中j列最大元素的所在行数：max_row
    max_number = np.fabs(a[j][j])
    max_row = j

    for i in list(range(j, a.shape[0])):
        if np.fabs(a[i][j]) > np.fabs(max_number):
            max_number = a[i][j]
            max_row = i
            pass
        pass
    return max_row


def change_row_a(a, max_row, i):
    # A矩阵行交换
    for j in (list(range(i, a.shape[1]))):
        temp = a[i][j]
        a[i][j] = a[max_row][j]
        a[max_row][j] = temp
        pass


def change_row_b(b, max_row, i):
    # b矩阵行交换
    temp = b[i][0]
    b[i][0] = b[max_row][0]
    b[max_row][0] = temp


def gaussian_elimination(a, b):
    # 使用高斯消元法，对A矩阵进行消元
    for i in list(range(0, a.shape[0])):  # 逐行处理矩阵数据
        max_row = find_max(a, i)  # 在当前系数矩阵的(i, i)子矩阵中获取第一列中最大元素所在行数，准备进行交换
        change_row_a(a, max_row, i)  # 将最大元素所在行数交换至(i, i)子矩阵的第一行
        change_row_b(b, max_row, i)  # 同时交换值矩阵b的元素，使方程的系数与值保持对应关系

        for k in list(range(i + 1, a.shape[0])):
            h = (-a[k][i] / a[i][i])  # 计算接下来消元需要用到的比例因子
            pass
            for j in list(range(i, a.shape[1])):
                a[k][j] = h * a[i][j] + a[k][j]  # 消元
                pass
            b[k][0] = h * b[i][0] + b[k][0]
        pass


def solve(a, b):
    # 对增广矩阵进行求解
    x = np.zeros((a.shape[0], 1))
    x[-1] = b[-1] / a[-1][-1]

    for i in list(range(a.shape[0] - 2, -1, -1)):  # 从上三角方程最后一行开始解方程，倒着计算
        sum_a = 0
        for j in list(range(i + 1, a.shape[0])):
            sum_a += a[i][j] * x[j]
        x[i] = (b[i] - sum_a) / a[i][i]
        pass
    print('方程的解向量是 \n{}'.format(x))
    return x


def answer(a, b):
    # 求解函数
    k = a
    check_up(a, b)
    gaussian_elimination(a, b)
    solve(a, b)


题目序号 = int(input('请键入第几题的序号'))
if 题目序号 == 1:
    # 输入矩阵数据时，注意整数数据要多打小数点，否则会识别为int类型数据,而非float。
    A = np.array([[3.01, 6.03, 1.99], [1.27, 4.16, -1.23], [0.987, -4.81, 9.34]])
    b = np.array([[1.], [1.], [1.]])
    pass
elif 题目序号 == 2:
    A = np.array([[3.01, 6.03, 1.99], [1.27, 4.16, -1.23], [0.99, -4.81, 9.34]])
    b = np.array([[1.], [1.], [1.]])
    pass
elif 题目序号 == 3:
    # 使用课本p148页例4进行检验
    A = np.array([[0.001, 2.000, 3.000], [-1.000, 3.712, 4.623], [-2.000, 1.072, 5.643]])
    b = np.array([[1.], [2.], [3.]])
    pass

answer(A, b)
