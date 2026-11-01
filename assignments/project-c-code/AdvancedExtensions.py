import math

from Core import Matrix, Vector

EPSILON = 1e-6


def SquareSubMatrix(mat: Matrix, r: int, c: int) -> Matrix:
    row_count = mat.M_Rows
    col_count = mat.N_Cols
    sub_mat = Matrix(row_count-1, col_count-1)
    for x in range(row_count):
        for y in range(col_count):
            if x != r and y != c:
                if x > r and y > c: 
                    sub_mat[x-1, y-1] = mat[x ,y] 
                elif x > r: 
                    sub_mat[x-1, y] = mat[x ,y] 
                elif y > c: 
                    sub_mat[x, y-1] = mat[x ,y] 
                else: 
                    sub_mat[x, y] = mat[x ,y] 

    return sub_mat


def Determinant(mat: Matrix) -> float:
    row_count = mat.M_Rows
    col_count = mat.N_Cols
    if col_count == 2:
        return mat[0, 0]*mat[1, 1] - mat[0, 1]*mat[1, 0]
    det_result = 0
    for indx in range(col_count): 
        sub_mat = SquareSubMatrix(mat, 0, indx)
        det_result += ((-1)**indx)*mat[0, indx]*Determinant(sub_mat)

    return det_result


def VectorNorm(vec: Vector) -> float:
    norm_val = 0.0
    for indx in range(len(vec)):
        norm_val += vec[indx]**2
    return math.sqrt(norm_val)


def SetColumn(mat: Matrix, vec: Vector, col: int) -> Matrix:
    row_count = mat.M_Rows
    col_count = mat.N_Cols
    new_mat = Matrix(row_count, col_count)
    if len(vec) != mat.M_Rows: 
        return False 
    else: 
        for indx in range(row_count): 
            for col_indx in range(col_count): 
                if col_indx != col: 
                    new_mat[indx, col_indx] = mat[indx, col_indx]
                else: 
                    new_mat[indx, col_indx] = vec[indx]

    return new_mat

def GramSchmidt(mat: Matrix) -> tuple:
    row_count = mat.M_Rows
    col_count = mat.N_Cols

    Q_mat = Matrix(row_count, col_count)
    R_mat = Matrix(col_count, col_count)

    def getColumn(matrix, col_indx):
        return [matrix[i, col_indx] for i in range(row_count)]

    def setColumn(matrix, vec, col_indx):
        for i in range(row_count):
            matrix[i, col_indx] = vec[i]
        return matrix

    def dotProduct(vec1, vec2):
        return sum(i*j for i, j in zip(vec1, vec2))

    def scalarMultiply(vec, scalar):
        return [scalar*i for i in vec]

    def subtractVectors(vec1, vec2):
        return [i-j for i, j in zip(vec1, vec2)]

    def normalize(vec):
        vec_norm = math.sqrt(dotProduct(vec, vec))
        return scalarMultiply(vec, 1/vec_norm)

    for i in range(col_count):
        vec = getColumn(mat, i)
        for j in range(i):
            q_vec = getColumn(Q_mat, j)
            R_mat[j, i] = dotProduct(q_vec, vec)
            vec = subtractVectors(vec, scalarMultiply(q_vec, R_mat[j, i]))
        R_mat[i, i] = math.sqrt(dotProduct(vec, vec))
        q_vec = scalarMultiply(vec, 1/R_mat[i, i])
        Q_mat = setColumn(Q_mat, q_vec, i)

    return Q_mat, R_mat
