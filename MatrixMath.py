from typing import List


class MatrixMath:
    @staticmethod
    def add3d(matrixa: List[List[List[float]]], matrixb: List[List[List[float]]]) -> List[List[List[float]]]:
        result: List[List[List[float]]] = matrixa.copy()
        for x in range(len(matrixa)):
            for y in range(len(matrixa[x])):
                for z in range(len(matrixa[x][y])):
                    result[x][y][z] += matrixb[x][y][z]
        return result

    @staticmethod
    def mul_scalar3d(matrixa: List[List[List[float]]], scalar: float):
        result: List[List[List[float]]] = matrixa.copy()
        for x in range(len(matrixa)):
            for y in range(len(matrixa[x])):
                for z in range(len(matrixa[x][y])):
                    result[x][y][z] *= scalar
        return result
