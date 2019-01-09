from typing import List


class MatrixMath:
    @staticmethod
    def add3d(matrixa: List[List[List[float]]], matrixb: List[List[List[float]]]) -> List[List[List[float]]]:
        """Adds corresponding elements of two 3-dimensional lists of the same shape."""
        result: List[List[List[float]]] = matrixa.copy()
        for x in range(len(matrixa)):
            for y in range(len(matrixa[x])):
                for z in range(len(matrixa[x][y])):
                    result[x][y][z] += matrixb[x][y][z]
        return result

    @staticmethod
    def mul_scalar3d(matrixa: List[List[List[float]]], scalar: float):
        """Multiplies all elements of 3-dimensional list by given scalar."""
        result: List[List[List[float]]] = matrixa.copy()
        for x in range(len(matrixa)):
            for y in range(len(matrixa[x])):
                for z in range(len(matrixa[x][y])):
                    result[x][y][z] *= scalar
        return result
