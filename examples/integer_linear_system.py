import numpy as np

from hsnf.integer_system import solve_integer_linear_system

if __name__ == "__main__":
    A = np.array([[6, 4, 10], [-1, 1, -5]])
    b = np.array([4, 11])
    basis, x_special = solve_integer_linear_system(A, b)
    print("special solution")
    print(x_special)
    print("general solution basis")
    print(basis)
