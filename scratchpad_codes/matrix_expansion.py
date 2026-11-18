import numpy as np

# Let's test with a symmetric matrix M
a = np.array([1, 2])
b = np.array([3, 4])
M_sym = np.array([[2, 1], [1, 3]])  # symmetric

lhs = (a - b).T @ M_sym @ (a - b)
rhs = a.T @ M_sym @ a - 2 * b.T @ M_sym @ a + b.T @ M_sym @ b
print(f"Symmetric M: LHS = {lhs}, RHS = {rhs}")

# Let's test with a non-symmetric matrix M
M_nonsym = np.array([[2, 5], [1, 3]])  # not symmetric
lhs_ns = (a - b).T @ M_nonsym @ (a - b)
rhs_ns = a.T @ M_nonsym @ a - 2 * b.T @ M_nonsym @ a + b.T @ M_nonsym @ b
print(f"Non-symmetric M: LHS = {lhs_ns}, RHS = {rhs_ns}")
