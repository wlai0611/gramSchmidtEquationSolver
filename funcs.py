import numpy as np
import math
import pickle
import json

def norm(vec):
    return np.dot(vec,vec)**0.5

def project(v1,v2):
    '''
    project v1 onto v2
    return how much to scale
    '''
    if np.allclose(v2,np.zeros(v2.shape[0])):
        return 0
    return np.dot(v1,v2)/np.dot(v2,v2)


#A = np.array([[1,4],[1,0],[0,0]],dtype=float)
#A = np.array([[1,1,1],[1,-1,1],[-1,-1,1]],dtype=float)

def gram_schmidt(A):
    '''
    A must be of type float
    '''
    R = np.zeros((A.shape[1],A.shape[1]))
    Q = np.zeros(A.shape)
    vec1norm = norm(A[:,0])
    Q[:,0] = A[:,0]/vec1norm
    R[0,0] = vec1norm

    for col in range(1,A.shape[1]):
        vector = A[:,col]
        orthogonal_component = vector.copy()
        for prior_col in range(col):
            q                               =  Q[:,prior_col]
            multiples_of_q                  =  project(vector,q)
            R[prior_col, col]               =  multiples_of_q
            projection_onto_prior_q         =  multiples_of_q * q
            orthogonal_component            = orthogonal_component - projection_onto_prior_q
            
        orthogonal_component_length     = norm(orthogonal_component)
        if math.isclose(orthogonal_component_length, 0, abs_tol=1e-15):
            orthogonal_component_length = 0
            normalized_orthogonal_component = np.zeros(A.shape[0]) #avoid division by 0 => nans
        else:
            normalized_orthogonal_component = orthogonal_component/orthogonal_component_length
        R[col,col]                      = orthogonal_component_length
        Q[:,col]                        = normalized_orthogonal_component
    return Q,R

def back_substitution(upper_triangular, response_var):
    x       = np.zeros(upper_triangular.shape[0])
    x[-1]   = response_var[-1]/upper_triangular[-1,-1] #get the last variable's value

    for i in range(response_var.shape[0]-2,-1,-1): #start from the second to last variable
        prior_coefficients = x[:i:-1]
        substitution       = prior_coefficients@upper_triangular[i,:i:-1]
        pivot              = upper_triangular[i,i]
        solution           = (response_var[i]-substitution)/pivot
        if math.isclose(solution, 0, abs_tol=1e-15):
            x[i] = 0
        else:
            x[i] = solution

    return x
'''
A = np.array([[1,1,1],[1,-1,1],[-1,-1,1]],dtype=float)
Q,R  = gram_schmidt(A)
b    = np.array([6., 2., 0.]) # inside of the plane formed by A[:,0] and A[:,1]

b_rotated    = Q.T @ b
b_projection = Q @ Q.T @ b


if np.allclose(b,b_projection):
    print("")
'''

def is_in_column_space(Q, b):
    b_project_on_Q = Q @ Q.T @ b
    return np.allclose(b_project_on_Q, b)

def calculate_determinant(R):
    determinant = 1
    for col in range(R.shape[1]):
        determinant = determinant * R[col,col]
    return determinant

def solve(A, b):
    Q,R  = gram_schmidt(A)
    if is_in_column_space(Q, b):
        determinant = calculate_determinant(R)
        if math.isclose(determinant,0):
            return np.inf
        b_rotated    = Q.T @ b
        x = back_substitution( R, b_rotated )
        return x
    else:
        return None

def json_to_numpy(data):
    dct  = json.loads(data)
    A    = np.array(dct['independentVars'])
    b    = np.array(dct['dependentVars'])
    return A,b

A=np.array([[1,2],[3,2],[0,0]],dtype=float)
b=np.array([5,7,0],dtype=float)
solve(A,b)

solve(np.array([[4,2,1],[2,1,-1]],dtype=float),np.array([1,1],dtype=float),)
A = np.array([[1,2,1],[1,2,2]])
b = np.array([2,3])
print(solve(A,b))
A = np.array([[1,2,0],[0,0,1]])
b = np.array([1,1])
print(f"Expected Infinity got {str(solve(A,b))}")
A = np.array([[1,2,3],[0,0,0]])
b = np.array([1,1])
print(f"Expected None got {str(solve(A,b))}")
A = np.array([[1,2,3],[4,5,6],[10,2,-3]])
b = np.array([25, 67, 55])
print(f"expected: 4,9,1 got {str(solve(A,b))}")
A = np.array([[1,2,3],[4,5,6],[10,2,-3],[1,4,8]])
b = np.array([202.02, 472.05, 172.02, 482.04]) 
print(f"expected: 34,0.01,56 got {str(solve(A,b))}")

A = np.array([[1,2,3],[4,5,6],[10,2,-3],[1,4,8]])
b = np.array([202.02, 472.05, 172.02, 482.04]) 
print(f"expected: 34,0.01,56 got {str(solve(A,b))}")

np.random.seed(123)
A = np.random.randint(1,100,(2,8))
np.random.seed(123)
x = np.random.randint(1,100,(A.shape[1]))
b = A@x
print(f"expected: {str(x)} got {str(solve(A,b))}")

#singular case

