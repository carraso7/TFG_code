# -*- coding: utf-8 -*-
"""
Created on Wed May  7 13:58:52 2025

@author: carlo
"""
import math
 
def print_B_matrix(matrix, name):
    """
    Debugging function to visualize matrices

    Returns
    -------
    None.

    """
    n = len(matrix)
    print(f"\n{name}:")

    # Print column headers
    header = "   " + ' '.join(f"{j:2}" for j in range(n))
    print(header)

    # Print each row with its index
    for i, row in enumerate(matrix):
        row_str = f"{i:2}|" + ' '.join(f"{'-' if elem == float('-inf') else f'{elem:2}'}" for elem in row)
        print(row_str)


class SAT2_solver:
        
    def mult_matrix_or_and(self, A, B):  
        """
        Multiply two binary square matrices A and B using logical AND 
        (for multiplication) and logical OR (for summation), returning 
        a Python list of lists with 0s and 1s. Sequential implementation.
    
        Raises:
        - ValueError if A and B are not square or not the same size.
        """
        # Check A is square
        if not all(len(row) == len(A) for row in A):
            raise ValueError("Matrix A is not square.")
    
        # Check B is square
        if not all(len(row) == len(B) for row in B):
            raise ValueError("Matrix B is not square.")
        
        # Check same size
        if len(A) != len(B):
            raise ValueError("Matrices A and B must be the same size.")
    
        n = len(A)  # size of the square matrix
    
        result = []
        for i in range(n):
            row_result = []
            for j in range(n):
                # Build the list of booleans for this (i, j)
                bool_list = [(A[i][k] and B[k][j]) for k in range(n)]
                
                # Reduce with OR
                value = any(bool_list)
                
                row_result.append(1 if value else 0)
            result.append(row_result)
        
        return result
    
    def __mult_matrix_max_plus(self, A, B): 
        """
        Multiply two square matrices A and B using + (for multiplication)
        and max (for summation), returning a Python list of lists with numbers.
        Sequential implementation.
    
        Raises:
        - ValueError if A and B are not square or not the same size.
        """
        # Check A is square
        if not all(len(row) == len(A) for row in A):
            raise ValueError("Matrix A is not square.")
    
        # Check B is square
        if not all(len(row) == len(B) for row in B):
            raise ValueError("Matrix B is not square.")
        
        # Check same size
        if len(A) != len(B):
            raise ValueError("Matrices A and B must be the same size.")
    
        n = len(A)
    
        result = []
        for i in range(n):
            row_result = []
            for j in range(n):
                # Build the list of numbers for this (i, j)
                val_list = [A[i][k] + B[k][j] for k in range(n)]
                
                # Reduce with max
                value = max(val_list)
                
                row_result.append(value)
            result.append(row_result)
        
        return result

    def __negated_same_str_component(self, A1, n_variables, n_variable):
        """
        Returns true if the positive and negative variable assigned to the 
        integer value `n_variable` are in the same strong component according
        to matrix A1 that indicates the reachable nodes from each node in a 
        directed graph (transitive closure).


        Returns
        -------
        bool
            true if both variables are in the same str. comp., false otherwise.

        """
        n_negated_variable = n_variables + n_variable
        return ((A1[n_variable][n_negated_variable] == 1) and (A1[n_negated_variable][n_variable] == 1))
            
        
        
    def __init_transitive_closure(self, implications, n_variables):
        """
        Initializes the transitive closure matrix of a directed graph.

        Parameters
        ----------
        implications : List of tuples of integers
            List of tuples (u, v) where u and v are ints in range 
            2 * n_variables.
        n_variables : int
            number of variables of the 2CNF problem.

        Returns
        -------
        transitive_closure : (2 * n_variables) * (2 * n_variables) list matrix
            Initialization matrix for calculating the transitive closure. The 
            matrix has a 1 in the positions where there is an edge between the
            row node and the column node and 0s in the other positions.

        """
        n_nodes = n_variables * 2
    
        # Initialize matrix with 0s
        transitive_closure = [[0 for _ in range(n_nodes)] for _ in range(n_nodes)]
        
        # Set diagonal entries to 1 
        for i in range(n_nodes):
            transitive_closure[i][i] = 1
        
        # Add edges from implications
        for u, v in implications:
            transitive_closure[u][v] = 1
    
        return transitive_closure
    
    def __init_longest_paths(self, A1):
        """
        Initializes matrix B ( longest path to sink matrix ) from the 
        transitive closure matrix A1.        

        Parameters
        ----------
        A1 : (2 * n_variables) * (2 * n_variables) list matrix
            Matrix representing transitive closure of the graph.

        Returns
        -------
        B : (2 * n_variables) * (2 * n_variables) list matrix
            Initialization longest path to sink matrix with:
            - 0 in the diagonal,
            - 0 if y and z are in the same strong component,
            - 1 if there's a connection from z's component to y's component,
            - -inf otherwise.

        """
        n = len(A1)
        
        # Initialize B matrix
        B = [[float('-inf') for _ in range(n)] for _ in range(n)]
        
        # Fill B based on the new clarified conditions
        for y in range(n):
            for z in range(n):
                if y == z:
                    # Always set diagonal to 0
                    B[y][z] = 0
                elif A1[y][z] == 1 and A1[z][y] == 1:
                    # Same strong component (non-diagonal)
                    B[y][z] = 0
                elif A1[y][z] == 1 and A1[z][y] == 0:
                    # Implication between two strong components (only one way) 
                    B[y][z] = 1
        return B

    
    def is_solvable(self, implications, n_variables):
        """
        Determines whether a given 2-SAT formula is satisfiable 
    
        Parameters
        ----------
        implications : list of tuples with integers.
            Implications of the 2CNF problem represented as tuples with index 
            in range(2*n_variables). Each integer of the tuple represents a 
            variable of the 2CNF problem or a negated variable of the problem. 
            The positive variables are indexed in range(0, n_variables) and the
            corresponding negative variables are indexed in 
            range(n_variables, 2*n_variables). The negated variable of a 
            variable indexed with integer n is indexed with integer 
            n + n_variables
        n_variables : int
            Number of base variables of the 2CNF problem (before negation).
    
        Returns
        -------
        bool
            True if satisfiable, False otherwise.
        info: dict
            dictionary with two keys:
                - 'A1': A1 matrix obtained in the process
                - 'B1': B1 matrix obtained in the process
                
        """
        adj_matrix = self.__init_transitive_closure(implications, n_variables)
        A1 = adj_matrix 
        for _ in range(int(math.log2(n_variables * 2)) + 1): # 2*n_variables because we take the negated variables as well
            A1 = self.mult_matrix_or_and(A1, A1)
        for n_variable in range(n_variables):
            if self.__negated_same_str_component(A1, n_variables, n_variable):
                return False, A1
        return True, A1

    def get_truth_assigment(self, implications, n_variables): 
        """
        Solves the 2-SAT problem by computing a satisfying truth assignment, 
        if possible.
    
        Parameters
        ----------
        implications : list of tuples with integers.
            Implications of the 2CNF problem represented as tuples with index 
            in range(2*n_variables). Each integer of the tuple represents a 
            variable of the 2CNF problem or a negated variable of the problem. 
            The positive variables are indexed in range(0, n_variables) and the
            corresponding negative variables are indexed in 
            range(n_variables, 2*n_variables). The negated variable of a 
            variable indexed with integer n is indexed with integer 
            n + n_variables
        n_variables : int
            Number of base variables of the 2CNF problem (before negation).
    
        Returns
        -------
        results: list of booleans
            A list of boolean values assigning a truth value to each variable 
            if solvable; None otherwise.
        info: dict
            A dictionary with:
                - 'A1': Transitive closure matrix,
                - 'B1': Max-plus matrix of longest paths,
                
        """
        solvable, A1 = self.is_solvable(implications, n_variables)
        if not solvable:    
            info = {"A1" : A1, "B1" : None}
            return None, info
        B1 = self.__init_longest_paths(A1) 
        for _ in range(int(math.log2(n_variables * 2)) + 1):
            B1 = self.__mult_matrix_max_plus(B1, B1)
            
        # Construct satisfying assignment.
        results = []
        
        def compute_s_vector(A1): 
            n = len(A1)
            s = [None] * n
            for y in range(n):
                same_component = [z for z in range(n) if A1[y][z] == 1 and A1[z][y] == 1]
                s[y] = min(same_component)
            return s    
        
        s = compute_s_vector(A1)
        for n_variable in range(n_variables):
            fy = max(B1[n_variable])
            f_not_y = max(B1[n_variables + n_variable])
        
            if fy < f_not_y:
                results.append(True)
            elif fy > f_not_y:
                results.append(False)
            else:
                # tie: choose using s(y)
                s_y = s[n_variable]
                s_not_y = s[n_variables + n_variable]
                results.append(s_y < s_not_y)
        info = {"A1" : A1, "B1" : B1}
        return results, info
    