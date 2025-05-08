# -*- coding: utf-8 -*-
"""
Created on Wed May  7 13:58:52 2025

@author: carlo
"""
import math


class SAT2_Solver:
        
    def mult_matrix_or_and(self, A, B):  ## TODO CHEQUEAR Y TMBN CHEQUEAR DOC
        """
        Multiply two binary square matrices A and B using logical AND (for multiplication)
        and logical OR (for summation), returning a Python list of lists with 0s and 1s.
    
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
    
    def mult_matrix_max_plus(self, A, B): ## TODO CHEQUEAR Y TMBN CHEQUEAR DOC
        """
        Multiply two square matrices A and B using + (for multiplication)
        and max (for summation), returning a Python list of lists with numbers.
    
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
        n_negated_variable = n_variables + n_variable
        return ((A1[n_variable][n_variables] == 1) and (A1[n_variables][n_variable] == 1))
            
        
        
    def init_transitive_closure(self, implications, n_variables):
        """
        Initializes the transitive closure matrix of a directed graph.
    
        Parameters:
        - implications: List of tuples (u, v) where u and v are ints in range n_variables.
                   Each implication defines an edge in the graph.
        - n_variables: Total number of variables.
    
        Returns:
        - transitive_closure: list of lists with 0s and 1s, size (n_variables * 2) x (n_variables * 2).
        """
        n_nodes = n_variables * 2
    
        # Initialize matrix with 0s
        transitive_closure = [[0 for _ in range(n_nodes)] for _ in range(n_nodes)]
        
        # Set diagonal entries to 1 
        for i in range(n_nodes):
            transitive_closure[i][i] = 1
        
        # Add edges from implications
        for u, v in implications:
            ### TODO MIRAR SI ESTO VA ASÍ BIEN, EL FOR Y SI NOS VIENEN LAS IMPLICACIONES NEGATIVAS Y POSITIVAS EN implications
            transitive_closure[u][v] = 1
    
        return transitive_closure
    
    def init_longest_paths(self, implications, A1):
        """
        Initializes matrix B from the transitive closure matrix A1.
    
        Parameters:
        - implications: List of tuples (u, v) indicating directed edges between variables.
        - A1: Transitive closure matrix (list of lists of 0s and 1s), size (n_variables * 2).
    
        Returns:
        - B: List of lists (same size as A1) with:
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
        adj_matrix = self.init_transitive_closure(implications, n_variables)
        A1 = adj_matrix ## TODO CHEQUEAR SI EL +1 DE ABAJO ESTÁ BIEN
        for _ in range(int(math.log2(n_variables * 2)) + 1): # 2*n_variables because we take the negated variables as well
            A1 = self.mult_matrix_or_and(A1, A1)
        for n_variable in range(n_variables):
            if self.__negated_same_str_component(A1, n_variables, n_variable):
                return False, A1
        return True, A1
    
    def get_truth_assigment(self, implications, n_variables):
        solvable, A1 = self.is_solvable(implications, n_variables)
        if not solvable:
            return None
        B1 = self.init_longest_paths(implications, A1) ### TODO VER SI implications SON NECESARIAS, CREO Q NO
        for _ in range(int(math.log2(n_variables * 2)) + 1):## TODO CHEQUEAR SI EL +1  ESTÁ BIEN
            B1 = self.mult_matrix_max_plus(B1, B1)
        results = []
        for n_variable in range(n_variables):
            # Each variable is true ifff its longest path to a sink is lower 
            # than the longest path to a sink of the negated variable. 
            results.append(max(B1[n_variable]) <= max(B1[n_variables + n_variable]))
        info = {"A1" : A1, "B1" : B1}
        return results, info
                