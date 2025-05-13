#include <omp.h>

void square_matrix(double* mat, double* result, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += mat[i * n + k] * mat[k * n + j];
            }
            result[i * n + j] = sum;
        }
    }
}
