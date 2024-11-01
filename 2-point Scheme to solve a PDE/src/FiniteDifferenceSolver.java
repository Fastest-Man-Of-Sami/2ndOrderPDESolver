import java.util.Arrays;

public class FiniteDifferenceSolver {

    public static void main(String[] args) {
        System.out.printf("%-10s%-20s%-20s%n", "h", "||u_h - φ_h||_∞", "||u_h - φ_h||_∞ / h^2");

        // Loop over p = 1 to 14
        for (int p = 1; p <= 14; p++) {
            double h = 1.0 / Math.pow(2, p);
            int N = (int) (1 / h) - 1;  // Number of interior points
            double[] x = new double[N];

            // Generate grid points
            for (int i = 0; i < N; i++) {
                x[i] = h * (i + 1);
            }

            // Construct the coefficient matrix A and right-hand side b
            double[][] A = new double[N][N];
            double[] b = new double[N];

            for (int i = 0; i < N; i++) {
                A[i][i] = 2 + (h * h) * x[i] * x[i];  // Main diagonal
                if (i > 0) A[i][i - 1] = -1;         // Lower diagonal
                if (i < N - 1) A[i][i + 1] = -1;     // Upper diagonal

                // Right-hand side
                b[i] = (1 + 4 * x[i] + 2 * x[i] * x[i] - Math.pow(x[i], 4)) * Math.exp(x[i]) * h * h;
            }
            b[0] += 1;  // Adjust for the boundary condition at x = 0

            // Solve the linear system using LU decomposition
            double[] u_h = solveLU(A, b);

            // Compute the true solution and the error norms
            double errorInfNorm = 0;
            double[] phi_h = new double[N];

            for (int i = 0; i < N; i++) {
                phi_h[i] = (1 - x[i] * x[i]) * Math.exp(x[i]);
                errorInfNorm = Math.max(errorInfNorm, Math.abs(u_h[i] - phi_h[i]));
            }
            double errorH2Norm = errorInfNorm / (h * h);

            // Print the results
            System.out.printf("%-10.8f %-20.8e %-20.8e%n", h, errorInfNorm, errorH2Norm);
        }
}

    // LU Decomposition method to solve Ax = b
    public static double[] solveLU(double[][] A, double[] b) {
        int N = A.length;
        double[][] LU = new double[N][N];
        int[] pivot = new int[N];

        // Copy A to LU
        for (int i = 0; i < N; i++) {
            LU[i] = Arrays.copyOf(A[i], N);
        }

        // Perform LU decomposition with partial pivoting
        for (int i = 0; i < N; i++) {
            pivot[i] = i;
        }
        for (int k = 0; k < N; k++) {
            // Find the pivot
            double max = 0.0;
            int maxIndex = k;
            for (int i = k; i < N; i++) {
                double absVal = Math.abs(LU[i][k]);
                if (absVal > max) {
                    max = absVal;
                    maxIndex = i;
                }
            }
            if (max == 0) throw new RuntimeException("Matrix is singular.");

            // Swap rows
            int temp = pivot[k];
            pivot[k] = pivot[maxIndex];
            pivot[maxIndex] = temp;
            double[] tmpRow = LU[k];
            LU[k] = LU[maxIndex];
            LU[maxIndex] = tmpRow;

            // Perform elimination
            for (int i = k + 1; i < N; i++) {
                LU[i][k] /= LU[k][k];
                for (int j = k + 1; j < N; j++) {
                    LU[i][j] -= LU[i][k] * LU[k][j];
                }
            }
        }

        // Solve Ly = Pb
        double[] y = new double[N];
        for (int i = 0; i < N; i++) {
            y[i] = b[pivot[i]];
            for (int j = 0; j < i; j++) {
                y[i] -= LU[i][j] * y[j];
            }
        }

        // Solve Ux = y
        double[] x = new double[N];
        for (int i = N - 1; i >= 0; i--) {
            x[i] = y[i];
            for (int j = i + 1; j < N; j++) {
                x[i] -= LU[i][j] * x[j];
            }
            x[i] /= LU[i][i];
        }

        return x;
    }
}
