package org.mq.core.math;

import org.la4j.Matrix;
import org.la4j.Vector;

public class Math
{
    public static Matrix Normalize(Matrix M)
    {
        double[][] R = new double[M.rows()][M.columns()];

        for (int j = 0; j < M.columns(); j++)
        {
            Vector V = M.getColumn(j);

            double min = V.min();
            double max = V.max();
            double mean = V.sum() / V.length();

            if (min - max != 0)
            {
                for (int i = 0; i < M.rows(); i++)
                {
                    R[i][j] = (M.get(i, j) - mean) / (max - min);
                }
            }
            else
            {
                for (int i = 0; i < M.rows(); i++)
                {
                    R[i][j] = M.get(i, j);
                }
            }
        }

        return Matrix.from2DArray(R);
    }


    public static Matrix Reciprocal(Matrix M)
    {
        double[][] R = new double[M.rows()][M.columns()];

        for (int i = 0; i < M.rows(); i++)
        {
            for (int j = 0; j < M.columns(); j++)
            {
                R[i][j] = 1 / M.get(i, j);
            }
        }

        return Matrix.from2DArray(R);
    }

    public static Matrix Exponential(Matrix M)
    {
        double[][] R = new double[M.rows()][M.columns()];

        for (int i = 0; i < M.rows(); i++)
        {
            for (int j = 0; j < M.columns(); j++)
            {
                R[i][j] = java.lang.Math.exp(M.get(i, j));
            }
        }

        return Matrix.from2DArray(R);
    }

    public static Matrix Log(Matrix M)
    {
        double[][] R = new double[M.rows()][M.columns()];

        for (int i = 0; i < M.rows(); i++)
        {
            for (int j = 0; j < M.columns(); j++)
            {
                R[i][j] = java.lang.Math.log(M.get(i, j));
            }
        }

        return Matrix.from2DArray(R);
    }

    public static Matrix SchurProduct(Matrix M1, Matrix M2)
    {
        double[][] R = new double[M1.rows()][M1.columns()];

        for (int i = 0; i < M1.rows(); i++)
        {
            for (int j = 0; j < M1.columns(); j++)
            {
                R[i][j] = M1.get(i, j) * M2.get(i, j);
            }
        }

        return Matrix.from2DArray(R);
    }
}
