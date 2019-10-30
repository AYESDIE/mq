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
}
