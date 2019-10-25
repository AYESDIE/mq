package org.mq.core.optimizers;

import org.junit.Assert;
import org.junit.Test;
import org.la4j.Matrix;
import org.mq.core.optimizers.gradient_descent.GradientDescent;
import org.mq.core.optimizers.test_functions.GradientDescentTestFunction;

public class GradientDescentTest
{
    @Test
    public void GradientDescentTest()
    {
        GradientDescentTestFunction function = new GradientDescentTestFunction();
        GradientDescent optimizer = new GradientDescent(0.01, 2000, 1e-20);

        double[][] A = {{1, 3, 2}};
        Matrix M = Matrix.from2DArray(A);

        M = optimizer.Optimize(function, M);
        Assert.assertEquals(M.get(0, 0), 0.000, 1e-5);
        Assert.assertEquals(M.get(0, 1), 0.000, 1e-5);
        Assert.assertEquals(M.get(0, 2), 0.000, 1e-5);
    }
}
