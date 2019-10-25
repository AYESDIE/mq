package org.mq.core.optimizers.test_functions;

import org.la4j.Matrix;
import org.mq.core.optimizers.functions.DifferentiableFunction;
import org.mq.core.optimizers.gradient_descent.GradientDescent;

public class GradientDescentTestFunction extends DifferentiableFunction
{
    public GradientDescentTestFunction() {}

    public double Evaluate(Matrix iterate)
    {
        Matrix Q = iterate.multiply(iterate.transpose());
        return Q.get(0, 0);
    }

    public Matrix Gradient(Matrix iterate)
    {
        return iterate.multiply(2);
    }

}
