package org.mq.core.optimizers.functions;

import org.la4j.Matrix;

public class DifferentiableFunction
{
    public double Evaluate(Matrix iterate)
    {
        return 0;
    }

    public Matrix Gradient(Matrix iterate)
    {
        return iterate;
    }
}
