package org.mq.core.optimizers.test_functions;

import org.la4j.Matrix;
import org.mq.core.optimizers.functions.DifferentiableFunction;
import org.mq.core.optimizers.sgd.SGD;

public class SGDTestFunction extends DifferentiableFunction
{
    public SGDTestFunction() {}

    public int numFunctions()
    {
        return 1;
    }

    public double Evaluate(Matrix iterate,
                           int id,
                           int batchSize)
    {
        Matrix temp = iterate.slice(0, id, iterate.rows(), id + batchSize);
        temp = temp.transpose().multiply(temp);
        return temp.get(0,0);
    }

    public Matrix Gradient(Matrix iterate,
                           int id,
                           int batchSize)
    {
        Matrix gradient = iterate.multiply(2);
        return gradient;
    }
}
