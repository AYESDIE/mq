package org.mq.methods.logistic_regression;

import org.la4j.Matrix;
import org.mq.core.optimizers.sgd.SGD;

public class LogisticRegression
{
    public static void main(String[] args)
    {
        double[][] D = {{1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9},
                        {10, 11, 12}};

        Matrix dataset = Matrix.from2DArray(D);

        double[][] L = {{0, 0, 1 ,1}};
        Matrix labels = Matrix.from2DArray(L);

        // start
        LogisticRegressionFunction lrf = new LogisticRegressionFunction(dataset, labels, true);
        SGD sgd = new SGD(0.03, 100000, 1e-20, 1);

        Matrix parameters = lrf.initializeWeights();
        parameters = sgd.Optimize(lrf, parameters);

        System.out.println(LogisticRegressionFunction.Reciprocal(LogisticRegressionFunction.Exponential((dataset.multiply(parameters.slice(1, 0, parameters.rows(), parameters.columns())).add(parameters.get(0, 0))).multiply(-1)).add(1)));
    }
}
