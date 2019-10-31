package org.mq.methods.logistic_regression;

import org.la4j.Matrix;
import org.mq.core.math.Math;
import org.mq.core.optimizers.Optimizers;
import org.mq.core.optimizers.sgd.SGD;

public class LogisticRegression
{
    public LogisticRegression(Matrix dataset,
                              Matrix labels)
    {
        fitIntercept = false;
        LogisticRegressionFunction lrf = new LogisticRegressionFunction(dataset, labels);

        SGD sgd = new SGD();

        parameters = lrf.initializeWeights();
        parameters = sgd.Optimize(lrf, parameters);
    }

    public <OptimizerType extends Optimizers> LogisticRegression(Matrix dataset,
                                                                 Matrix labels,
                                                                 OptimizerType optimizer)
    {
        fitIntercept = false;
        LogisticRegressionFunction lrf = new LogisticRegressionFunction(dataset, labels);
        parameters = lrf.initializeWeights();
        parameters = optimizer.Optimize(lrf, parameters);
    }

    public <OptimizerType extends Optimizers> LogisticRegression(Matrix dataset,
                                                                 Matrix labels,
                                                                 OptimizerType optimizer,
                                                                 boolean fitIntercept)
    {
        this.fitIntercept = fitIntercept;
        LogisticRegressionFunction lrf = new LogisticRegressionFunction(dataset, labels, fitIntercept);
        parameters = lrf.initializeWeights();
        parameters = optimizer.Optimize(lrf, parameters);
    }

    public <OptimizerType extends Optimizers> LogisticRegression(Matrix dataset,
                                                                 Matrix labels,
                                                                 OptimizerType optimizer,
                                                                 boolean fitIntercept,
                                                                 double lambda)
    {
        this.fitIntercept = fitIntercept;
        LogisticRegressionFunction lrf = new LogisticRegressionFunction(dataset, labels, fitIntercept, lambda);
        parameters = lrf.initializeWeights();
        parameters = optimizer.Optimize(lrf, parameters);
    }

    public Matrix compute(Matrix testDataset)
    {
        Matrix sigmoid;
        if (fitIntercept)
        {
            double bias = parameters.get(0, 0);
            parameters = parameters.slice(1, 0, parameters.rows(), parameters.columns());
            sigmoid = Math.Reciprocal(Math.Exponential((testDataset.multiply(parameters).add(bias)).multiply(-1)).add(1)).transpose();
        }
        else
        {
            sigmoid = Math.Reciprocal(Math.Exponential(testDataset.multiply(parameters).multiply(-1)).add(1)).transpose();
        }

        return assignLabels(sigmoid);
    }

    private Matrix assignLabels(Matrix M)
    {
        double[][] R = new double[M.rows()][M.columns()];

        for (int i = 0; i < M.rows(); i++)
        {
            for (int j = 0; j < M.columns(); j++)
            {
                if (M.get(i, j) < 0.5)
                {
                    R[i][j] = 0;
                }
                else
                {
                    R[i][j] = 1;
                }
            }
        }

        return Matrix.from2DArray(R);
    }

    private Matrix parameters;
    private boolean fitIntercept;
}
