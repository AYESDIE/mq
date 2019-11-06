package org.mq.methods.logistic_regression;

import org.la4j.Matrix;
import org.la4j.Vector;
import org.mq.core.math.Math;
import org.mq.core.optimizers.functions.DifferentiableFunction;

public class LogisticRegressionFunction extends DifferentiableFunction
{
    public LogisticRegressionFunction(Matrix dataset,
                                      Matrix labels)
    {
        this.dataset = dataset;
        this.labels = labels;
        this.fitIntercept = false;
        this.lambda = 0;
    }

    public LogisticRegressionFunction(Matrix dataset,
                                      Matrix labels,
                                      boolean fitIntercept)
    {
        this.dataset = dataset;
        this.labels = labels;
        this.fitIntercept = fitIntercept;
        this.lambda = 0;
    }

    public LogisticRegressionFunction(Matrix dataset,
                                      Matrix labels,
                                      boolean fitIntercept,
                                      double lambda)
    {
        this.dataset = dataset;
        this.labels = labels;
        this.fitIntercept = fitIntercept;
        this.lambda = lambda;
    }

    public double Evaluate(Matrix iterate)
    {
        Matrix sigmoid;
        if (fitIntercept)
        {
            double bias = iterate.get(0, 0);
            iterate = iterate.slice(1, 0, iterate.rows(), iterate.columns());
            sigmoid = Math.Reciprocal(Math.Exponential((dataset.multiply(iterate).add(bias)).multiply(-1)).add(1));
        }
        else
        {
            sigmoid = Math.Reciprocal(Math.Exponential(dataset.multiply(iterate).multiply(-1)).add(1));
        }

        Matrix error1 = Math.SchurProduct(labels.transpose(), Math.Log(sigmoid)).multiply(-1);
        Matrix error2 = Math.SchurProduct(labels.transpose().multiply(-1).add(1), Math.Log(sigmoid.multiply(-1).add(1))).multiply(-1);
        Matrix error = error1.add(error2);
        double loss = error.divide(numFunctions()).sum();

        double reg = iterate.transpose().multiplyByItsTranspose().multiply(lambda / (2 * numFunctions())).get(0, 0);

        return loss + reg;
    }

    public double Evaluate(Matrix iterate,
                           int id,
                           int batchSize)
    {
        Matrix sigmoid;
        int lastId = batchSize + id;

        if (fitIntercept)
        {
            double bias = iterate.get(0, 0);
            iterate = iterate.slice(1, 0, iterate.rows(), iterate.columns());
            sigmoid = Math.Reciprocal(Math.Exponential((dataset.slice(id, 0, lastId, dataset.columns()).multiply(iterate).add(bias)).multiply(-1)).add(1));
        }
        else
        {
            sigmoid = Math.Reciprocal(Math.Exponential(dataset.slice(id, 0, lastId, dataset.columns()).multiply(iterate).multiply(-1)).add(1));
        }

        Matrix error1 = Math.SchurProduct(labels.slice(0, id, 1, lastId).transpose(), Math.Log(sigmoid)).multiply(-1);
        Matrix error2 = Math.SchurProduct(labels.slice(0, id, 1, lastId).transpose().multiply(-1).add(1), Math.Log(sigmoid.multiply(-1).add(1))).multiply(-1);
        Matrix error = error1.add(error2);
        double loss = error.divide(batchSize).sum();

        double reg = iterate.transpose().multiplyByItsTranspose().multiply(lambda / (2 * numFunctions())).get(0, 0);

        return loss + reg;
    }

    public Matrix Gradient(Matrix iterate)
    {
        Matrix sigmoid;

        if (fitIntercept)
        {
            double bias = iterate.get(0, 0);
            Matrix parameters = iterate.slice(1, 0, iterate.rows(), iterate.columns());
            sigmoid = Math.Reciprocal(Math.Exponential((dataset.multiply(parameters).add(bias)).multiply(-1)).add(1));
        }
        else
        {
            sigmoid = Math.Reciprocal(Math.Exponential(dataset.multiply(iterate).multiply(-1)).add(1));
        }

        Matrix error = sigmoid.subtract(labels.transpose());
        Matrix gradient;

        if (fitIntercept)
        {
            double[] B = new double[]{error.divide(numFunctions()).sum()};
            Vector V = Vector.fromArray(B);
            gradient = (error.transpose().divide(numFunctions())).multiply(dataset).insertColumn(0, V);

        }
        else
        {
            gradient = (error.transpose().divide(numFunctions())).multiply(dataset);
        }

        Matrix reg = iterate.multiply(lambda / (2 * numFunctions())).transpose();
        return gradient.add(reg).transpose();
    }

    public Matrix Gradient(Matrix iterate,
                           int id,
                           int batchSize)
    {
        Matrix sigmoid;
        int lastId = batchSize + id;

        if (fitIntercept)
        {
            double bias = iterate.get(0, 0);
            Matrix parameters = iterate.slice(1, 0, iterate.rows(), iterate.columns());
            sigmoid = Math.Reciprocal(Math.Exponential((dataset.slice(id, 0, lastId, dataset.columns()).multiply(parameters).add(bias)).multiply(-1)).add(1));
        }
        else
        {
            sigmoid = Math.Reciprocal(Math.Exponential(dataset.slice(id, 0, lastId, dataset.columns()).multiply(iterate).multiply(-1)).add(1));
        }

        Matrix error = sigmoid.subtract(labels.slice(0, id, 1, lastId).transpose());
        Matrix gradient;

        if (fitIntercept)
        {
            double[] B = new double[]{error.divide(batchSize).sum()};
            Vector V = Vector.fromArray(B);
            gradient = (error.transpose().divide(batchSize)).multiply(dataset.slice(id, 0, lastId, dataset.columns())).insertColumn(0, V);

        }
        else
        {
            gradient = (error.transpose().divide(batchSize)).multiply(dataset.slice(id, 0, lastId, dataset.columns()));
        }

        Matrix reg = iterate.multiply(lambda / (2 * batchSize)).transpose();
        return gradient.add(reg).transpose();
    }

    public int numFunctions()
    {
        return dataset.rows();
    }

    public Matrix initializeWeights()
    {
        if (fitIntercept)
        {
            return Matrix.zero(dataset.columns() + 1, 1);
        }
        else
        {
            return Matrix.zero(dataset.columns(), 1);
        }
    }

    private Matrix dataset;
    private Matrix labels;
    private boolean fitIntercept;
    private double lambda;
}
