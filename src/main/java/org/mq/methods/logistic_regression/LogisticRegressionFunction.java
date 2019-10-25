package org.mq.methods.logistic_regression;

import org.la4j.Matrix;
import org.la4j.Vector;
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

    public double Evaluate(Matrix iterate)
    {
        Matrix sigmoid;
        if (fitIntercept)
        {
            double bias = iterate.get(0, 0);
            iterate = iterate.slice(1, 0, iterate.rows(), iterate.columns());
            sigmoid = Reciprocal(Exponential((dataset.multiply(iterate).add(bias)).multiply(-1)).add(1));
        }
        else
        {
            sigmoid = Reciprocal(Exponential(dataset.multiply(iterate).multiply(-1)).add(1));
        }

        Matrix error1 = SchurProduct(labels.transpose(), Log(sigmoid)).multiply(-1);
        Matrix error2 = SchurProduct(labels.transpose().multiply(-1).add(1), Log(sigmoid.multiply(-1).add(1))).multiply(-1);
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
            sigmoid = Reciprocal(Exponential((dataset.slice(id, 0, lastId, dataset.columns()).multiply(iterate).add(bias)).multiply(-1)).add(1));
        }
        else
        {
            sigmoid = Reciprocal(Exponential(dataset.slice(id, 0, lastId, dataset.columns()).multiply(iterate).multiply(-1)).add(1));
        }

        Matrix error1 = SchurProduct(labels.slice(0, id, 1, lastId).transpose(), Log(sigmoid)).multiply(-1);
        Matrix error2 = SchurProduct(labels.slice(0, id, 1, lastId).transpose().multiply(-1).add(1), Log(sigmoid.multiply(-1).add(1))).multiply(-1);
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
            sigmoid = Reciprocal(Exponential((dataset.multiply(parameters).add(bias)).multiply(-1)).add(1));
        }
        else
        {
            sigmoid = Reciprocal(Exponential(dataset.multiply(iterate).multiply(-1)).add(1));
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
        return gradient.add(reg);
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
            sigmoid = Reciprocal(Exponential((dataset.slice(id, 0, lastId, dataset.columns()).multiply(parameters).add(bias)).multiply(-1)).add(1));
        }
        else
        {
            sigmoid = Reciprocal(Exponential(dataset.slice(id, 0, lastId, dataset.columns()).multiply(iterate).multiply(-1)).add(1));
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
        return gradient.add(reg);
    }

    public int numFunctions()
    {
        return dataset.rows();
    }

    private static Matrix Reciprocal(Matrix M)
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

    private static Matrix Exponential(Matrix M)
    {
        double[][] R = new double[M.rows()][M.columns()];

        for (int i = 0; i < M.rows(); i++)
        {
            for (int j = 0; j < M.columns(); j++)
            {
                R[i][j] = Math.exp(M.get(i, j));
            }
        }

        return Matrix.from2DArray(R);
    }

    private static Matrix Log(Matrix M)
    {
        double[][] R = new double[M.rows()][M.columns()];

        for (int i = 0; i < M.rows(); i++)
        {
            for (int j = 0; j < M.columns(); j++)
            {
                R[i][j] = Math.log(M.get(i, j));
            }
        }

        return Matrix.from2DArray(R);
    }

    private static Matrix SchurProduct(Matrix M1, Matrix M2)
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

    private Matrix dataset;
    private Matrix labels;
    private boolean fitIntercept;
    private double lambda;
}
