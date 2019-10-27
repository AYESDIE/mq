package org.mq.methods.linear_regression;
import org.la4j.Matrix;
import org.la4j.Vector;
import org.mq.core.optimizers.functions.DifferentiableFunction;
import org.mq.core.optimizers.gradient_descent.GradientDescent;

import org.mq.core.optimizers.sgd.SGD;

public class LinearRegressionFunction extends DifferentiableFunction {


    public LinearRegressionFunction(Matrix dataset,
                                    Matrix labels)
    {
        this.dataset = dataset;
        this.labels = labels;
        this.fitIntercept = false;
        this.lambda = 0.0;
    }

    public LinearRegressionFunction(Matrix dataset,
                                    Matrix labels,
                                    boolean fitIntercept)
    {
        this.dataset = dataset;
        this.labels = labels;
        this.fitIntercept = fitIntercept;
        this.lambda = 0.0;
    }

    public LinearRegressionFunction(Matrix dataset,
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
        double loss;
        Matrix error;

        if (fitIntercept)
        {
            double bias = iterate.get(0, 0);
            iterate = iterate.slice(0, 1, 1, iterate.columns());

            error = ((iterate.multiply(dataset.transpose())).add(bias)).subtract(labels);
        }
        else
        {
            error = iterate.multiply(dataset.transpose()).subtract(labels);
        }

        Matrix cost = error.multiplyByItsTranspose().multiply(1.0 / (2 * numFunctions()));
        double reg = iterate.multiply(iterate.transpose()).multiply(lambda * 1.0/(2 * numFunctions())).get(0, 0);
        loss = cost.get(0, 0);
        return loss + reg;
    }

    public double Evaluate(Matrix iterate,
                           int id,
                           int batchSize)
    {
        double loss;
        Matrix error;
        int lastId = id + batchSize;

        if (fitIntercept)
        {
            double bias = iterate.get(0, 0);
            iterate = iterate.slice(0, 1, 1, iterate.columns());

            error = ((iterate.multiply(dataset.slice(id, 0, lastId, dataset.columns()).transpose())).add(bias)).
                    subtract(labels.slice(0, id, 1, lastId));
        }
        else
        {
            error = iterate.multiply(dataset.slice(id, 0, lastId, dataset.columns()).transpose()).
                    subtract(labels.slice(0, id, 1, lastId));
        }

        Matrix cost = error.multiplyByItsTranspose().multiply(1.0 / (2 * batchSize));
        double reg = iterate.multiply(iterate.transpose()).multiply(lambda * 1.0/(2 * batchSize)).get(0, 0);
        loss = cost.get(0, 0);
        return loss + reg;
    }

    public Matrix Gradient(Matrix iterate)
    {
        Matrix gradient;
        Matrix error;
        if (fitIntercept)
        {
            double bias = iterate.get(0, 0);
            Matrix parameters = iterate.slice(0, 1, 1, iterate.columns());
            error = (parameters.multiply(dataset.transpose()).add(bias)).subtract(labels);

            gradient = (error.multiply(dataset)).divide(numFunctions());

            double[] B = new double[]{error.divide(numFunctions()).sum()};
            Vector V = Vector.fromArray(B);
            gradient = gradient.insertColumn(0, V);
        }
        else
        {
            gradient = (iterate.multiply(dataset.transpose()).subtract(labels));
            gradient = (gradient.multiply(dataset)).divide(numFunctions());
        }

        Matrix reg = (iterate.multiply(lambda * 1.0 / numFunctions()));
        return gradient.add(reg);
    }



    public Matrix Gradient(Matrix iterate,
                           int id,
                           int batchSize)
    {
        Matrix gradient;
        Matrix error;

        if (fitIntercept)
        {
            double bias = iterate.get(0, 0);
            Matrix parameters = iterate.slice(0, 1, 1, iterate.columns());
            error = (parameters.multiply(dataset.slice(id, 0, batchSize + id, dataset.columns()).transpose()).
                     add(bias)).subtract(labels.slice(0,id,1,batchSize + id));

            gradient = (error.multiply(dataset.slice(id, 0, batchSize + id, dataset.columns())).divide(batchSize));

            double[] B = new double[]{error.divide(batchSize).sum()};
            Vector V = Vector.fromArray(B);
            gradient = gradient.insertColumn(0, V);
        }
        else
        {
            gradient = (iterate.multiply(dataset.slice(id, 0, batchSize + id, dataset.columns()).transpose()).
                        subtract(labels.slice(0,id,1,batchSize + id)));
            gradient = (gradient.multiply(dataset.slice(id, 0, batchSize + id, dataset.columns())).divide(batchSize));
        }

            Matrix reg = (iterate.multiply(lambda * 1.0 / batchSize));
            return gradient.add(reg);
    }

    public int numFunctions()
    {
        return dataset.rows();
    }

    private Matrix dataset;
    private Matrix labels;
    private boolean fitIntercept;
    private double lambda;
}