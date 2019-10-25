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
    public Matrix Gradient(Matrix iterate)
    {
        Matrix gradient;
        Matrix error;
        if(fitIntercept)
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

        System.out.println(gradient);
        Matrix reg = (iterate.multiply(lambda * 1.0 / numFunctions()));
        return gradient.add(reg);
    }

    public int numFunctions()
    {
        return dataset.rows();
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

    public Matrix Gradient(Matrix iterate,
                           int id,
                           int batchSize)
    {
        Matrix dataset1;
        int columns = dataset.columns();
        dataset1 = dataset.slice(id, 0, batchSize + id, columns);

        Matrix grad;

        Matrix labels1;
        labels1 = labels.slice(0, id, 1, batchSize + id);

        if(fitIntercept)
        {
            double bias = iterate.get(0, 0);
            Matrix parameters = iterate.slice(0, 1, 1, iterate.columns());
            grad = ((parameters.multiply(dataset1.transpose()).add(bias)).subtract(labels1));
        }
        else
        {
            grad = (iterate.multiply(dataset1.transpose()).subtract(labels1));
        }

            Matrix reg = (iterate.multiply(lambda * 1.0 / batchSize));
            return grad.multiply(dataset1).multiply(1.0 / batchSize).add(reg);
    }


    private Matrix dataset;
    private Matrix labels;
    private boolean fitIntercept;
    private double lambda;
}