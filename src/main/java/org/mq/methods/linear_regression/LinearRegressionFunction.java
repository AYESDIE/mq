package org.mq.methods.linear_regression;
import org.la4j.Matrix;
import org.mq.core.optimizers.functions.DifferentiableFunction;
import org.mq.core.optimizers.gradient_descent.GradientDescent;

import org.mq.core.optimizers.sgd.SGD;

public class LinearRegressionFunction extends DifferentiableFunction {


    public LinearRegressionFunction(Matrix dataset,
                                    Matrix labels)
    {
        this.dataset = dataset;
        this.labels = labels;
        this.lambda = 0.0;
    }

    public LinearRegressionFunction(Matrix dataset,
                                    Matrix labels,
                                    double lambda)
    {
        this.dataset = dataset;
        this.labels = labels;
        this.lambda = lambda;
    }

    public double Evaluate(Matrix iterate)
    {

        double loss;
        Matrix cost;
        Matrix first = (iterate.multiply(dataset.transpose()).subtract(labels));
        Matrix fourth = (iterate.multiply(iterate.transpose()).multiply(lambda * 1.0/(2 * numFunctions())));
        cost = ((first.multiply(first.transpose())).multiply(1.0/(2 * numFunctions())).add(fourth));
        loss = cost.get(0, 0);

        return loss;
    }
    public Matrix Gradient(Matrix iterate)
    {
        Matrix second = (iterate.multiply(dataset.transpose()).subtract(labels));
        Matrix third = (iterate.multiply(lambda * 1.0/(numFunctions())));
        return (second.multiply(dataset)).multiply(1.0/numFunctions()).add(third);
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
        Matrix dataset1;
        int columns = dataset.columns();
        dataset1 = dataset.slice(id, 0, batchSize + id, columns);
        //System.out.println(dataset.slice(0, 0, 1, columns));

        Matrix labels1;
        labels1 = labels.slice(0, id, 1, batchSize + id);
        //System.out.println(labels.slice(0, 0, 1, 1));
        //System.out.println(labels.slice(0, 1, 1, 2));
        //System.out.println(labels.slice(0, 2, 1, 3));

        Matrix cost;
        Matrix first = (iterate.multiply(dataset1.transpose()).subtract(labels1));
        Matrix fourth = (iterate.multiply(iterate.transpose()).multiply(lambda * 1.0/(2 * batchSize)));
        cost = ((first.multiply(first.transpose())).multiply(1.0/(2 * batchSize)).add(fourth));
        loss = cost.get(0, 0);

        return loss;
    }

    public Matrix Gradient(Matrix iterate,
                           int id,
                           int batchSize)
    {
        Matrix dataset1;
        int columns = dataset.columns();
        dataset1 = dataset.slice(id, 0, batchSize + id, columns);

        Matrix labels1;
        labels1 = labels.slice(0, id, 1, batchSize + id);

        Matrix second = (iterate.multiply(dataset1.transpose()).subtract(labels1));
        Matrix third = (iterate.multiply(lambda * 1.0/batchSize));

        return (second.multiply(dataset1)).multiply(1.0/batchSize).add(third);

    }


    private Matrix dataset;
    private Matrix labels;
    private double lambda;

    public static void main(String[] args)
    {
        double[][] array = {{1, 2}, {3, 4}, {5, 6}};
        Matrix dataset = Matrix.from2DArray(array);
        //System.out.println(dataset);

        array = new double[][]{{1, 2, 3}};
        Matrix labels = Matrix.from2DArray(array);
        //System.out.println(labels);

        array = new double[][]{{0.1, 0.2}};
        Matrix params = Matrix.from2DArray(array);
        //System.out.println(params);

        LinearRegressionFunction lrf = new LinearRegressionFunction(dataset, labels);
        //lrf.Evaluate(params, 0, 1);
        //System.out.println(lrf.Evaluate(params));

        //System.out.println(lrf.Evaluate(params, 0, 1));
        //System.out.println(lrf.Evaluate(params, 1, 1));
        //System.out.println(lrf.Evaluate(params, 2, 1));

        //double a = lrf.Evaluate(params, 0, 1);
        //double b = lrf.Evaluate(params, 1, 2);
        //double c = lrf.Evaluate(params, 2, 3);



        //System.out.println(lrf.Gradient(params));

        System.out.println(lrf.Gradient(params, 0, 1));
        System.out.println(lrf.Gradient(params, 1, 1));
        System.out.println(lrf.Gradient(params, 2, 1));

        SGD gd = new SGD(0.003, 30000, 1e-20, 1);
        params = gd.Optimize(lrf, params);
        System.out.println(params);

        System.out.println(params.multiply(dataset.transpose()));
    }

}