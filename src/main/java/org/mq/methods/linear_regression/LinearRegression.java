package org.mq.methods.linear_regression;

import org.la4j.Matrix;
import org.mq.core.optimizers.sgd.SGD;

public class LinearRegression {

    public static void main(String[] args)
    {
        double[][] D = {{1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9},
                        {10, 11, 12}};

        Matrix dataset = Matrix.from2DArray(D);

        double[][] L = {{1, 2, 3, 4}};
        Matrix labels = Matrix.from2DArray(L);

        // start
        LinearRegressionFunction lrf = new LinearRegressionFunction(dataset, labels, true);
        SGD sgd = new SGD(0.001, 20000, 1e-10, 1);

        Matrix parameters = lrf.initializeWeights();
        parameters = sgd.Optimize(lrf, parameters);
        System.out.println(parameters);

        //System.out.println(parameters);
        System.out.println(parameters.slice(0,1,1,parameters.columns()));

        System.out.println(((parameters.slice(0, 1, 1, parameters.columns()).multiply(dataset.transpose())).add(parameters.get(0, 0))));
        System.out.println(((parameters.slice(0,1,1,dataset.columns() + 1)).multiply(dataset.transpose())).add(parameters.get(0,0)));
    }
}
