package org.mq.methods;

import org.junit.Assert;
import org.junit.Test;
import org.la4j.Matrix;
import org.mq.core.optimizers.sgd.SGD;
import org.mq.methods.linear_regression.LinearRegressionFunction;

public class LinearRegressionTest
{
    @Test
    public void LinearRegressionFunctionSimpleEvaluate()
    {
        double[][] D = {{1, 2, 3},
                        {1, 4, 5},
                        {1, 6, 7},
                        {1, 8, 9}};
        Matrix dataset = Matrix.from2DArray(D);

        double[][] L = new double[][]{{1, 2, 3, 4}};
        Matrix labels = Matrix.from2DArray(L);

        LinearRegressionFunction lrf = new LinearRegressionFunction(dataset, labels);

        double[][] I = new double[][]{{0.1, 0.2, 0.3}};
        Matrix iterate = Matrix.from2DArray(I);
        Assert.assertEquals(lrf.Evaluate(iterate), 0.08, 1e-5);

        I = new double[][]{{0.0, 0.0, 0.0}};
        iterate = Matrix.from2DArray(I);
        Assert.assertEquals(lrf.Evaluate(iterate), 3.75, 1e-5);

        I = new double[][]{{-0.167, 0.333, 0.167}};
        iterate = Matrix.from2DArray(I);
        Assert.assertEquals(lrf.Evaluate(iterate), 0.00, 1e-5);
    }

    @Test
    public void LinearRegressionFunctionSimpleSeparableEvaluate()
    {
        double[][] D = {{1, 2, 3},
                        {1, 4, 5},
                        {1, 6, 7},
                        {1, 8, 9}};
        Matrix dataset = Matrix.from2DArray(D);

        double[][] L = new double[][]{{1, 2, 3, 4}};
        Matrix labels = Matrix.from2DArray(L);

        LinearRegressionFunction lrf = new LinearRegressionFunction(dataset, labels);

        double[][] I = new double[][]{{0.1, 0.2, 0.3}};
        Matrix iterate = Matrix.from2DArray(I);
        double loss1 = lrf.Evaluate(iterate, 0, 1);
        double loss2 = lrf.Evaluate(iterate, 1, 1);
        double loss3 = lrf.Evaluate(iterate, 2, 1);
        double loss4 = lrf.Evaluate(iterate, 3, 1);
        double separableLoss = loss1 + loss2 + loss3 + loss4;
        Assert.assertEquals(lrf.Evaluate(iterate), separableLoss / 4, 1e-5);

        I = new double[][]{{0.0, 0.0, 0.0}};
        iterate = Matrix.from2DArray(I);
        loss1 = lrf.Evaluate(iterate, 0, 1);
        loss2 = lrf.Evaluate(iterate, 1, 1);
        loss3 = lrf.Evaluate(iterate, 2, 1);
        loss4 = lrf.Evaluate(iterate, 3, 1);
        separableLoss = loss1 + loss2 + loss3 + loss4;
        Assert.assertEquals(lrf.Evaluate(iterate), separableLoss / 4, 1e-5);

        I = new double[][]{{-0.167, 0.333, 0.167}};
        iterate = Matrix.from2DArray(I);
        loss1 = lrf.Evaluate(iterate, 0, 1);
        loss2 = lrf.Evaluate(iterate, 1, 1);
        loss3 = lrf.Evaluate(iterate, 2, 1);
        loss4 = lrf.Evaluate(iterate, 3, 1);
        separableLoss = loss1 + loss2 + loss3 + loss4;
        Assert.assertEquals(lrf.Evaluate(iterate), separableLoss / 4, 1e-5);
    }

    @Test
    public void LinearRegressionFunctionSimpleFitInterceptEvaluate()
    {
        double[][] D1 = {{1, 2, 3},
                         {1, 4, 5},
                         {1, 6, 7},
                         {1, 8, 9}};
        Matrix dataset1 = Matrix.from2DArray(D1);

        double[][] D2 = {{2, 3},
                         {4, 5},
                         {6, 7},
                         {8, 9}};
        Matrix dataset2 = Matrix.from2DArray(D2);

        double[][] L = new double[][]{{1, 2, 3, 4}};
        Matrix labels = Matrix.from2DArray(L);

        LinearRegressionFunction lrf1 = new LinearRegressionFunction(dataset1, labels);
        LinearRegressionFunction lrf2 = new LinearRegressionFunction(dataset2, labels, true, 0);

        double[][] I = new double[][]{{0.1, 0.2, 0.3}};
        Matrix iterate = Matrix.from2DArray(I);
        Assert.assertEquals(lrf1.Evaluate(iterate), lrf2.Evaluate(iterate), 1e-5);

        I = new double[][]{{0.0, 0.0, 0.0}};
        iterate = Matrix.from2DArray(I);
        Assert.assertEquals(lrf1.Evaluate(iterate), lrf2.Evaluate(iterate), 1e-5);

        I = new double[][]{{-0.167, 0.333, 0.167}};
        iterate = Matrix.from2DArray(I);
        Assert.assertEquals(lrf1.Evaluate(iterate), lrf2.Evaluate(iterate), 1e-5);
    }


    @Test
    public void LinearRegressionFunctionSimpleSeparableEvaluateFitIntercept()
    {
        double[][] D1 = {{1, 2, 3},
                         {1, 4, 5},
                         {1, 6, 7},
                         {1, 8, 9}};
        Matrix dataset1 = Matrix.from2DArray(D1);

        double[][] D2 = {{2, 3},
                         {4, 5},
                         {6, 7},
                         {8, 9}};
        Matrix dataset2 = Matrix.from2DArray(D2);

        double[][] L = new double[][]{{1, 2, 3, 4}};
        Matrix labels = Matrix.from2DArray(L);

        LinearRegressionFunction lrf1 = new LinearRegressionFunction(dataset1, labels);
        LinearRegressionFunction lrf2 = new LinearRegressionFunction(dataset2, labels, true, 0.0);

        double[][] I = new double[][]{{0.1, 0.2, 0.3}};
        Matrix iterate = Matrix.from2DArray(I);
        double loss1 = lrf2.Evaluate(iterate, 0, 1);
        double loss2 = lrf2.Evaluate(iterate, 1, 1);
        double loss3 = lrf2.Evaluate(iterate, 2, 1);
        double loss4 = lrf2.Evaluate(iterate, 3, 1);
        double separableLoss = loss1 + loss2 + loss3 + loss4;
        Assert.assertEquals(lrf1.Evaluate(iterate), separableLoss / 4, 1e-5);

        I = new double[][]{{0.0, 0.0, 0.0}};
        iterate = Matrix.from2DArray(I);
        loss1 = lrf2.Evaluate(iterate, 0, 1);
        loss2 = lrf2.Evaluate(iterate, 1, 1);
        loss3 = lrf2.Evaluate(iterate, 2, 1);
        loss4 = lrf2.Evaluate(iterate, 3, 1);
        separableLoss = loss1 + loss2 + loss3 + loss4;
        Assert.assertEquals(lrf1.Evaluate(iterate), separableLoss / 4, 1e-5);

        I = new double[][]{{-0.167, 0.333, 0.167}};
        iterate = Matrix.from2DArray(I);
        loss1 = lrf2.Evaluate(iterate, 0, 1);
        loss2 = lrf2.Evaluate(iterate, 1, 1);
        loss3 = lrf2.Evaluate(iterate, 2, 1);
        loss4 = lrf2.Evaluate(iterate, 3, 1);
        separableLoss = loss1 + loss2 + loss3 + loss4;
        Assert.assertEquals(lrf1.Evaluate(iterate), separableLoss / 4, 1e-5);
    }

    @Test
    public void LinearRegressionFunctionSimpleGradient()
    {
        double[][] array = {{1, 2},
                            {3, 4},
                            {5, 6}};
        Matrix dataset = Matrix.from2DArray(array);

        array = new double[][]{{1, 2, 3}};
        Matrix labels = Matrix.from2DArray(array);

        array = new double[][]{{0.1, 0.2}};
        Matrix params = Matrix.from2DArray(array);

        LinearRegressionFunction lrf = new LinearRegressionFunction(dataset, labels);


        System.out.println(lrf.Gradient(params, 0, 1));
        System.out.println(lrf.Gradient(params, 1, 1));
        System.out.println(lrf.Gradient(params, 2, 1));

        SGD gd = new SGD(0.003, 30000, 1e-20, 1);
        params = gd.Optimize(lrf, params);
        System.out.println(params);

        System.out.println(params.multiply(dataset.transpose()));

    }
    @Test
    public void LinearRegressionFunctionSimpleFitInterceptGradient()
    {
    double[][] D1 = {{1, 2, 3},
                     {1, 4, 5},
                     {1, 6, 7},
                     {1, 8, 9}};
    Matrix dataset1 = Matrix.from2DArray(D1);

    double[][] D2 = {{2, 3},
                     {4, 5},
                     {6, 7},
                     {8, 9}};
    Matrix dataset2 = Matrix.from2DArray(D2);

    double[][] L = new double[][]{{1, 2, 3, 4}};
    Matrix labels = Matrix.from2DArray(L);

    LinearRegressionFunction lrf1 = new LinearRegressionFunction(dataset1, labels);
    LinearRegressionFunction lrf2 = new LinearRegressionFunction(dataset2, labels, true, 0);

    double[][] I = new double[][]{{1, 1, 1}};
    Matrix iterate = Matrix.from2DArray(I);

    Matrix Gradient1 = lrf1.Gradient(iterate);
    Matrix Gradient2 = lrf2.Gradient(iterate);

    Assert.assertEquals(Gradient1.get(0,0), Gradient2.get(0,0), 1e-5);
    Assert.assertEquals(Gradient1.get(0,1), Gradient2.get(0,1), 1e-5);
    Assert.assertEquals(Gradient1.get(0,2), Gradient2.get(0,2), 1e-5);

    I = new double[][]{{0.0, 0.0, 0.0}};
    iterate = Matrix.from2DArray(I);
    Assert.assertEquals(Gradient1.get(0,0), Gradient2.get(0,0), 1e-5);
    Assert.assertEquals(Gradient1.get(0,1), Gradient2.get(0,1), 1e-5);
    Assert.assertEquals(Gradient1.get(0,2), Gradient2.get(0,2), 1e-5);

    I = new double[][]{{123.5, 140.5, 157.5}};
    iterate = Matrix.from2DArray(I);
    Assert.assertEquals(Gradient1.get(0,0), Gradient2.get(0,0), 1e-5);
    Assert.assertEquals(Gradient1.get(0,1), Gradient2.get(0,1), 1e-5);
    Assert.assertEquals(Gradient1.get(0,2), Gradient2.get(0,2), 1e-5);

    }

    @Test
    public void LinearRegressionFunctionSimpleFitInterceptSeparableGradient()
    {
        double[][] D1 = {{1, 2, 3},
                         {1, 4, 5},
                         {1, 6, 7},
                         {1, 8, 9}};

        Matrix dataset1 = Matrix.from2DArray(D1);

        double[][] D2 = {{2, 3},
                         {4, 5},
                         {6, 7},
                         {8, 9}};
        Matrix dataset2 = Matrix.from2DArray(D2);

        double[][] L = new double[][]{{1, 2, 3, 4}};
        Matrix labels = Matrix.from2DArray(L);

        LinearRegressionFunction lrf1 = new LinearRegressionFunction(dataset1, labels);
        LinearRegressionFunction lrf2 = new LinearRegressionFunction(dataset2, labels, true, 0);

        double[][] I = new double[][]{{1, 1, 1}};
        Matrix iterate = Matrix.from2DArray(I);

        Matrix Gradient1 = lrf1.Gradient(iterate);
        Matrix Gradient2 = lrf2.Gradient(iterate);

        Assert.assertEquals(Gradient1.get(0,0), Gradient2.get(0,0), 1e-5);
        Assert.assertEquals(Gradient1.get(0,1), Gradient2.get(0,1), 1e-5);
        Assert.assertEquals(Gradient1.get(0,2), Gradient2.get(0,2), 1e-5);

        I = new double[][]{{0.0, 0.0, 0.0}};
        iterate = Matrix.from2DArray(I);
        Assert.assertEquals(Gradient1.get(0,0), Gradient2.get(0,0), 1e-5);
        Assert.assertEquals(Gradient1.get(0,1), Gradient2.get(0,1), 1e-5);
        Assert.assertEquals(Gradient1.get(0,2), Gradient2.get(0,2), 1e-5);

        I = new double[][]{{123.5, 140.5, 157.5}};
        iterate = Matrix.from2DArray(I);
        Assert.assertEquals(Gradient1.get(0,0), Gradient2.get(0,0), 1e-5);
        Assert.assertEquals(Gradient1.get(0,1), Gradient2.get(0,1), 1e-5);
        Assert.assertEquals(Gradient1.get(0,2), Gradient2.get(0,2), 1e-5);

    }


}
