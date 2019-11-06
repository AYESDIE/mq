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

        double[][] D = {{1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9},
                        {10, 11, 12}};
        Matrix dataset = Matrix.from2DArray(D);

        double[][] L = new double[][]{{1, 2, 3, 4}};
        Matrix labels = Matrix.from2DArray(L);

        LinearRegressionFunction lrf = new LinearRegressionFunction(dataset, labels);

        double[][] I = new double[][]{{1, 1, 1}};
        Matrix iterate = Matrix.from2DArray(I);

        Matrix Gradient = lrf.Gradient(iterate);
        Assert.assertEquals(Gradient.get(0,0), 123.5, 1e-5);
        Assert.assertEquals(Gradient.get(0,1), 140.5, 1e-5);
        Assert.assertEquals(Gradient.get(0,2), 157.5, 1e-5);

        I = new double[][]{{0.0, 0.0, 1.0/3}};
        iterate = Matrix.from2DArray(I);
        Gradient = lrf.Gradient(iterate);
        Assert.assertEquals(Gradient.get(0,0), 0, 1e-5);
        Assert.assertEquals(Gradient.get(0,1), 0, 1e-5);
        Assert.assertEquals(Gradient.get(0,2), 0, 1e-5);

        I = new double[][]{{10, -204.5, 23.5}};
        iterate = Matrix.from2DArray(I);
        Gradient = lrf.Gradient(iterate);
        Assert.assertEquals(Gradient.get(0,0), -7980.25, 1e-5);
        Assert.assertEquals(Gradient.get(0,1), -9080.75, 1e-5);
        Assert.assertEquals(Gradient.get(0,2), -10181.25, 1e-5);
    }

    @Test
    public void LinearRegressionFunctionSimpleSeparableGradient()
    {
        double[][] D = {{1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9},
                        {10, 11, 12}};
        Matrix dataset = Matrix.from2DArray(D);

        double[][] L = new double[][]{{1, 2, 3, 4}};
        Matrix labels = Matrix.from2DArray(L);

        LinearRegressionFunction lrf = new LinearRegressionFunction(dataset, labels);

        double[][] I = new double[][]{{1, 1, 1}};
        Matrix iterate = Matrix.from2DArray(I);
        Matrix gradient = lrf.Gradient(iterate);
        Matrix separableGradient = lrf.Gradient(iterate, 0, 1);
        separableGradient = separableGradient.add(lrf.Gradient(iterate, 1, 1));
        separableGradient = separableGradient.add(lrf.Gradient(iterate, 2, 1));
        separableGradient = separableGradient.add(lrf.Gradient(iterate, 3, 1));
        separableGradient = separableGradient.divide(4);
        Assert.assertEquals(gradient.get(0,0), separableGradient.get(0, 0), 1e-5);
        Assert.assertEquals(gradient.get(0,1), separableGradient.get(0, 1), 1e-5);
        Assert.assertEquals(gradient.get(0,2), separableGradient.get(0, 2), 1e-5);

        I = new double[][]{{0.0, 0.0, 1.0/3}};
        iterate = Matrix.from2DArray(I);
        gradient = lrf.Gradient(iterate);
        separableGradient = lrf.Gradient(iterate, 0, 1);
        separableGradient = separableGradient.add(lrf.Gradient(iterate, 1, 1));
        separableGradient = separableGradient.add(lrf.Gradient(iterate, 2, 1));
        separableGradient = separableGradient.add(lrf.Gradient(iterate, 3, 1));
        separableGradient = separableGradient.divide(4);
        Assert.assertEquals(gradient.get(0,0), separableGradient.get(0, 0), 1e-5);
        Assert.assertEquals(gradient.get(0,1), separableGradient.get(0, 1), 1e-5);
        Assert.assertEquals(gradient.get(0,2), separableGradient.get(0, 2), 1e-5);

        I = new double[][]{{10, -204.5, 23.5}};
        iterate = Matrix.from2DArray(I);
        gradient = lrf.Gradient(iterate);
        separableGradient = lrf.Gradient(iterate, 0, 1);
        separableGradient = separableGradient.add(lrf.Gradient(iterate, 1, 1));
        separableGradient = separableGradient.add(lrf.Gradient(iterate, 2, 1));
        separableGradient = separableGradient.add(lrf.Gradient(iterate, 3, 1));
        separableGradient = separableGradient.divide(4);
        Assert.assertEquals(gradient.get(0,0), separableGradient.get(0, 0), 1e-5);
        Assert.assertEquals(gradient.get(0,1), separableGradient.get(0, 1), 1e-5);
        Assert.assertEquals(gradient.get(0,2), separableGradient.get(0, 2), 1e-5);
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
    Gradient1 = lrf1.Gradient(iterate);
    Gradient2 = lrf2.Gradient(iterate);
    Assert.assertEquals(Gradient1.get(0,0), Gradient2.get(0,0), 1e-5);
    Assert.assertEquals(Gradient1.get(0,1), Gradient2.get(0,1), 1e-5);
    Assert.assertEquals(Gradient1.get(0,2), Gradient2.get(0,2), 1e-5);

    I = new double[][]{{123.5, 140.5, 157.5}};
    iterate = Matrix.from2DArray(I);
    Gradient1 = lrf1.Gradient(iterate);
    Gradient2 = lrf2.Gradient(iterate);
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
        Matrix gradient = lrf1.Gradient(iterate);
        Matrix separableGradient = lrf2.Gradient(iterate, 0, 1);
        separableGradient = separableGradient.add(lrf2.Gradient(iterate, 1, 1));
        separableGradient = separableGradient.add(lrf2.Gradient(iterate, 2, 1));
        separableGradient = separableGradient.add(lrf2.Gradient(iterate, 3, 1));
        separableGradient = separableGradient.divide(4);
        Assert.assertEquals(gradient.get(0,0), separableGradient.get(0, 0), 1e-5);
        Assert.assertEquals(gradient.get(0,1), separableGradient.get(0, 1), 1e-5);
        Assert.assertEquals(gradient.get(0,2), separableGradient.get(0, 2), 1e-5);

        I = new double[][]{{0.0, 0.0, 1.0/3}};
        iterate = Matrix.from2DArray(I);
        gradient = lrf1.Gradient(iterate);
        separableGradient = lrf2.Gradient(iterate, 0, 1);
        separableGradient = separableGradient.add(lrf2.Gradient(iterate, 1, 1));
        separableGradient = separableGradient.add(lrf2.Gradient(iterate, 2, 1));
        separableGradient = separableGradient.add(lrf2.Gradient(iterate, 3, 1));
        separableGradient = separableGradient.divide(4);
        Assert.assertEquals(gradient.get(0,0), separableGradient.get(0, 0), 1e-5);
        Assert.assertEquals(gradient.get(0,1), separableGradient.get(0, 1), 1e-5);
        Assert.assertEquals(gradient.get(0,2), separableGradient.get(0, 2), 1e-5);

        I = new double[][]{{10, -204.5, 23.5}};
        iterate = Matrix.from2DArray(I);
        gradient = lrf1.Gradient(iterate);
        separableGradient = lrf2.Gradient(iterate, 0, 1);
        separableGradient = separableGradient.add(lrf2.Gradient(iterate, 1, 1));
        separableGradient = separableGradient.add(lrf2.Gradient(iterate, 2, 1));
        separableGradient = separableGradient.add(lrf2.Gradient(iterate, 3, 1));
        separableGradient = separableGradient.divide(4);
        Assert.assertEquals(gradient.get(0,0), separableGradient.get(0, 0), 1e-5);
        Assert.assertEquals(gradient.get(0,1), separableGradient.get(0, 1), 1e-5);
        Assert.assertEquals(gradient.get(0,2), separableGradient.get(0, 2), 1e-5);
    }

    @Test
    public void LinearRegressionFunctionInitializeWeights()
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

        double[][] L = new double[][]{{0, 0, 1 ,1}};
        Matrix labels = Matrix.from2DArray(L);

        LinearRegressionFunction lrf1 = new LinearRegressionFunction(dataset1, labels);
        LinearRegressionFunction lrf2 = new LinearRegressionFunction(dataset2, labels, true);

        Assert.assertEquals(lrf1.Evaluate(lrf1.initializeWeights()), lrf2.Evaluate(lrf2.initializeWeights()), 1e-3);
    }


}
