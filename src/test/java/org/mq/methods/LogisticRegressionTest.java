package org.mq.methods;

import org.junit.Assert;
import org.junit.Test;
import org.la4j.Matrix;
import org.mq.core.optimizers.sgd.SGD;
import org.mq.core.optimizers.test_functions.SGDTestFunction;
import org.mq.methods.logistic_regression.LogisticRegressionFunction;

public class LogisticRegressionTest
{
    @Test
    public void LogisticRegressionFunctionSimpleEvaluate()
    {
        double[][] D = {{1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9},
                        {10, 11, 12}};
        Matrix dataset = Matrix.from2DArray(D);

        double[][] L = {{0, 0, 1 ,1}};
        Matrix labels = Matrix.from2DArray(L);

        LogisticRegressionFunction lrf = new LogisticRegressionFunction(dataset, labels);

        double[][] P = new double[][]{{1, 1, 1}};
        Matrix iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        Assert.assertEquals(lrf.Evaluate(iterate), 5.2506, 1e-3);

        P = new double[][]{{9.382532, 0.88376, -7.615013}};
        iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        Assert.assertEquals(lrf.Evaluate(iterate), 0.0095, 1e-3);

        P = new double[][]{{-5, 3, -7}};
        iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        Assert.assertEquals(lrf.Evaluate(iterate), 43.7500, 1e-3);
    }

    @Test
    public void LogisticRegressionFunctionSimpleEvaluateFitIntercept()
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

        double[][] L = {{0, 0, 1 ,1}};
        Matrix labels = Matrix.from2DArray(L);

        LogisticRegressionFunction lrf1 = new LogisticRegressionFunction(dataset1, labels);
        LogisticRegressionFunction lrf2 = new LogisticRegressionFunction(dataset2, labels, true);

        double[][] P = new double[][]{{1, 1, 1}};
        Matrix iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        Assert.assertEquals(lrf1.Evaluate(iterate), lrf2.Evaluate(iterate), 1e-3);

        P = new double[][]{{9.382532, 0.88376, -7.615013}};
        iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        Assert.assertEquals(lrf1.Evaluate(iterate), lrf2.Evaluate(iterate), 1e-3);

        P = new double[][]{{-5, 3, -7}};
        iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        Assert.assertEquals(lrf1.Evaluate(iterate), lrf2.Evaluate(iterate), 1e-3);
    }

    @Test
    public void LogisticRegressionFunctionSimpleSeparableEvaluate()
    {
        double[][] D = {{1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9},
                        {10, 11, 12}};
        Matrix dataset = Matrix.from2DArray(D);

        double[][] L = {{0, 0, 1 ,1}};
        Matrix labels = Matrix.from2DArray(L);

        LogisticRegressionFunction lrf = new LogisticRegressionFunction(dataset, labels);

        double[][] P = new double[][]{{1, 1, 1}};
        Matrix iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        double loss1 = lrf.Evaluate(iterate, 0, 1);
        double loss2 = lrf.Evaluate(iterate, 1, 1);
        double loss3 = lrf.Evaluate(iterate, 2, 1);
        double loss4 = lrf.Evaluate(iterate, 3, 1);
        double separableLoss = loss1 + loss2 + loss3 + loss4;
        Assert.assertEquals(lrf.Evaluate(iterate), separableLoss / 4, 1e-3);

        P = new double[][]{{9.382532, 0.88376, -7.615013}};
        iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        loss1 = lrf.Evaluate(iterate, 0, 1);
        loss2 = lrf.Evaluate(iterate, 1, 1);
        loss3 = lrf.Evaluate(iterate, 2, 1);
        loss4 = lrf.Evaluate(iterate, 3, 1);
        separableLoss = loss1 + loss2 + loss3 + loss4;
        Assert.assertEquals(lrf.Evaluate(iterate), separableLoss / 4, 1e-3);

        P = new double[][]{{-5, 3, -7}};
        iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        loss1 = lrf.Evaluate(iterate, 0, 1);
        loss2 = lrf.Evaluate(iterate, 1, 1);
        loss3 = lrf.Evaluate(iterate, 2, 1);
        loss4 = lrf.Evaluate(iterate, 3, 1);
        separableLoss = loss1 + loss2 + loss3 + loss4;
        Assert.assertEquals(lrf.Evaluate(iterate), separableLoss / 4, 1e-3);
    }

    @Test
    public void LogisticRegressionFunctionSimpleSeparableEvaluateFitIntercept()
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

        double[][] L = {{0, 0, 1 ,1}};
        Matrix labels = Matrix.from2DArray(L);

        LogisticRegressionFunction lrf1 = new LogisticRegressionFunction(dataset1, labels);
        LogisticRegressionFunction lrf2 = new LogisticRegressionFunction(dataset2, labels, true);

        double[][] P = new double[][]{{1, 1, 1}};
        Matrix iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        double loss1 = lrf2.Evaluate(iterate, 0, 1);
        double loss2 = lrf2.Evaluate(iterate, 1, 1);
        double loss3 = lrf2.Evaluate(iterate, 2, 1);
        double loss4 = lrf2.Evaluate(iterate, 3, 1);
        double separableLoss = loss1 + loss2 + loss3 + loss4;
        Assert.assertEquals(lrf1.Evaluate(iterate), separableLoss / 4, 1e-3);

        P = new double[][]{{9.382532, 0.88376, -7.615013}};
        iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        loss1 = lrf2.Evaluate(iterate, 0, 1);
        loss2 = lrf2.Evaluate(iterate, 1, 1);
        loss3 = lrf2.Evaluate(iterate, 2, 1);
        loss4 = lrf2.Evaluate(iterate, 3, 1);
        separableLoss = loss1 + loss2 + loss3 + loss4;
        Assert.assertEquals(lrf1.Evaluate(iterate), separableLoss / 4, 1e-3);

        P = new double[][]{{-5, 3, -7}};
        iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        loss1 = lrf2.Evaluate(iterate, 0, 1);
        loss2 = lrf2.Evaluate(iterate, 1, 1);
        loss3 = lrf2.Evaluate(iterate, 2, 1);
        loss4 = lrf2.Evaluate(iterate, 3, 1);
        separableLoss = loss1 + loss2 + loss3 + loss4;
        Assert.assertEquals(lrf1.Evaluate(iterate), separableLoss / 4, 1e-3);
    }

    @Test
    public void LogisticRegressionFunctionSimpleGradient()
    {
        double[][] D = {{1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9},
                        {10, 11, 12}};
        Matrix dataset = Matrix.from2DArray(D);

        double[][] L = {{0, 0, 1 ,1}};
        Matrix labels = Matrix.from2DArray(L);

        LogisticRegressionFunction lrf = new LogisticRegressionFunction(dataset, labels);

        double[][] P = new double[][]{{1, 1, 1}};
        Matrix iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        Matrix gradient = lrf.Gradient(iterate);
        Assert.assertEquals(gradient.get(0, 0), 1.249382, 1e-3);
        Assert.assertEquals(gradient.get(0, 1), 1.748763, 1e-3);
        Assert.assertEquals(gradient.get(0, 2), 2.248145, 1e-3);

        P = new double[][]{{9.382532, 0.88376, -7.615013}};
        iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        gradient = lrf.Gradient(iterate);
        Assert.assertEquals(gradient.get(0, 0), -0.002367, 1e-3);
        Assert.assertEquals(gradient.get(0, 1), 0.00022, 1e-3);
        Assert.assertEquals(gradient.get(0, 2), 0.001927, 1e-3);

        P = new double[][]{{-5, 3, -7}};
        iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        gradient = lrf.Gradient(iterate);
        Assert.assertEquals(gradient.get(0, 0), -4.25, 1e-3);
        Assert.assertEquals(gradient.get(0, 1), -4.75, 1e-3);
        Assert.assertEquals(gradient.get(0, 2), -5.25, 1e-3);
    }

    @Test
    public void LogisticRegressionFunctionSimpleSeparableGradient()
    {
        double[][] D = {{1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9},
                        {10, 11, 12}};
        Matrix dataset = Matrix.from2DArray(D);

        double[][] L = {{0, 0, 1 ,1}};
        Matrix labels = Matrix.from2DArray(L);

        LogisticRegressionFunction lrf = new LogisticRegressionFunction(dataset, labels);

        double[][] P = new double[][]{{1, 1, 1}};
        Matrix iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        Matrix gradient = lrf.Gradient(iterate);
        Matrix separableGradient = lrf.Gradient(iterate, 0, 1);
        separableGradient = separableGradient.add(lrf.Gradient(iterate, 1, 1));
        separableGradient = separableGradient.add(lrf.Gradient(iterate, 2, 1));
        separableGradient = separableGradient.divide(4);
        Assert.assertEquals(gradient.get(0, 0), separableGradient.get(0, 0), 1e-3);
        Assert.assertEquals(gradient.get(0, 1), separableGradient.get(0, 1), 1e-3);
        Assert.assertEquals(gradient.get(0, 2), separableGradient.get(0, 2), 1e-3);
    }

    @Test
    public void LogisticRegressionFunctionSimpleGradientFitIntercept()
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

        double[][] L = {{0, 0, 1 ,1}};
        Matrix labels = Matrix.from2DArray(L);

        LogisticRegressionFunction lrf1 = new LogisticRegressionFunction(dataset1, labels);
        LogisticRegressionFunction lrf2 = new LogisticRegressionFunction(dataset2, labels, true);

        double[][] P = new double[][]{{1, 1, 1}};
        Matrix iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        Matrix gradient1 = lrf1.Gradient(iterate);
        Matrix gradient2 = lrf2.Gradient(iterate);
        Assert.assertEquals(gradient1.get(0, 0), gradient2.get(0, 0), 1e-3);
        Assert.assertEquals(gradient1.get(0, 1), gradient2.get(0, 1), 1e-3);
        Assert.assertEquals(gradient1.get(0, 2), gradient2.get(0, 2), 1e-3);

        P = new double[][]{{9.382532, 0.88376, -7.615013}};
        iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        gradient1 = lrf1.Gradient(iterate);
        gradient2 = lrf2.Gradient(iterate);
        Assert.assertEquals(gradient1.get(0, 0), gradient2.get(0, 0), 1e-3);
        Assert.assertEquals(gradient1.get(0, 1), gradient2.get(0, 1), 1e-3);
        Assert.assertEquals(gradient1.get(0, 2), gradient2.get(0, 2), 1e-3);

        P = new double[][]{{-5, 3, -7}};
        iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        gradient1 = lrf1.Gradient(iterate);
        gradient2 = lrf2.Gradient(iterate);
        Assert.assertEquals(gradient1.get(0, 0), gradient2.get(0, 0), 1e-3);
        Assert.assertEquals(gradient1.get(0, 1), gradient2.get(0, 1), 1e-3);
        Assert.assertEquals(gradient1.get(0, 2), gradient2.get(0, 2), 1e-3);
    }

    @Test
    public void LogisticRegressionFunctionSimpleSeparableGradientFitIntercept()
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

        double[][] L = {{0, 0, 1 ,1}};
        Matrix labels = Matrix.from2DArray(L);

        LogisticRegressionFunction lrf1 = new LogisticRegressionFunction(dataset1, labels);
        LogisticRegressionFunction lrf2 = new LogisticRegressionFunction(dataset2, labels, true);

        double[][] P = new double[][]{{1, 1, 1}};
        Matrix iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        Matrix gradient = lrf1.Gradient(iterate);
        Matrix separableGradient = lrf2.Gradient(iterate, 0, 1);
        separableGradient = separableGradient.add(lrf2.Gradient(iterate, 1, 1));
        separableGradient = separableGradient.add(lrf2.Gradient(iterate, 2, 1));
        separableGradient = separableGradient.divide(4);
        Assert.assertEquals(gradient.get(0, 0), separableGradient.get(0, 0), 1e-3);
        Assert.assertEquals(gradient.get(0, 1), separableGradient.get(0, 1), 1e-3);
        Assert.assertEquals(gradient.get(0, 2), separableGradient.get(0, 2), 1e-3);
    }
}
