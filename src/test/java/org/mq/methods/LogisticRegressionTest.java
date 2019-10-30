package org.mq.methods;

import org.junit.Assert;
import org.junit.Test;
import org.la4j.Matrix;
import org.mq.core.math.Math;
import org.mq.core.optimizers.sgd.SGD;
import org.mq.methods.logistic_regression.LogisticRegression;
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
        Assert.assertEquals(gradient.get(1, 0), 1.748763, 1e-3);
        Assert.assertEquals(gradient.get(2, 0), 2.248145, 1e-3);

        P = new double[][]{{9.382532, 0.88376, -7.615013}};
        iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        gradient = lrf.Gradient(iterate);
        Assert.assertEquals(gradient.get(0, 0), -0.002367, 1e-3);
        Assert.assertEquals(gradient.get(1, 0), 0.00022, 1e-3);
        Assert.assertEquals(gradient.get(2, 0), 0.001927, 1e-3);

        P = new double[][]{{-5, 3, -7}};
        iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        gradient = lrf.Gradient(iterate);
        Assert.assertEquals(gradient.get(0, 0), -4.25, 1e-3);
        Assert.assertEquals(gradient.get(1, 0), -4.75, 1e-3);
        Assert.assertEquals(gradient.get(2, 0), -5.25, 1e-3);
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
        Assert.assertEquals(gradient.get(1, 0), separableGradient.get(1, 0), 1e-3);
        Assert.assertEquals(gradient.get(2, 0), separableGradient.get(2, 0), 1e-3);
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
        Assert.assertEquals(gradient1.get(1, 0), gradient2.get(1, 0), 1e-3);
        Assert.assertEquals(gradient1.get(2, 0), gradient2.get(2, 0), 1e-3);

        P = new double[][]{{9.382532, 0.88376, -7.615013}};
        iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        gradient1 = lrf1.Gradient(iterate);
        gradient2 = lrf2.Gradient(iterate);
        Assert.assertEquals(gradient1.get(0, 0), gradient2.get(0, 0), 1e-3);
        Assert.assertEquals(gradient1.get(1, 0), gradient2.get(1, 0), 1e-3);
        Assert.assertEquals(gradient1.get(2, 0), gradient2.get(2, 0), 1e-3);

        P = new double[][]{{-5, 3, -7}};
        iterate = Matrix.from2DArray(P);
        iterate = iterate.transpose();
        gradient1 = lrf1.Gradient(iterate);
        gradient2 = lrf2.Gradient(iterate);
        Assert.assertEquals(gradient1.get(0, 0), gradient2.get(0, 0), 1e-3);
        Assert.assertEquals(gradient1.get(1, 0), gradient2.get(1, 0), 1e-3);
        Assert.assertEquals(gradient1.get(2, 0), gradient2.get(2, 0), 1e-3);
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
        Assert.assertEquals(gradient.get(1, 0), separableGradient.get(1, 0), 1e-3);
        Assert.assertEquals(gradient.get(2, 0), separableGradient.get(2, 0), 1e-3);
    }

    @Test
    public void LogisticRegressionFunctionInitializeWeights()
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

        Assert.assertEquals(lrf1.Evaluate(lrf1.initializeWeights()), lrf2.Evaluate(lrf2.initializeWeights()), 1e-3);
    }


    @Test
    public void LogisticRegressionSimpleTest()
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

        SGD sgd = new SGD(0.01, 10000, 1e-20, 1);

        LogisticRegression lr1 = new LogisticRegression(dataset1, labels, sgd);
        LogisticRegression lr2 = new LogisticRegression(dataset2, labels, sgd, true);

        Matrix pred1 = lr1.compute(dataset1);
        Matrix pred2 = lr2.compute(dataset2);

        Assert.assertEquals(pred1.columns(), 4);
        Assert.assertEquals(pred2.columns(), 4);

        Assert.assertEquals(pred1.rows(), 1);
        Assert.assertEquals(pred2.rows(), 1);

        for (int i = 0; i < pred1.rows(); i++)
        {
            for (int j = 0; j < pred1.columns(); j++)
            {
                Assert.assertEquals(pred1.get(i, j), labels.get(i, j), 1e-20);
                Assert.assertEquals(pred2.get(i, j), labels.get(i, j), 1e-20);
            }
        }
    }

    @Test
    public void LogisticRegressionComplexTest()
    {
        Matrix dataset = Matrix.fromCSV("1,34.62365962451697,78.0246928153624,0\n" +
                "1,30.28671076822607,43.89499752400101,0\n" +
                "1,35.84740876993872,72.90219802708364,0\n" +
                "1,60.18259938620976,86.30855209546826,1\n" +
                "1,79.0327360507101,75.3443764369103,1\n" +
                "1,45.08327747668339,56.3163717815305,0\n" +
                "1,61.10666453684766,96.51142588489624,1\n" +
                "1,75.02474556738889,46.55401354116538,1\n" +
                "1,76.09878670226257,87.42056971926803,1\n" +
                "1,84.43281996120035,43.53339331072109,1\n" +
                "1,95.86155507093572,38.22527805795094,0\n" +
                "1,75.01365838958247,30.60326323428011,0\n" +
                "1,82.30705337399482,76.48196330235604,1\n" +
                "1,69.36458875970939,97.71869196188608,1\n" +
                "1,39.53833914367223,76.03681085115882,0\n" +
                "1,53.9710521485623,89.20735013750205,1\n" +
                "1,69.07014406283025,52.74046973016765,1\n" +
                "1,67.94685547711617,46.67857410673128,0\n" +
                "1,70.66150955499435,92.92713789364831,1\n" +
                "1,76.97878372747498,47.57596364975532,1\n" +
                "1,67.37202754570876,42.83843832029179,0\n" +
                "1,89.67677575072079,65.79936592745237,1\n" +
                "1,50.534788289883,48.85581152764205,0\n" +
                "1,34.21206097786789,44.20952859866288,0\n" +
                "1,77.9240914545704,68.9723599933059,1\n" +
                "1,62.27101367004632,69.95445795447587,1\n" +
                "1,80.1901807509566,44.82162893218353,1\n" +
                "1,93.114388797442,38.80067033713209,0\n" +
                "1,61.83020602312595,50.25610789244621,0\n" +
                "1,38.78580379679423,64.99568095539578,0\n" +
                "1,61.379289447425,72.80788731317097,1\n" +
                "1,85.40451939411645,57.05198397627122,1\n" +
                "1,52.10797973193984,63.12762376881715,0\n" +
                "1,52.04540476831827,69.43286012045222,1\n" +
                "1,40.23689373545111,71.16774802184875,0\n" +
                "1,54.63510555424817,52.21388588061123,0\n" +
                "1,33.91550010906887,98.86943574220611,0\n" +
                "1,64.17698887494485,80.90806058670817,1\n" +
                "1,74.78925295941542,41.57341522824434,0\n" +
                "1,34.1836400264419,75.2377203360134,0\n" +
                "1,83.90239366249155,56.30804621605327,1\n" +
                "1,51.54772026906181,46.85629026349976,0\n" +
                "1,94.44336776917852,65.56892160559052,1\n" +
                "1,82.36875375713919,40.61825515970618,0\n" +
                "1,51.04775177128865,45.82270145776001,0\n" +
                "1,62.22267576120188,52.06099194836679,0\n" +
                "1,77.19303492601364,70.45820000180959,1\n" +
                "1,97.77159928000232,86.7278223300282,1\n" +
                "1,62.07306379667647,96.76882412413983,1\n" +
                "1,91.56497449807442,88.69629254546599,1\n" +
                "1,79.94481794066932,74.16311935043758,1\n" +
                "1,99.2725269292572,60.99903099844988,1\n" +
                "1,90.54671411399852,43.39060180650027,1\n" +
                "1,34.52451385320009,60.39634245837173,0\n" +
                "1,50.2864961189907,49.80453881323059,0\n" +
                "1,49.58667721632031,59.80895099453265,0\n" +
                "1,97.64563396007767,68.86157272420604,1\n" +
                "1,32.57720016809309,95.59854761387875,0\n" +
                "1,74.24869136721598,69.82457122657193,1\n" +
                "1,71.79646205863379,78.45356224515052,1\n" +
                "1,75.3956114656803,85.75993667331619,1\n" +
                "1,35.28611281526193,47.02051394723416,0\n" +
                "1,56.25381749711624,39.26147251058019,0\n" +
                "1,30.05882244669796,49.59297386723685,0\n" +
                "1,44.66826172480893,66.45008614558913,0\n" +
                "1,66.56089447242954,41.09209807936973,0\n" +
                "1,40.45755098375164,97.53518548909936,1\n" +
                "1,49.07256321908844,51.88321182073966,0\n" +
                "1,80.27957401466998,92.11606081344084,1\n" +
                "1,66.74671856944039,60.99139402740988,1\n" +
                "1,32.72283304060323,43.30717306430063,0\n" +
                "1,64.0393204150601,78.03168802018232,1\n" +
                "1,72.34649422579923,96.22759296761404,1\n" +
                "1,60.45788573918959,73.09499809758037,1\n" +
                "1,58.84095621726802,75.85844831279042,1\n" +
                "1,99.82785779692128,72.36925193383885,1\n" +
                "1,47.26426910848174,88.47586499559782,1\n" +
                "1,50.45815980285988,75.80985952982456,1\n" +
                "1,60.45555629271532,42.50840943572217,0\n" +
                "1,82.22666157785568,42.71987853716458,0\n" +
                "1,88.9138964166533,69.80378889835472,1\n" +
                "1,94.83450672430196,45.69430680250754,1\n" +
                "1,67.31925746917527,66.58935317747915,1\n" +
                "1,57.23870631569862,59.51428198012956,1\n" +
                "1,80.36675600171273,90.96014789746954,1\n" +
                "1,68.46852178591112,85.59430710452014,1\n" +
                "1,42.0754545384731,78.84478600148043,0\n" +
                "1,75.47770200533905,90.42453899753964,1\n" +
                "1,78.63542434898018,96.64742716885644,1\n" +
                "1,52.34800398794107,60.76950525602592,0\n" +
                "1,94.09433112516793,77.15910509073893,1\n" +
                "1,90.44855097096364,87.50879176484702,1\n" +
                "1,55.48216114069585,35.57070347228866,0\n" +
                "1,74.49269241843041,84.84513684930135,1\n" +
                "1,89.84580670720979,45.35828361091658,1\n" +
                "1,83.48916274498238,48.38028579728175,1\n" +
                "1,42.2617008099817,87.10385094025457,1\n" +
                "1,99.31500880510394,68.77540947206617,1\n" +
                "1,55.34001756003703,64.9319380069486,1\n" +
                "1,74.77589300092767,89.52981289513276,1");

        Matrix labels = dataset.slice(0, 3, dataset.rows(), 4).transpose();
        dataset = Math.Normalize(dataset.slice(0, 0, dataset.rows(), 3));

        SGD sgd = new SGD(0.01, 10000, 1e-5, 1);
        LogisticRegression lr = new LogisticRegression(dataset, labels, sgd, true);

        Matrix preds = lr.compute(dataset);
        int count = 0;
        for (int i = 0; i < preds.columns(); i++)
        {
            if (preds.get(0, i) == labels.get(0, i))
            {
                count++;
            }
        }

        Assert.assertTrue((double)count/dataset.rows() > 0.9);
    }
}
