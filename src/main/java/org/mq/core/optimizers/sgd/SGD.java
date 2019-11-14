package org.mq.core.optimizers.sgd;

import org.la4j.Matrix;
import java.lang.Math;
import org.mq.core.optimizers.functions.DifferentiableFunction;

public class SGD
{
    public SGD()
    {
        stepSize = 0.01;
        maxIterations = 100000;
        tolerance = 1e-5;
        batchSize = 1;
    }

    public SGD(double stepSize)
    {
        this.stepSize = stepSize;
        maxIterations = 100000;
        tolerance = 1e-5;
        batchSize = 1;
    }

    public SGD(double stepSize,
               long maxIterations,
               double tolerance,
               int batchSize)
    {
        this.stepSize = stepSize;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.batchSize = batchSize;
    }

    public <DifferentiableFunctionType extends DifferentiableFunction> Matrix Optimize(
            DifferentiableFunctionType function,
            Matrix iterate)
    {
        long numFunctions = function.numFunctions();

        int currentFunction = 0;
        double overallObjective = 0;
        double lastObjective = Double.MAX_VALUE;

        for (int i = 0; i < numFunctions; i ++)
        {
            overallObjective += function.Evaluate(iterate, i, batchSize);
        }

        for (long i = 1; i < maxIterations; i++, currentFunction++)
        {
            if ((i % (maxIterations / 10)) == 0)
            {
                System.out.print("SGD: iteration ");
                System.out.print(i);
                System.out.print(", objective ");
                System.out.print(overallObjective);
                System.out.println(".");
            }


            if ((currentFunction % numFunctions) == 0)
            {
                if (Double.isNaN(overallObjective) || Double.isInfinite(overallObjective)) {

                    System.out.print("Gradient Descent: converged to ");
                    System.out.print(overallObjective);
                    System.out.println("; terminating with failure.  Try a smaller step size?");

                    loss = overallObjective;
                    return iterate;
                }

                if (Math.abs(lastObjective - overallObjective) < tolerance)
                {
                    System.out.print("Gradient Descent: minimized within tolerance ");
                    System.out.print(overallObjective);
                    System.out.println("; terminating optimization.");

                    loss = overallObjective;
                    return iterate;
                }

                lastObjective = overallObjective;
                overallObjective = 0;
                currentFunction = 0;
            }

            Matrix gradient;
            gradient = function.Gradient(iterate, currentFunction, batchSize);

            iterate = iterate.subtract(gradient.multiply(stepSize));

            overallObjective += function.Evaluate(iterate, currentFunction, batchSize);
        }

        System.out.println("Gradient Descent: maximum iterations ( ");
        System.out.print(maxIterations);
        System.out.print(") reached; terminating optimization.");

        loss = overallObjective;
        return iterate;
    }



    private double stepSize;
    private long maxIterations;
    private double tolerance;
    private int batchSize;
    private double loss;
}
