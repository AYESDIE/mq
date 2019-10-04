package org.mq.core.optimizers.gradient_descent;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.la4j.LinearAlgebra;
import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.DenseMatrix;
import java.lang.Math;
import org.mq.core.optimizers.functions.DifferentiableFunction;

public class GradientDescent
{
    public GradientDescent()
    {
        stepSize = 0.01;
        maxIterations = 100000;
        tolerance = 1e-5;
    }

    public GradientDescent(double stepSize)
    {
        this.stepSize = stepSize;
        maxIterations = 100000;
        tolerance = 1e-5;
    }

    public GradientDescent(double stepSize,
                           long maxIterations,
                           double tolerance)
    {
        this.stepSize = stepSize;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
    }

    public <DifferentiableFunctionType extends DifferentiableFunction> Matrix Optimize(
            DifferentiableFunctionType function,
            Matrix iterate)
    {
        // Set the maximum value for the objectives.
        double overallObjective = Double.MAX_VALUE;
        double lastObjective =  Double.MAX_VALUE;

        for (long i = 1; i < maxIterations; i++)
        {
            overallObjective = function.Evaluate(iterate);

            if (Double.isNaN(overallObjective) || Double.isInfinite(overallObjective))
            {
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

            System.out.print("Loss: ");
            System.out.println(overallObjective);

            Matrix gradient;
            gradient = function.Gradient(iterate);

            iterate = iterate.subtract(gradient.multiply(stepSize));

            lastObjective = overallObjective;
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
    private double loss;
}