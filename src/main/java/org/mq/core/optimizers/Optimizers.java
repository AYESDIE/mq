package org.mq.core.optimizers;

import org.la4j.Matrix;
import org.mq.core.optimizers.functions.DifferentiableFunction;

public class Optimizers {
    public <DifferentiableFunctionType extends DifferentiableFunction> Matrix Optimize(
            DifferentiableFunctionType function,
            Matrix iterate)
    {
        return iterate;
    }
}
