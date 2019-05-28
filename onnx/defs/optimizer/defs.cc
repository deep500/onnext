// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
static const char* Adam_ver11_doc = R"DOC(
    Compute one iteration of Adam, a stochastic gradient based optimization
    algorithm. This operator can conduct the optimization of multiple tensor variables.

    Let's define the behavior of this operator. As you can imagine, ADAGRAD requires
    some parameters:
     
     - The initial learning-rate "R".
     - The update count "T". That is, the number of training iterations conducted.
     - A L2-norm regularization coefficient "lambda".
     - A small constant "epsilon" to avoid dividing-by-zero. 
     - Two coefficients, alpha and beta. 

    At each Adam iteration, the optimized tensors are moved along a direction
    computed based on their exponentially-averaged historical gradient and
    exponentially-averaged historical squared gradient. Assume that only a single
    tensor "X" is being optimized. We need
    
     - the value of "X", 
     - "X"'s gradient (denoted by "G"),
     - "X"'s exponentially-averaged historical gradient (denoted by "M"), and
     - "X"'s exponentially-averaged historical squared gradient (denoted by "V").

    Consequently, this operator's input tensor list is ["R," "T," "X," "G," "M," "V"].
    Other parameters are given as attributes because they are usually constants.
    Moreover, the corresponding output tensors are 
    
     - the new value of "X" (called "X_new"),
     - the new exponentially-averaged historical gradient (denoted by "M_new"), and
     - the new exponentially-averaged historical squared gradient (denoted by "V_new").

    Those outputs are computed following the pseudo code below.

    Let "+", "-", "*", and "/" are all element-wise arithmetic operations with
    numpy-style broadcasting support. The pseudo code to compute those outputs is:

      // Add gradient of 0.5 * lambda * ||X||_2^2, where ||X||_2 is the 2-norm.
      G_regularized = lambda * X + G;

      // Update exponentially-averaged historical gradient.
      M_new = alpha * M + (1 - alpha) * G_regularized;

      // Update exponentially-averaged historical squared gradient.
      V_new = beta * V + (1 - beta) * G_regularized * G_regularized;

      // The gradient will be element-wisely divided by the following tensor.
      H = Sqrt(V) + epsilon;

      // Compute learning-rate. Note that "alpha^T"/"beta^T" is alpha's/beta's T-th power.
      R_ = R * Sqrt(1 - beta^T) / (1 - alpha^T);

      // Compute new value of "X."
      X_new = X - R_ * M / H

    If there are multiple inputs to be optimized, the pseudo code will be applied
    independently to each of them.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Adam,
    11,
    OpSchema()
        .SetDoc(Adam_ver11_doc)
        .Input(0, "R", "The initial learning rate.", "T1")
        .Input(1, "T", "The update count of \"X\". It should be a scalar.", "T2")
        .Input(
            2,
            "inputs",
            "It sequentially contains the tensors to be optimized, the "
            "averaged gradient, and the averaged squared gradient. For example, to optimize "
            "tensors \"X_1\" and \"X_2,\" The input list would be [\"X_1\", \"X_2\", "
            "averaged gradient of \"X_1\", averaged gradient of \"X_2\", "
            "averaged squared gradient of \"X_1\", averaged squared gradient of \"X_2\"].",
            "T3",
            OpSchema::Variadic,
            false)
        .Output(
            0,
            "outputs",
            "It sequentially contains the new values of optimized tensors, then the new "
            "values of averaged gradient, and finally values of averaged squared gradient. For example, "
            "if two tensor \"X_1\" and \"X_2\" are optimized, the output list would be "
            "[new value of \"X_1,\", new value of \"X_2,\" new averaged gradient of \"X_1\", "
            "new averaged gradient of \"X_2,\" new averaged squared gradient of \"X_1,\" "
            "new averaged squared gradient of \"X_2\"].",
            "T3",
            OpSchema::Variadic,
            false)
        .Attr(
            "alpha",
            "Coefficient of previous gradient in running average.",
            AttributeProto::FLOAT,
            0.9f)
        .Attr(
            "beta",
            "Coefficient of previous squared gradient in running average."
            "The effective learning rate is computed by r = R / (1 + T * decay_factor). "
            "Default to 0 so that increasing update counts doesn't reduce the learning rate.",
            AttributeProto::FLOAT,
            0.999f)
        .Attr(
            "lambda",
            "Regularization coefficient of 0.5 * lambda * ||X||_2^2. Default to 0, "
            "which means no regularization.",
            AttributeProto::FLOAT,
            0.0f)
        .Attr(
            "epsilon",
            "Small scalar to avoid dividing by zero.",
            AttributeProto::FLOAT,
            1e-6f)
        .TypeConstraint(
            "T1",
            {"tensor(float)", "tensor(double)"},
            "Constrain input types to float scalars.")
        .TypeConstraint(
            "T2",
            {"tensor(int64)"},
            "Constrain output types to 64-bit integer scalars.")
        .TypeConstraint(
            "T3",
            {"tensor(float)", "tensor(double)"},
            "Constrain input types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext &ctx) {
        }}));
} // namespace ONNX_NAMESPACE
