#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void softmax_regression_one_batch(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    float *logits = new float[m * k];
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            logits[i * k + j] = 0;
            for (int l = 0; l < n; l++)
            {
                logits[i * k + j] += theta[l * k + j] * X[i * n + l];
            }
        }
    }
    float *probs = new float[m * k];
    for (int i = 0; i < m; i++)
    {
        float sum_probs = 0;
        for (int j = 0; j < k; j++)
        {
            probs[i * k + j] = exp(logits[i * k + j]);
            sum_probs += probs[i * k + j];
        }
        for (int j = 0; j < k; j++)
        {
            probs[i * k + j] /= sum_probs;
        }
    }
    // p - one_hot
    float *grad_logits = new float[m * k];
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            grad_logits[i * k + j] = probs[i * k + j];
            if (y[i] == j)
            {
                grad_logits[i * k + j] -= 1.0;
            }
        }
    }

    float *grad_theta = new float[n * k];
    for (int j = 0; j < n; j++)
    {
        for (int l = 0; l < k; l++)
        {
            grad_theta[j * k + l] = 0;
            for (int i = 0; i < m; i++)
            {
                grad_theta[j * k + l] += grad_logits[i * k + l] * X[i * n + j];
            }
        }
    }
    for (int i = 0; i < n * k; i++)
    {
        theta[i] -= lr * (grad_theta[i] /= m);
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */
    int num_batches = m / batch;
    for (int i = 0; i < num_batches; i++)
    {
        const float *X_batch = X + i * batch * n;
        const unsigned char *y_batch = y + i * batch;
        softmax_regression_one_batch(X_batch, y_batch, theta, batch, n, k, lr, batch);
    }

    /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m)
{
    m.def("softmax_regression_epoch_cpp", [](py::array_t<float, py::array::c_style> X, py::array_t<unsigned char, py::array::c_style> y, py::array_t<float, py::array::c_style> theta, float lr, int batch)
          { softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch); }, py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"), py::arg("batch"));
}
