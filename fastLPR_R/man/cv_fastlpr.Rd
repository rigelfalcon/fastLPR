\name{cv_fastlpr}
\alias{cv_fastlpr}
\title{Fast Local Polynomial Regression with automatic bandwidth selection}
\usage{
cv_fastlpr(x, y, h = NULL, opt = NULL)
}
\description{
This is the main function for fastLPR toolbox. It performs nonparametric
regression using kernel-weighted local polynomial methods with NUFFT
(Non-Uniform Fast Fourier Transform) acceleration. The function automatically
selects the optimal bandwidth via Generalized Cross-Validation (GCV) when
multiple bandwidth candidates are provided.
}
\arguments{
  \item{x}{N x d matrix of predictors (N samples, d dimensions).
    Can be real or complex-valued. Each row is one observation.
    For complex-valued predictors, use x = real + 1i*imag.}
  \item{y}{N x 1 vector of responses.
    Can be real or complex-valued. Must have same number of rows as x.}
  \item{h}{Bandwidth parameter(s) (optional, default: automatic selection).
    Scalar: same bandwidth for all dimensions.
    1 x d vector: different bandwidth per dimension.
    k x d matrix: grid of k bandwidth combinations for GCV selection.
    Use get_hlist() to generate bandwidth candidates.}
  \item{opt}{Options list (optional) with fields:
    \describe{
      \item{order}{Polynomial order (default: 0)
        0 = Nadaraya-Watson (local constant)
        1 = Local linear regression
        2 = Local quadratic regression}
      \item{kernel_type}{Kernel function (default: 'gaussian')
        'gaussian' or 'epanechnikov'}
      \item{dstd}{Number of DOF samples for variance estimation (default: 0)
        Set to 5-20 for heteroscedastic variance estimation}
      \item{y_type_out}{Output type (default: 'mean')
        'mean' = estimate conditional mean E[Y|X]
        'variance' = estimate conditional variance Var[Y|X]}
      \item{N}{Grid resolution for evaluation (default: auto)}
      \item{xrange}{Evaluation range (default: data range)}
    }}
}
\value{
Regression results list containing:
  \describe{
    \item{yhat}{Fitted values at evaluation grid points (N_grid x 1)}
    \item{fpp_yhat}{Interpolant object for prediction at any point.
      Use: y_pred = regs$fpp_yhat(x_new)}
    \item{gcv_yhat}{GCV results list (if multiple bandwidths provided)
      \describe{
        \item{h1se}{Selected bandwidth (1-SE rule)}
        \item{hmin}{Bandwidth with minimum GCV}
        \item{gcv}{GCV values for all bandwidths}
      }}
    \item{xq}{Evaluation grid points (list for multi-dimensional)}
    \item{xlist}{Grid vectors (list)}
    \item{opt}{Options used for regression}
    \item{xraw}{Original predictor data}
    \item{yraw}{Original response data}
  }
}
\examples{
\donttest{
# Example 1: 1D regression with automatic bandwidth selection
x <- matrix(runif(500) * 20, ncol = 1)
y <- sin(x) + 0.2 * rnorm(500)
hlist <- get_hlist(20, c(0.01, 1), "logspace")
opt <- list(order = 1)  # Local linear
regs <- cv_fastlpr(x, y, hlist, opt)

# Plot results
plot(x, y, pch = ".", col = "black")
# Note: fpp_yhat is an interpolation function
x_pred <- seq(min(x), max(x), length.out = 100)
y_pred <- regs$fpp_yhat(x_pred)
lines(x_pred, y_pred, col = "blue", lwd = 2)
}
}
\author{
Ying Wang, Min Li
}
\references{
Wang, Y., & Li, M. (2024). Fast and Exact Kernel-Weighted Regression
for Large-Scale Scattered and Complex-Valued Data.
Journal of Statistical Software (under review).
}
\seealso{
\code{\link{cv_fastkde}}, \code{\link{get_hlist}}, \code{\link{fastlpr_interval}}, \code{\link{fastlpr_plot}}
}
\keyword{nonparametric}
\keyword{regression}
\keyword{kernel smoothing}
\keyword{local polynomial}
\keyword{NUFFT}