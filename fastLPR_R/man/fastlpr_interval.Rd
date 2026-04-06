\name{fastlpr_interval}
\alias{fastlpr_interval}
\title{Compute Confidence or Prediction Intervals}
\usage{
fastlpr_interval(mu, sigma, alpha = 0.05, type = "confidence")
}
\description{
Computes pointwise confidence intervals for the conditional mean or
prediction intervals for new observations, using the mean and variance
estimates from local polynomial regression.
}
\arguments{
  \item{mu}{Mean regression result from \code{\link{cv_fastlpr}} (a list with field \code{fpp_yhat}).}
  \item{sigma}{Variance regression result from \code{\link{cv_fastlpr}} (a list with field \code{fpp_yhat}), or a numeric vector/matrix of standard deviations.}
  \item{alpha}{Significance level (default: 0.05 for 95\% intervals).}
  \item{type}{Type of interval: \code{"confidence"} for the conditional mean
    or \code{"prediction"} for new observations (default: \code{"confidence"}).}
}
\value{
List with lower and upper interval bounds.
}
\seealso{
\code{\link{cv_fastlpr}}, \code{\link{fastlpr_plot_interval}}
}
\keyword{regression}
\examples{
\donttest{
x <- matrix(runif(200), ncol = 1)
y <- sin(2 * pi * x) + rnorm(200, sd = 0.2)
mu <- cv_fastlpr(x, y)
sigma <- cv_fastlpr(x, (y - mu$yhat)^2)
ci <- fastlpr_interval(mu, sigma)
}
}
