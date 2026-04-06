\name{fastlpr_plot}
\alias{fastlpr_plot}
\title{Plot regression results}
\usage{
fastlpr_plot(fpp_yhat, x_range = NULL, n_points = 1000, ...)
}
\description{
Visualizes fitted regression results.
}
\arguments{
  \item{fpp_yhat}{Interpolant function or result object.}
  \item{x_range}{Optional range for plotting.}
  \item{n_points}{Number of points for plotting (default: 1000).}
  \item{...}{Additional plotting parameters.}
}
\value{
Called for its side effect (plotting). Returns \code{NULL} invisibly.
}
\seealso{
\code{\link{cv_fastlpr}}
}
\keyword{hplot}
\examples{
\donttest{
x <- matrix(runif(200), ncol = 1)
y <- sin(2 * pi * x) + rnorm(200, sd = 0.2)
fit <- cv_fastlpr(x, y)
fastlpr_plot(fit$fpp_yhat)
}
}
