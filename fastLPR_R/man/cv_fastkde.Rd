\name{cv_fastkde}
\alias{cv_fastkde}
\title{Fast Kernel Density Estimation with automatic bandwidth selection}
\usage{
cv_fastkde(x, h = NULL, opt = NULL)
}
\description{
Performs kernel density estimation using NUFFT acceleration with automatic
bandwidth selection via Likelihood Cross-Validation (LCV).
}
\arguments{
  \item{x}{N x d matrix of data points (N samples, d dimensions).}
  \item{h}{Bandwidth parameter(s) or grid for LCV selection.}
  \item{opt}{Options list with fields similar to cv_fastlpr.}
}
\value{
KDE results list containing estimated density, evaluation grid, and LCV results.
}
\seealso{
\code{\link{cv_fastlpr}}, \code{\link{get_hlist}}
}
\keyword{nonparametric}
\keyword{density estimation}
\examples{
\donttest{
x <- matrix(rnorm(200), ncol = 1)
kde <- cv_fastkde(x)
}
}
