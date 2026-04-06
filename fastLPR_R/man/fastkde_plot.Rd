\name{fastkde_plot}
\alias{fastkde_plot}
\title{Plot KDE results}
\usage{
fastkde_plot(fpp, x_range = NULL, n_points = 1000, ...)
}
\description{
Visualizes kernel density estimation results.
}
\arguments{
  \item{fpp}{KDE interpolator from cv_fastkde.}
  \item{x_range}{Optional range for plotting.}
  \item{n_points}{Number of points for plotting (default: 1000).}
  \item{...}{Additional plotting parameters.}
}
\value{
Called for its side effect (plotting). Returns \code{NULL} invisibly.
}
\seealso{
\code{\link{cv_fastkde}}
}
\keyword{hplot}
\examples{
\donttest{
x <- matrix(rnorm(200), ncol = 1)
kde <- cv_fastkde(x)
fastkde_plot(kde$fpp)
}
}
