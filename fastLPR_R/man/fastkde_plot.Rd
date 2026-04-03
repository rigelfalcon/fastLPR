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
\seealso{
\code{\link{cv_fastkde}}
}
\keyword{hplot}
