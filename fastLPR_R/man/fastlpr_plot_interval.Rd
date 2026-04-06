\name{fastlpr_plot_interval}
\alias{fastlpr_plot_interval}
\title{Plot Confidence or Prediction Interval Bands}
\description{
Visualizes confidence or prediction interval bands for local polynomial
regression estimates. Adds shaded interval regions to an existing plot.
}
\usage{
fastlpr_plot_interval(ci, col = "green", alpha = 0.2, add = TRUE, ...)
}
\arguments{
  \item{ci}{An interval structure returned by \code{\link{fastlpr_interval}}.}
  \item{col}{Color for the interval band (default: \code{"green"}).}
  \item{alpha}{Transparency level for the band (default: 0.2).}
  \item{add}{Logical; if \code{TRUE} (default), add to existing plot.}
  \item{\dots}{Additional arguments passed to plotting functions.}
}
\value{
Called for its side effect (plotting). Returns \code{NULL} invisibly.
}
\seealso{
\code{\link{fastlpr_interval}}, \code{\link{cv_fastlpr}}
}
\examples{
\donttest{
set.seed(42)
x <- sort(runif(200))
y <- sin(2 * pi * x) + rnorm(200, sd = 0.3)
hlist <- get_hlist(20, c(0.01, 0.5))
regs <- cv_fastlpr(x, y, hlist)
}
}
