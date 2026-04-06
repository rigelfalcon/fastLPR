\name{fastkde_plot_bandwidth}
\alias{fastkde_plot_bandwidth}
\title{Plot Bandwidth Selection Diagnostics for KDE}
\description{
Visualizes the bandwidth selection process for kernel density estimation,
showing the LCV score as a function of bandwidth with the selected optimum marked.
}
\usage{
fastkde_plot_bandwidth(kde, main = "Bandwidth Selection (LCV)",
                       xlab = NULL, ylab = NULL, ...)
}
\arguments{
  \item{kde}{A \code{fastkde_result} object returned by \code{\link{cv_fastkde}}.}
  \item{main}{Title for the plot.}
  \item{xlab}{Label for the x-axis (default: auto-generated).}
  \item{ylab}{Label for the y-axis (default: auto-generated).}
  \item{\dots}{Additional arguments passed to \code{plot}.}
}
\value{
Called for its side effect (plotting). Returns \code{NULL} invisibly.
}
\examples{
\donttest{
x <- matrix(rnorm(500), ncol = 1)
hlist <- get_hlist(20, c(0.05, 2))
kde <- cv_fastkde(x, hlist)
fastkde_plot_bandwidth(kde)
}
}
