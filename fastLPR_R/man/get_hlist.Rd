\name{get_hlist}
\alias{get_hlist}
\title{Generate bandwidth candidates for cross-validation}
\usage{
get_hlist(n, range, spacing = "logspace")
}
\description{
Creates a grid of bandwidth candidates for cross-validation.
Port from MATLAB's get_hlist.m (unified API v2.0).
}
\arguments{
  \item{n}{Number of bandwidth candidates per dimension.
    Scalar: same number of points for all dimensions.
    Vector: specify number of points per dimension.
    Typical values: 20 for 1D, c(15, 15) for 2D.}
  \item{range}{Range of bandwidths (min, max) for each dimension.
    1D: c(h_min, h_max) or matrix with 1 row.
    Multi-D: matrix with one row per dimension.
    Rule of thumb: h_min = 0.1 * sd(x), h_max = 1.0 * sd(x).}
  \item{spacing}{Spacing function type (default: "logspace").
    "logspace": logarithmic spacing (better for exploring bandwidth scales).
    "linear": linear spacing.}
}
\value{
Matrix of bandwidth candidates (n_total x d).
Each row is one bandwidth combination.
}
\examples{
# 1D: 20 bandwidths from 0.01 to 1 (log scale, default)
hlist <- get_hlist(20, c(0.01, 1))

# 2D: 10x10 grid of bandwidths
hlist <- get_hlist(10, rbind(c(0.01, 1), c(0.01, 1)))

# Linear spacing
hlist <- get_hlist(10, c(0.5, 0.6), "linear")
}
\seealso{
\code{\link{cv_fastlpr}}, \code{\link{cv_fastkde}}
}
\keyword{utilities}
