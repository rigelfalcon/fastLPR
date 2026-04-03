\name{multispace}
\alias{multispace}
\title{Generate multi-dimensional grid}
\usage{
multispace(x_min, x_max, N, space = "linear", type = "array", transpose = FALSE)
}
\description{
Utility function to generate multi-dimensional grids.
}
\arguments{
  \item{x_min}{Minimum values per dimension.}
  \item{x_max}{Maximum values per dimension.}
  \item{N}{Grid size per dimension.}
  \item{space}{Grid spacing: "linear" (default) or "log".}
  \item{type}{Output type: "array" (default) or "meshgrid".}
  \item{transpose}{Whether to transpose output (default: FALSE).}
}
\value{
List of grid vectors.
}
\keyword{utilities}
