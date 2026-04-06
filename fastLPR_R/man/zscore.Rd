\name{zscore}
\alias{zscore}
\title{Z-score normalization}
\usage{
zscore(x)
}
\description{
Standardizes data to zero mean and unit variance.
}
\arguments{
  \item{x}{Numeric vector or matrix.}
}
\value{
Standardized data with attributes for mean and standard deviation.
}
\keyword{utilities}
\examples{
result <- zscore(rnorm(100, mean = 5, sd = 2))
mean(result$z)  # approximately 0
}
