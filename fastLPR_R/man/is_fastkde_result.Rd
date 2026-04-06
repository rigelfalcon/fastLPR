\name{is_fastkde_result}
\alias{is_fastkde_result}
\title{Check if Object is a fastkde_result}
\description{
Tests whether an object is of class \code{fastkde_result}.
}
\usage{
is_fastkde_result(x)
}
\arguments{
  \item{x}{An R object to test.}
}
\value{
\code{TRUE} if \code{x} inherits from class \code{"fastkde_result"},
\code{FALSE} otherwise.
}
\examples{
is_fastkde_result(list())  # FALSE
}
