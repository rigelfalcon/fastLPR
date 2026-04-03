\name{fastlpr_interval}
\alias{fastlpr_interval}
\title{Compute confidence intervals}
\usage{
fastlpr_interval(mu, sigma, alpha = 0.05)
}
\description{
Computes confidence intervals using mean and variance estimates.
}
\arguments{
  \item{mu}{Mean regression results.}
  \item{sigma}{Variance regression results.}
  \item{alpha}{Significance level (default: 0.05).}
}
\value{
List with lower and upper confidence bounds.
}
\seealso{
\code{\link{cv_fastlpr}}
}
\keyword{regression}
