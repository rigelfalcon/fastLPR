\name{fastlpr_predict}
\alias{fastlpr_predict}
\title{Predict at new points}
\usage{
fastlpr_predict(regs, x_new)
}
\description{
Evaluates fitted regression at new predictor values.
}
\arguments{
  \item{regs}{Regression results from cv_fastlpr.}
  \item{x_new}{New predictor values (matrix).}
}
\value{
Predicted response values.
}
\seealso{
\code{\link{cv_fastlpr}}
}
\keyword{regression}
\examples{
\donttest{
x <- matrix(runif(200), ncol = 1)
y <- sin(2 * pi * x) + rnorm(200, sd = 0.2)
fit <- cv_fastlpr(x, y)
x_new <- matrix(seq(0, 1, length.out = 50), ncol = 1)
yhat <- fastlpr_predict(fit, x_new)
}
}
