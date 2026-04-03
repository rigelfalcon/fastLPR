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
