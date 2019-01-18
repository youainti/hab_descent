functions {
// ... function declarations and definitions ...
}
data {
// ... declarations ...
  int<lower = 0> N;
  int<lower = 0> k;

  matrix[N,k] alt_X;
  vector[N] alt_y;
}
transformed data {
// ... declarations ... statements ...
}
parameters {
// ... declarations ...
  real alpha; //intercept
  vector[k] beta; //coefficient
  real<lower = 0> stdev; //standard devation
}


transformed parameters {
// ... declarations ... statements ...
}


model {
  alt_y ~ normal(alt_X * beta + alpha, stdev);
}


generated quantities {
// ... declarations ... statements ...
}
