myinitHMMcont_Reg <- function (States, startCoefs, transCoefs, emissionCoefs, 
                               sds)
{
  nStates <- length(States)
  if (!is.matrix(startCoefs)) {
    stop("startCoefs is not a matrix!")
  }
  if (nrow(startCoefs) != nStates) {
    stop("startCoefs is not a matrix with nStates rows 
         (including reference row as 1st row)!")
  }
  if (!is.list(transCoefs)) {
    stop("transCoefs is not a list!")
  }
  if (length(transCoefs) != nStates) {
    stop("transCoefs is not a list of nStates elemnts!")
  }
  for (state in States) {
    mat <- transCoefs[[state]]
    if (!is.matrix(mat)) {
      stop(paste0("transCoefs for state", state, "is not a matrix!"))
    }
    if (nrow(mat) != nStates) {
      stop(paste0("transCoefs for state", state, "is not a matrix with 
      nStates rows (including reference row as 1st row).!"))
    }
  }
  if (!is.matrix(emissionCoefs)) {
    stop("emissionCoefs is not a matrix!")
  }
  if (nrow(emissionCoefs) != nStates) {
    stop("emissionCoefs is not a matrix with nStates rows!")
  }
  if (!is.vector(sds)) {
    stop("sds is not a vector!")
  }
  if (length(sds) != nStates) {
    stop("sds is not a vector of nStates elemnts!")
  }
  names(sds) <- States
  rownames(startCoefs) <- States
  rownames(emissionCoefs) <- States
  names(transCoefs) <- States
  for (from_state in States) {
    rownames(transCoefs[[from_state]]) <- States
  }
  emissionParams <- lapply(States, function(state) {
    list(coefs = emissionCoefs[state, ], sd = sds[state])
  })
  names(emissionParams) <- States
  return(list(
    States = States,
    startCoefs = startCoefs,
    transCoefs = transCoefs,
    emissionParams = emissionParams
  ))
}


softmax <- function (x)
{
  exp_x <- exp(x - max(x))
  exp_x / sum(exp_x)
}


myforwardcont_Reg <- function (hmm, observation, Xs, Xt, Xe)
{
  nObservations <- length(observation)
  nStates <- length(hmm$States)
  start_logits <- hmm$startCoefs %*% Xs
  start_probs <- softmax(start_logits)
  emission_means <- sapply(hmm$States, function(state) {
    as.vector(hmm$emissionParams[[state]]$coefs %*% Xe)
  })
  trans_probs_list <- vector("list", nObservations - 1)
  for (k in 1:(nObservations - 1)) {
    TM <- matrix(NA, nStates, nStates)
    dimnames(TM) <- list(from = hmm$States, to = hmm$States)
    for (state in hmm$States) {
      logits <- hmm$transCoefs[[state]] %*% Xt[, k]
      TM[state, ] <- as.vector(softmax(logits))
    }
    trans_probs_list[[k]] <- TM
  }
  f <- array(NA, c(nStates, nObservations))
  dimnames(f) <- list(states = hmm$States, index = 1:nObservations)
  for (state in hmm$States) {
    mu <- emission_means[1, state]
    sigma <- hmm$emissionParams[[state]]$sd
    f[state, 1] <- log(start_probs[state,]) + 
      dnorm(observation[1], mean = mu, sd = sigma, log = TRUE)
  }
  for (k in 2:nObservations) {
    for (state in hmm$States) {
      logsum <- -Inf
      for (previousState in hmm$States) {
        temp <- f[previousState, k - 1] + 
          log(trans_probs_list[[k - 1]][previousState, state])
        logsum <- max(temp, logsum) + log(1 + exp(-abs(temp - logsum)))
      }
      mu <- emission_means[k, state]
      sigma <- hmm$emissionParams[[state]]$sd
      f[state, k] <- dnorm(observation[k], mean = mu, 
                           sd = sigma, log = TRUE) + logsum
    }
  }
  return(f)
}


mybackwardcont_Reg <- function (hmm, observation, Xs, Xt, Xe)
{
  nObservations <- length(observation)
  nStates <- length(hmm$States)
  emission_means <- sapply(hmm$States, function (state) {
    as.vector(hmm$emissionParams[[state]]$coefs %*% Xe)
  })
  trans_probs_list <- vector("list", nObservations - 1)
  for (k in 1:(nObservations - 1)) {
    TM <- matrix(NA, nStates, nStates)
    dimnames(TM) <- list(from = hmm$States, to = hmm$States)
    for (state in hmm$States) {
      logits <- hmm$transCoefs[[state]] %*% Xt[, k]
      TM[state, ] <- as.vector(softmax(logits))
    }
    trans_probs_list[[k]] <- TM
  }
  b <- array(NA, c(nStates, nObservations))
  dimnames(b) <- list(states = hmm$States, index = 1:nObservations)
  for (state in hmm$States) {
    b[state, nObservations] = log(1)
  }
  for (k in (nObservations - 1):1) {
    for (state in hmm$States) {
      logsum <- -Inf
      for (nextState in hmm$States) {
        mu <- emission_means[k + 1, nextState]
        sigma <- hmm$emissionParams[[nextState]]$sd
        temp <- b[nextState, k + 1] +
          log(trans_probs_list[[k]][state, nextState]) +
          dnorm(observation[k + 1], mean = mu, sd = sigma, log = TRUE)
        logsum <- max(temp, logsum) + log(1 + exp(-abs(temp - logsum)))
      }
      b[state, k] <- logsum
    }
  }
  return(b)
}


log_sum_exp <- function(x)
{
  max_x <- max(x)
  max_x + log(sum(exp(x - max_x)))
}


myBWRcont_Reg <- function (hmm, obs_list, Xs_list, Xt_list, Xe_list) 
{
  nOBSseq <- length(obs_list)
  nStates <- length(hmm$States)
  logstart_Y <- matrix(nrow = nOBSseq, ncol = nStates)
  dimnames(logstart_Y) <- list(index = 1:nOBSseq, states = hmm$States)
  start_X <- do.call(cbind, Xs_list)
  logxi_list <- setNames(vector("list", nStates), hmm$States)
  logxi_list <- lapply(logxi_list, function(dummy) setNames(vector("list", nStates), hmm$States))
  trans_X <- do.call(cbind, Xt_list)
  loggamma_list <- setNames(vector("list", nStates), hmm$States)
  emission_Y <- unlist(obs_list)
  emission_X <- do.call(cbind, Xe_list)
  for (idx in 1:nOBSseq) {
    observation <- obs_list[[idx]]
    nObservations <- length(observation)
    Xs <- Xs_list[[idx]]
    Xt <- Xt_list[[idx]]
    Xe <- Xe_list[[idx]]
    f <- myforwardcont_Reg(hmm, observation, Xs, Xt, Xe)
    b <- mybackwardcont_Reg(hmm, observation, Xs, Xt, Xe)
    start_logits <- hmm$startCoefs %*% Xs
    start_probs <- softmax(start_logits)
    emission_means <- sapply(hmm$States, function(state) {
      as.vector(hmm$emissionParams[[state]]$coefs %*% Xe)
    })
    trans_probs_list <- vector("list", nObservations - 1)
    for (k in 1:(nObservations - 1)) {
      TM <- matrix(NA, nStates, nStates)
      dimnames(TM) <- list(from = hmm$States, to = hmm$States)
      for (state in hmm$States) {
        logits <- hmm$transCoefs[[state]] %*% Xt[, k]
        TM[state, ] <- as.vector(softmax(logits))
      }
      trans_probs_list[[k]] <- TM
    }
    likelihood <- f[hmm$States[1], nObservations]
    for (state in hmm$States[-1]) {
      j <- f[state, nObservations]
      likelihood <- max(likelihood, j) + log(1 + exp(-abs(likelihood - j)))
    }
    for (state in hmm$States) {
      logstart_Y[idx, state] <- f[state, 1] + b[state, 1] - likelihood
    }
    for (state in hmm$States) {
      for (i in 1:nObservations) {
        loggamma_list[[state]] <- c(loggamma_list[[state]], 
                                    f[state, i] + b[state, i] - likelihood)
      }
    }
    for (state in hmm$States) {
      for (nextState in hmm$States) {
        for (i in 1:(nObservations - 1)) {
          logxi_list[[state]][[nextState]] <- c(logxi_list[[state]][[nextState]],
            f[state, i] + log(trans_probs_list[[i]][state, nextState]) +
                  dnorm(observation[i + 1], 
                        mean = emission_means[i + 1, nextState], 
                        sd = hmm$emissionParams[[nextState]]$sd, log = TRUE) + 
                  b[nextState, i + 1] - likelihood)
        }
      }
    }
  }
  for (state in hmm$States[-1]) {
    if (nrow(logstart_Y) >= nrow(start_X)) {
      model <- glm((1 / (1 + exp(logstart_Y[, hmm$States[1]] - 
                                   logstart_Y[, state]))) ~ . - 1,
                   family = binomial, data = as.data.frame(t(start_X)))
      hmm$startCoefs[state, ] <- coef(model)
    } else {
      stop("too many predioctors for initial probabilities!")
    }
  }
  for (state in hmm$States) {
    for (nextState in hmm$States[-1]) {
      if (ncol(trans_X) >= nrow(trans_X)) {
        model <- glm((1 / (1 + exp(logxi_list[[state]][[hmm$States[1]]] - 
                                     logxi_list[[state]][[nextState]]))) ~ . - 1, 
                     family = binomial, 
                     data = as.data.frame(t(trans_X)))
        hmm$transCoefs[[state]][nextState, ] <- coef(model)
      } else {
        stop("too many predioctors for transition probabilities!")
      }
    }
  }
  for (state in hmm$States) {
    loggamma_shifted <- loggamma_list[[state]] - max(loggamma_list[[state]])
    if (sum(exp(loggamma_shifted)) > 0) {
      if (length(emission_Y) >= nrow(emission_X)) {
        model <- lm(emission_Y ~ . - 1, 
                    data = as.data.frame(t(emission_X)), 
                    weights = exp(loggamma_shifted))
        hmm$emissionParams[[state]]$coefs <- coef(model)
        hmm$emissionParams[[state]]$coefs[is.na(hmm$emissionParams[[state]]$coefs)] <- 0
        hmm$emissionParams[[state]]$sd <- sqrt(sum(exp(loggamma_shifted) * 
                                                residuals(model)^2) / 
          sum(exp(loggamma_shifted)))
      } else {
        stop("too many predioctors for emissions!")
      }
    } else {
      hmm$emissionParams[[state]]$coefs[] <- 0
      hmm$emissionParams[[state]]$sd <- 1
    }
    if (hmm$emissionParams[[state]]$sd < 1e-4) {
      hmm$emissionParams[[state]]$sd <- 1e-4
    }
  }
  return(hmm)
}


myBWcont_Reg <- function (hmm, obs_list, Xs_list, Xt_list, Xe_list, 
                          maxIterations = 100, delta = 1e-9, pseudoCount = 0)
{
  tempHmm <- hmm
  diff <- c()
  for (i in 1:maxIterations) {
    bw <- myBWRcont_Reg(tempHmm, obs_list, Xs_list, Xt_list, Xe_list)
    TC <- bw$transCoefs
    EP <- bw$emissionParams
    SC <- bw$startCoefs
    d <- sqrt(sum((do.call(rbind, tempHmm$transCoefs) - do.call(rbind, TC))^2)) + 
      sqrt(sum((unlist(tempHmm$emissionParams) - unlist(EP))^2)) + 
      sqrt(sum((tempHmm$startCoefs - SC)^2))
    diff <- c(diff, d)
    tempHmm$transProbs <- TC
    tempHmm$emissionParams <- EP
    tempHmm$startProbs <- SC
    if (d < delta) {
      break
    }
  }
  return(list(hmm = tempHmm, difference = diff))
}