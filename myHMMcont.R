myinitHMMcont <- function (States, startProbs = NULL, transProbs = NULL, 
                           emissionParams = NULL)
{
  nStates <- length(States)
  SV <- rep(1 / nStates, nStates)
  names(SV) <- States
  TM <- 0.5 * diag(nStates) + array(0.5 / nStates, c(nStates, nStates))
  dimnames(TM) <- list(from = States, to = States)
  if (is.null(emissionParams)) {
    emissionParams <- lapply(States, function(state) {
      list(mean = 0, sd = 1)
    })
    names(emissionParams) <- States
  }
  if (!is.null(startProbs)) {
    SV[] <- startProbs[]
  }
  if (!is.null(transProbs)) {
    TM[,] <- transProbs[,]
  }
  return(list(
    States = States,
    startProbs = SV,
    transProbs = TM,
    emissionParams = emissionParams
  ))
}


myforwardcont <- function (hmm, observation)
{
  nObservations <- length(observation)
  nStates <- length(hmm$States)
  hmm$transProbs[is.na(hmm$transProbs)] <- 0
  f <- array(NA, c(nStates, nObservations))
  dimnames(f) <- list(states = hmm$States, index = 1:nObservations)
  for (state in hmm$States) {
    mu <- hmm$emissionParams[[state]]$mean
    sigma <- hmm$emissionParams[[state]]$sd
    f[state, 1] <- log(hmm$startProbs[state]) + 
      dnorm(observation[1], mean = mu, sd = sigma, log = TRUE)
  }
  for (k in 2:nObservations) {
    for (state in hmm$States) {
      logsum <- -Inf
      for (previousState in hmm$States) {
        temp <- f[previousState, k - 1] + 
          log(hmm$transProbs[previousState, state])
        logsum <- max(temp, logsum) + log(1 + exp(-abs(temp - logsum)))
      }
      mu <- hmm$emissionParams[[state]]$mean
      sigma <- hmm$emissionParams[[state]]$sd
      f[state, k] <- dnorm(observation[k], mean = mu, sd = sigma, log = TRUE) + 
        logsum
    }
  }
  return(f)
}


mybackwardcont <- function (hmm, observation)
{
  nObservations <- length(observation)
  nStates <- length(hmm$States)
  hmm$transProbs[is.na(hmm$transProbs)] <- 0
  b <- array(NA, c(nStates, nObservations))
  dimnames(b) <- list(states = hmm$States, index = 1:nObservations)
  for (state in hmm$States) {
    b[state, nObservations] = log(1)
  }
  for (k in (nObservations - 1):1) {
    for (state in hmm$States) {
      logsum <- -Inf
      for (nextState in hmm$States) {
        mu <- hmm$emissionParams[[nextState]]$mean
        sigma <- hmm$emissionParams[[nextState]]$sd
        temp <- b[nextState, k + 1] + log(hmm$transProbs[state, nextState]) + 
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


myBWRcont <- function (hmm, obs_list) {
  nOBSseq <- length(obs_list)
  nStates <- length(hmm$States)
  logstart_Y <- matrix(nrow = nOBSseq, ncol = nStates)
  dimnames(logstart_Y) <- list(index = 1:nOBSseq, states = hmm$States)
  TransitionMatrix <- hmm$transProbs
  TransitionMatrix[,] <- 0
  logxi_list <- setNames(vector("list", nStates), hmm$States)
  logxi_list <- lapply(logxi_list, function(dummy) setNames(vector("list", nStates), hmm$States))
  EmissionList <- hmm$emissionParams
  EmissionList[] <- lapply(EmissionList, function(y) { y[] <- 0; y })
  loggamma_list <- setNames(vector("list", nStates), hmm$States)
  for (idx in 1:nOBSseq) {
    observation <- obs_list[[idx]]
    nObservations <- length(observation)
    f <- myforwardcont(hmm, observation)
    b <- mybackwardcont(hmm, observation)
    likelihood <- f[hmm$States[1], nObservations]
    for (state in hmm$States[-1]) {
      j <- f[state, nObservations]
      likelihood <- max(likelihood, j) + log(1 + exp(-abs(likelihood - j)))
    }
    for (state in hmm$States) {
      for (nextState in hmm$States) {
        for (i in 1:(nObservations - 1)) {
          logxi_list[[state]][[nextState]] <- c(logxi_list[[state]][[nextState]], 
            f[state, i] + log(hmm$transProbs[state, nextState]) + 
                  dnorm(observation[i + 1], 
                        mean = hmm$emissionParams[[nextState]]$mean, 
                        sd = hmm$emissionParams[[nextState]]$sd, log = TRUE) + 
                  b[nextState, i + 1] - likelihood)
        }
      }
    }
    for (state in hmm$States) {
      loggamma_list[[state]] <- c(loggamma_list[[state]], 
                               f[state, ] + b[state, ] - likelihood)
    }
    for (state in hmm$States) {
      logstart_Y[idx, state] <- f[state, 1] + b[state, 1] - likelihood
    }
  }
  for (state in hmm$States) {
    for (nextState in hmm$States) {
      TransitionMatrix[state, nextState] <- log_sum_exp(logxi_list[[state]][[nextState]]) - 
        log_sum_exp(unlist(logxi_list[[state]]))
    }
  }
  TransitionMatrix <- exp(TransitionMatrix)
  for (state in hmm$States) {
    loggamma_shifted <- loggamma_list[[state]] - max(loggamma_list[[state]])
    if (sum(exp(loggamma_shifted)) > 0) {
      EmissionList[[state]]$mean <- sum(exp(loggamma_shifted) * 
                                          unlist(obs_list)) / 
        sum(exp(loggamma_shifted))
      EmissionList[[state]]$sd <- sqrt(sum(exp(loggamma_shifted) * 
                                             unlist(obs_list)^2) / 
                                         sum(exp(loggamma_shifted)) - 
                                         (EmissionList[[state]]$mean)^2)
    } else {
      EmissionList[[state]]$mean <- 0
      EmissionList[[state]]$sd <- 1
    }
    if (EmissionList[[state]]$sd < 1e-4) {
      EmissionList[[state]]$sd <- 1e-4
    }
  }
  return(list(
    TransitionMatrix = TransitionMatrix,
    EmissionList = EmissionList,
    startVec = exp(apply(logstart_Y, 2, log_sum_exp))))
}


myBWcont <- function (hmm, obs_list, maxIterations = 100, delta = 1e-9, 
                      pseudoCount = 0)
{
  tempHmm <- hmm
  tempHmm$transProbs[is.na(hmm$transProbs)] <- 0
  diff <- c()
  for (i in 1:maxIterations) {
    bw <- myBWRcont(tempHmm, obs_list)
    TM <- bw$TransitionMatrix
    EL <- bw$EmissionList
    SV <- bw$startVec
    TM[!is.na(hmm$transProbs)] <- TM[!is.na(hmm$transProbs)] + pseudoCount
    SV[!is.na(hmm$startProbs)] <- SV[!is.na(hmm$startProbs)] + pseudoCount
    TM <- TM / rowSums(TM)
    SV <- SV / sum(SV)
    d <- sqrt(sum((tempHmm$transProbs - TM)^2)) + 
      sqrt(sum((unlist(tempHmm$emissionParams) - unlist(EL))^2)) + 
      sqrt(sum((tempHmm$startProbs - SV)^2))
    diff <- c(diff, d)
    tempHmm$transProbs <- TM
    tempHmm$emissionParams <- EL
    tempHmm$startProbs <- SV
    if (d < delta) {
      break
    }
  }
  tempHmm$transProbs[is.na(hmm$transProbs)] <- NA
  tempHmm$startProbs[is.na(hmm$startProbs)] <- NA
  return(list(hmm = tempHmm, difference = diff))
}