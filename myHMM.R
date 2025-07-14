myinitHMM <- function (States, Symbols, startProbs = NULL, transProbs = NULL, 
                       emissionProbs = NULL) 
{
  nStates <- length(States)
  nSymbols <- length(Symbols)
  SV <- rep(1/nStates, nStates)
  TM <- 0.5 * diag(nStates) + array(0.5/(nStates), c(nStates, nStates))
  EM <- array(1/(nSymbols), c(nStates, nSymbols))
  names(SV) <- States
  dimnames(TM) <- list(from = States, to = States)
  dimnames(EM) <- list(states = States, symbols = Symbols)
  if (!is.null(startProbs)) {
    SV[] <- startProbs[]
  }
  if (!is.null(transProbs)) {
    TM[, ] <- transProbs[, ]
  }
  if (!is.null(emissionProbs)) {
    EM[, ] <- emissionProbs[, ]
  }
  return(list(States = States, Symbols = Symbols, startProbs = SV, 
              transProbs = TM, emissionProbs = EM))
}


myforward <- function (hmm, observation) 
{
  nObservations <- length(observation)
  nStates <- length(hmm$States)
  hmm$transProbs[is.na(hmm$transProbs)] <- 0
  hmm$emissionProbs[is.na(hmm$emissionProbs)] <- 0
  f <- array(NA, c(nStates, nObservations))
  dimnames(f) <- list(states = hmm$States, index = 1:nObservations)
  for (state in hmm$States) {
    f[state, 1] <- log(hmm$startProbs[state] * 
                         hmm$emissionProbs[state, observation[1]])
  }
  for (k in 2:nObservations) {
    for (state in hmm$States) {
      logsum <- -Inf
      for (previousState in hmm$States) {
        temp <- f[previousState, k - 1] + 
          log(hmm$transProbs[previousState, state])
        logsum <- max(logsum, temp) + log(1 + exp(-abs(logsum - temp)))
      }
      f[state, k] <- log(hmm$emissionProbs[state, observation[k]]) + 
        logsum
    }
  }
  return(f)
}


mybackward <- function (hmm, observation) 
{
  nObservations <- length(observation)
  nStates <- length(hmm$States)
  hmm$transProbs[is.na(hmm$transProbs)] <- 0
  hmm$emissionProbs[is.na(hmm$emissionProbs)] <- 0
  b <- array(NA, c(nStates, nObservations))
  dimnames(b) <- list(states = hmm$States, index = 1:nObservations)
  for (state in hmm$States) {
    b[state, nObservations] <- log(1)
  }
  for (k in (nObservations - 1):1) {
    for (state in hmm$States) {
      logsum <- -Inf
      for (nextState in hmm$States) {
        temp <- b[nextState, k + 1] + 
          log(hmm$transProbs[state, nextState]) + 
          log(hmm$emissionProbs[nextState, observation[k + 1]])
        logsum <- max(logsum, temp) + log(1 + exp(-abs(logsum - temp)))
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


myBWR <- function (hmm, obs_list) 
{
  nOBSseq <- length(obs_list)
  nStates <- length(hmm$States)
  logstart_Y <- matrix(nrow = nOBSseq, ncol = nStates)
  dimnames(logstart_Y) <- list(index = 1:nOBSseq, states = hmm$States)
  TransitionMatrix <- hmm$transProbs
  TransitionMatrix[, ] <- 0
  logxi_list <- setNames(vector("list", nStates), hmm$States)
  logxi_list <- lapply(logxi_list, function(dummy) setNames(vector("list", nStates), hmm$States))
  EmissionMatrix <- hmm$emissionProbs
  EmissionMatrix[, ] <- 0
  loggamma_list <- setNames(vector("list", nStates), hmm$States)
  for (idx in 1:nOBSseq) {
    observation <- obs_list[[idx]]
    nObservations <- length(observation)
    f <- myforward(hmm, observation)
    b <- mybackward(hmm, observation)
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
                  log(hmm$emissionProbs[nextState, observation[i + 1]]) + 
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
    for (s in hmm$Symbols) {
      EmissionMatrix[state, s] <- log_sum_exp(loggamma_list[[state]][unlist(obs_list) == s]) - 
        log_sum_exp(loggamma_list[[state]])
    }
  }
  EmissionMatrix <- exp(EmissionMatrix)
  return(list(TransitionMatrix = TransitionMatrix,
              EmissionMatrix = EmissionMatrix,
              startVec = exp(apply(logstart_Y, 2, log_sum_exp))))
}


myBW <- function (hmm, obs_list, maxIterations = 100, delta = 1e-9, 
                  pseudoCount = 0) 
{
  tempHmm <- hmm
  tempHmm$transProbs[is.na(hmm$transProbs)] <- 0
  tempHmm$emissionProbs[is.na(hmm$emissionProbs)] <- 0
  diff <- c()
  for (i in 1:maxIterations) {
    bw <- myBWR(tempHmm, obs_list)
    TM <- bw$TransitionMatrix
    EM <- bw$EmissionMatrix
    SV <- bw$startVec
    TM[!is.na(hmm$transProbs)] <- TM[!is.na(hmm$transProbs)] + pseudoCount
    EM[!is.na(hmm$emissionProbs)] <- EM[!is.na(hmm$emissionProbs)] + pseudoCount
    SV[!is.na(hmm$startProbs)] <- SV[!is.na(hmm$startProbs)] + pseudoCount
    TM <- (TM/rowSums(TM))
    EM <- (EM/rowSums(EM))
    SV <- (SV/sum(SV))
    d <- sqrt(sum((tempHmm$transProbs - TM)^2)) + 
      sqrt(sum((tempHmm$emissionProbs - EM)^2)) + 
      sqrt(sum((tempHmm$startProbs - SV)^2))
    diff <- c(diff, d)
    tempHmm$transProbs <- TM
    tempHmm$emissionProbs <- EM
    tempHmm$startProbs <- SV
    if (d < delta) {
      break
    }
  }
  tempHmm$transProbs[is.na(hmm$transProbs)] <- NA
  tempHmm$emissionProbs[is.na(hmm$emissionProbs)] <- NA
  tempHmm$startProbs[is.na(hmm$startProbs)] <- NA
  return(list(hmm = tempHmm, difference = diff))
}