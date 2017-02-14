# NOTE: I won't be using my usual pipeline because this dataset is too big

library(needs)
needs(skaggle, RANN.L1, nnet, xgboost, readr, data.table, feather, glmnet, ranger, kknn)

# Configuration
# ==================================================================================================

config = create.config('MAP@3', mode = 'single', layer = 0)

config$do.load       = T
config$do.stuff      = T
config$do.submit     = T

config$submt.id = '4_2'
config$ref.submt.id = '2'

#
# Training parameters
#

config$holdout.validation = F # can't afford CV
config$first.valid.time = 630000 # so about a third of the data is validation
config$data.variant = 'new' # { basic, best, mine, new }

# Submission
# ==================================================================================================

generate.submission = function(preds) {
  cat(date(), 'Generating submission\n')
  
  preds[is.na(preds)] = -1 # FIXME complement with the best public script if I have to
  submission = data.table(row_id = config$dte$row_id, place_id = paste(preds[, 1], preds[, 2], preds[, 3]))
  cat(date(), 'Writing submission file\n')
  readr::write_csv(submission, paste0(config$tmp.dir, '/sbmt-', config$submt.id, '.csv'))
  zip(paste0(config$tmp.dir, '/sbmt-', config$submt.id, '.zip'), paste0(config$tmp.dir, '/sbmt-', config$submt.id, '.csv'))
  
  ref.sbmt = fread(paste0(config$tmp.dir, '/sbmt-', config$ref.submt.id, '.csv'))
  ref.sbmt = ref.sbmt[order(row_id)]
  ref.preds = strsplit(ref.sbmt$place_id, ' ', fixed = T)
  ref.preds1 = as.numeric(unlist(lapply(ref.preds, function(x) x[[1]])))
  ref.preds1[is.na(ref.preds1)] = -1 # public scripts have some NA predictions...
  preds1 = preds[, 1]
  
  cat('Sanity check: first predictions in new and ref match', mean(preds1 == ref.preds1), 'of the time\n')
}

# Load data
# ==================================================================================================

prepare.data = function() {
  # Do elementry preprocessing and store the data as RData for a bit faster loading
  cat(date(), 'Reading train.csv\n')
  dat = fread(paste0(config$tmp.dir, '/train.csv'), integer64 = 'character', showProgress = F)
  cat(date(), 'Writing train.feather\n')
  write_feather(dat, paste0(config$tmp.dir, '/train.feather'))
  cat(date(), 'Reading test.csv\n')
  dat = fread(paste0(config$tmp.dir, '/test.csv'), integer64 = 'character', showProgress = F)
  cat(date(), 'Writing test.feather\n')
  write_feather(dat, paste0(config$tmp.dir, '/test.feather'))
}

load.data = function() {
  cat(date(), 'Loading data\n')
  
  train = as.data.table(read_feather(paste0(config$tmp.dir, '/train.feather')))
  test  = as.data.table(read_feather(paste0(config$tmp.dir, '/test.feather' )))

  # => there are 29,118,021 train samples
  # => there are 8,607,230 test samples
  
  add.time.features = function(dat) {
    # FIXME a better way to do this would learn these from the data. The current impl may suffer 
    # from cumulative error, and we don't know which days are weekend, or what day of the month is.

    if (config$data.variant == 'basic') {
      # Basics
      dat[, th := time %% 60] # minutes in each hour
      dat[, td := (floor(time /  60           ) + 1) %% 24] # hour in each day
      dat[, tw := (floor(time / (60 * 24     )) + 1) %%  7] # day in each week
      dat[, tm := (floor(time / (60 * 24     )) + 1) %% 30] # day in each month
      dat[, ty := (floor(time / (60 * 24 * 30)) + 1) %% 12] # month in each year
    } else if (config$data.variant == 'best') {
      # Best public script (2016-06-29)
      nr.digit = 4 # this isn't important
      dat[, x := x * 22]
      dat[, y := y * 52]
      dat[, minute := 2 * pi * (floor(time / 5) %% 288) / 288] # counter of 5min intervals that resets each day
      dat[, minute_sin := round(sin(minute) + 1, digits = nr.digit) * 0.56515]
      dat[, minute_cos := round(cos(minute) + 1, digits = nr.digit) * 0.56515]
      dat[, minute := NULL]
      dat[, day := 2 * pi * (floor(time / 1440) %% 365) / 365] # day counter that resets each year
      dat[, day_of_year_sin := round(sin(day) + 1, digits = nr.digit) * 0.32935]
      dat[, day_of_year_cos := round(cos(day) + 1, digits = nr.digit) * 0.32935]
      dat[, day := NULL]
      dat[, weekday := 2 * pi * (floor(time / 1440) %% 7) / 7] # day counter that resets each week
      dat[, weekday_sin := round(sin(weekday) + 1, digits = nr.digit) * 0.2670]
      dat[, weekday_cos := round(cos(weekday) + 1, digits = nr.digit) * 0.2670]
      dat[, weekday := NULL]
      dat[, year := floor(time / 525600) * 0.51785] # year counter
      dat[, accuracy := log10(accuracy) * 0.6] # a heuristic
      
      # FIXME for now I'll still use this for slicing
      dat[, td := abs(time %% (60 * 24) - (60 * 24) / 2)]
    } else if (config$data.variant == 'mine') {
      # My take - triangle wave to accomodate periodicity in computing L1 distance on time features
      dat[, th := abs(time %%  60                 -  60                 / 2)]
      dat[, td := abs(time %% (60 * 24          ) - (60 * 24          ) / 2)]
      dat[, tw := abs(time %% (60 * 24 *  7     ) - (60 * 24 *  7     ) / 2)]
      dat[, tm := abs(time %% (60 * 24 * 30     ) - (60 * 24 * 30     ) / 2)]
      dat[, ty := abs(time %% (60 * 24 * 30 * 12) - (60 * 24 * 30 * 12) / 2)]
    } else if (config$data.variant == 'new') {
      #dat[, moh :=                                                     %% 60 ] # minutes in each hour
      #dat[, h   := as.factor((floor((time - 1) /  60                )) %% 24)] # hour in each day
      #dat[, dow := as.factor((floor((time - 1) / (60 * 24          ))) %%  7)] # day in each week
      #dat[, dom := as.factor((floor((time - 1) / (60 * 24          ))) %% 30)] # day in each month
      #dat[, moy := as.factor((floor((time - 1) / (60 * 24 * 30     ))) %% 12)] # month in each year
      #dat[, yea := as.factor( floor((time - 1) / (60 * 24 * 30 * 12))       )] # year
      
      dat[, t1  := time                                 %% 60] # minute-resolution minutes in each hour
      dat[, t2  := (((time - 1) /  60                )) %% 24] # minute-resolution hour in each day
      dat[, t3  := (((time - 1) / (60 * 24          ))) %%  7] # minute-resolution day in each week
      dat[, t4  := (((time - 1) / (60 * 24          ))) %% 30] # minute-resolution day in each month
      dat[, t5  := (((time - 1) / (60 * 24 * 30     ))) %% 12] # minute-resolution month in each year
      dat[, t6  :=  ((time - 1) / (60 * 24 * 30 * 12))       ] # minute-resolution year

      if (0) {
        # I get the basic idea of these, but why sine and not triangle? why these phases?
        dat[, t7  := sin(2 * pi / 365 * (floor(time / 1440) %% 365))] # day in each year
        dat[, t8  := cos(2 * pi / 365 * (floor(time / 1440) %% 365))]
        dat[, t9  := sin(2 * pi /  30 * (floor(time / 1440) %%  30))] # day in each month
        dat[, t10 := cos(2 * pi /  30 * (floor(time / 1440) %%  30))]
        dat[, t11 := sin(2 * pi /   7 * (floor(time / 1440) %%   7))] # day in each week
        dat[, t12 := cos(2 * pi /   7 * (floor(time / 1440) %%   7))]
        dat[, t13 := sin(2 * pi /  24 * (floor(time /   60) %%  24))] # hour in each day
        dat[, t14 := cos(2 * pi /  24 * (floor(time /   60) %%  24))]
      } else {
        # These make more sense to me, but I'm keeping the [0,2] scale so that weights don't need to be changed too much
        # FIXME maybe I should drop the "floor"
        dat[, t7  := abs((floor((time - 1) / 1440) - 365 / 2) %% 365 - 365 / 2) / (365 / 4)] # day in each year
        dat[, t8  := abs((floor((time - 1) / 1440) - 365 / 4) %% 365 - 365 / 2) / (365 / 4)]
        dat[, t9  := abs((floor((time - 1) / 1440) -  30 / 2) %%  30 -  30 / 2) / ( 30 / 4)] # day in each month
        dat[, t10 := abs((floor((time - 1) / 1440) -  30 / 4) %%  30 -  30 / 2) / ( 30 / 4)]
        dat[, t11 := abs((floor((time - 1) / 1440) -   7 / 2) %%   7 -   7 / 2) / (  7 / 4)] # day in each week
        dat[, t12 := abs((floor((time - 1) / 1440) -   7 / 4) %%   7 -   7 / 2) / (  7 / 4)]
        dat[, t13 := abs((floor((time - 1) /   60) -  24 / 2) %%  24 -  24 / 2) / ( 24 / 4)] # hour in each day
        dat[, t14 := abs((floor((time - 1) /   60) -  24 / 4) %%  24 -  24 / 2) / ( 24 / 4)]
      }
      if (0) {
        # EXPERIMENT: additional phases of the same periodic features
        dat[, tt1  := ( time                       - 30) %% 60]
        dat[, tt2  := ((time - 1) /  60            - 12) %% 24]
        dat[, tt3  := ((time - 1) / (60 * 24     ) - 3 ) %%  7]
        dat[, tt4  := ((time - 1) / (60 * 24     ) - 15) %% 30]
        dat[, tt5  := ((time - 1) / (60 * 24 * 30) -  6) %% 12]
        dat[, tt6  := abs((floor((time - 1) / 1440) + 365 / 4) %% 365 - 365 / 2) / (365 / 4)]
        dat[, tt7  := abs((floor((time - 1) / 1440) +  30 / 4) %%  30 -  30 / 2) / ( 30 / 4)]
        dat[, tt8  := abs((floor((time - 1) / 1440) +   7 / 4) %%   7 -   7 / 2) / (  7 / 4)]
        dat[, tt9  := abs((floor((time - 1) /   60) +  24 / 4) %%  24 -  24 / 2) / ( 24 / 4)]
      }
      
      if (0) { # these don't seem to add anything
        # Frequency coding of time (binned to weeks)
        dat[, nx  := .N, by = floor(x / 100)]
        dat[, ny  := .N, by = floor(y / 100)]
        dat[, nxy := .N, by = .(xb = floor(x / 100), yb = floor(y / 100))]
        dat[, nt  := .N, by = floor((time - 1) / (1440 * 7))]
        dat[, nat := mean(accuracy), by = floor((time - 1) / (1440 * 7))]
        
        # Accuracy dynamics (may implicate other simulation signals)
        dat[, ap1 := floor((time - 1) / (1440 * 7)) %in% (10:18)]
        dat[, ap2 := floor((time - 1) / (1440 * 7)) %in% (21:25)]
        dat[, ap3 := floor((time - 1) / (1440 * 7)) %in% (41:44)]
      }
      
      # FIXME for now I'll still use this for slicing. I probably want to try out different phase offsets
      dat[, td := abs(time %% (60 * 24) - (60 * 24) / 2) / ((60 * 24) / 2)]
    } else {
      stop('wtf')
    }
  }
  
  add.time.features(train)
  add.time.features(test )
  
  # TODO: some of the best public scripts add examples with mirrored time if near the boundaries of the day cycle
  
  if (config$holdout.validation) {
    cat(date(), 'NOTE: setting aside 1/3 of the training data for validation\n')
    valid = train[time >= config$first.valid.time]
    train = train[time <  config$first.valid.time]
    # The organizers removed any examples with test place_id not in train
    valid = valid[place_id %in% unique(train$place_id)]
    valid[, row_id := -(1:nrow(valid))]
  }
  
  config$dtr <<- train
  config$dte <<- test
  if (config$holdout.validation) {
    config$dva <<- valid
  }
  
  gc()
}

# Generate predictions
# ==================================================================================================

gen.preds.fixed.bins = function() {
  # This simply splits the data into fixed bins on (x,y) and fits the most common place_ids in each
  
  # TODO: tune these (though it doesn't seem like there is much room for improvement)
  wbx =  500 / 10
  wby = 1000 / 10

  # Count and select top 3 place_ids in each bin
  # TODO use accuracy somehow?
  train.bin.counts = config$dtr[, .(w = .N), by = .(bx = floor(x * wbx), by = floor(y * wby), place_id)]
  bin.top3 = train.bin.counts[, as.list(place_id[order(w, decreasing = T)[1:3]]), by = .(bx, by, bt)]
  
  if (config$holdout.validation) {
    preds = merge(config$dva[, .(row_id, bx = floor(x * wbx), by = floor(y * wby))], bin.top3, by = c('bx', 'by'), all.x = T)
    preds = preds[order(row_id, decreasing = T), .(row_id, place_id1 = V1, place_id2 = V2, place_id3 = V3)]
    
    # FIXME this shouldn't happen often, but I need to fill it in sensibly
    preds[is.na(preds)] = 0
    
    # TODO this won't work like this, need to implement a simpler solution?
    map3 = eval.map3.core(preds[, 2:4, with = F], config$dva$place_id)
    cat(date(), 'Validation MAP@3 =', map3, '\n')
    # => about 0.44
  }
  
  if (config$do.submit) {
    # TODO
  }
}

gen.preds.nnxgb.bins = function() {
  # Starting from the current best public script:
  # https://www.kaggle.com/drarfc/facebook-v-predicting-check-ins/script-competition-facebook-v/run/280081
  # This splits the (x,y) space to fixed bins. Each bin is then augmented with a margin (so now the 
  # bins overlap, but only for the purpose of training), and a kNN is fitted to each.

  dx = max(config$dtr$x)
  dy = max(config$dtr$y)
  dt = max(config$dtr$td)

  bx = 50
  by = 50
  bt = 1
  xpad = 0
  ypad = 0
  tpad = 0
  xoff = dx / (2 * bx)
  yoff = dy / (2 * by)

  min.n = 8
  max.m = 150
  knn.k = 37
  knn.eps = 0 #1e-4 # in the final solution maybe I can reduce this to 0
  knn.wd = -2.25
  alpha.xgb = 0.5
  
  if (1) {
    # Everything
    I = 1:bx
    J = 1:by
    K = 1:bt
  } else if (1) {
    cat('NOTE: working on specific blocks: 26..50\n')
    I = 26:50
    J = 1:by
    K = 1:bt
  } else {
    cat('NOTE: working in quick experimentation mode\n')
    I = c(1, 4, 8)
    J = c(9, 14, 16)
    K = 1
  }
  
  if (config$do.submit) {
    preds = matrix(NA_character_, nrow(config$dte), 3)
    
    ijk = 0
    for (i in I) {
      xlo = xoff + (dx / bx) * (i - 1)
      xhi = xoff + (dx / bx) * i + ifelse(i == bx, 0, .Machine$double.neg.eps)
      xloa = xlo - xpad
      xhia = xhi + xpad
      
      for (j in J) {
        ylo = yoff + (dy / by) * (j - 1)
        yhi = yoff + (dy / by) * j + ifelse(j == by, 0, .Machine$double.neg.eps)
        yloa = ylo - ypad
        yhia = yhi + ypad
        
        for (k in K) {
          tlo = (dt / bt) * (k - 1)
          thi = (dt / bt) * k + ifelse(k == bt, 0, .Machine$double.neg.eps)
          tloa = tlo - tpad
          thia = thi + tpad
          
          ijk = ijk + 1   
          cat(date(), 'cell', ijk, 'of', length(I) * length(J) * length(K), '\n')
          
          # TODO how do I decrease the importance of older examples? maybe they are stale?
          # TODO it will probably work better if I use cosine distance on periodic time features (but I don't have a good package for this. Can do this with python, but it's too slow)
          
          train.bidx = (config$dtr$x >= xloa & config$dtr$x <= xhia & config$dtr$y >= yloa & config$dtr$y <= yhia & config$dtr$td >= tloa & config$dtr$td <= thia)
          test.bidx  = (config$dte$x >= xlo  & config$dte$x <= xhi  & config$dte$y >= ylo  & config$dte$y <= yhi  & config$dte$td >= tlo  & config$dte$td <= thi )
          
          ptbl = table(config$dtr$place_id[train.bidx])
          place_ids.kept = names(ptbl)[ptbl >= min.n]
          #place_ids.kept = names(head(sort(table(config$dtr$place_id[train.bidx]), decreasing = T), max.m))
          #cat('Fraction of cell examples retained', sum(train.bidx & (config$dtr$place_id %in% place_ids.kept)) / sum(train.bidx), '\n')
          train.bidx  = train.bidx & (config$dtr$place_id %in% place_ids.kept)

          #
          # KNN
          #
          
          tmp.train = as.data.frame(config$dtr[train.bidx, .(x * 23, y * 56, log10(accuracy) * 1.3, t6 * 0.55, t7 * 0.33, t8 * 0.33, t11 * 0.27, t12 * 0.27, t13 * 0.64, t14 * 0.64, place_id = factor(place_id, levels = place_ids.kept))])
          tmp.test  = as.data.frame(config$dte[test.bidx , .(x * 23, y * 56, log10(accuracy) * 1.3, t6 * 0.55, t7 * 0.33, t8 * 0.33, t11 * 0.27, t12 * 0.27, t13 * 0.64, t14 * 0.64)])
          
          preds0.knn = kknn(place_id ~ ., tmp.train, tmp.test, k = round(1.2 * knn.k), scale = F, distance = 1)$prob
          
          #
          # XGB
          #
          
          nr.classes = length(place_ids.kept)
          
          if (config$data.variant == 'mine') {
            feature.names = c('x', 'y', 'accuracy', 'th', 'td', 'tw', 'tm', 'ty', 'th2', 'td2', 'tw2', 'tm2', 'ty2')
          } else if (config$data.variant == 'best') {
            feature.names = c('accuracy', 'day_of_year_sin', 'day_of_year_cos', 'minute_sin', 'minute_cos', 'weekday_sin', 'weekday_cos', 'x', 'y', 'year')
          } else if (config$data.variant == 'new') {
            feature.names = c('x', 'y', 'accuracy', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14')
            #feature.names = c('x', 'y', 'accuracy', 'time', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 'nx', 'ny', 'nxy', 'nt', 'nat', 'ap1', 'ap2', 'ap3', 'ap')
          } else {
            stop('wtf')
          }
          
          dtrain = config$dtr[train.bidx, feature.names, with = F]
          dtest  = config$dte[test.bidx , feature.names, with = F]
          
          train.labels = as.integer(factor(config$dtr$place_id[train.bidx], levels = place_ids.kept)) - 1

          xtrain = xgb.DMatrix(data.matrix(dtrain), label = train.labels)
          xtest  = xgb.DMatrix(data.matrix(dtest ))

          xgb.params = list(
            booster           = 'gbtree',
            objective         = 'multi:softprob',
            eval_metric       = 'mlogloss',
            nrounds           = 300,
            eta               = 0.15, #0.08,
            max_depth         = 1,
            min_child_weight  = 5,
            #gamma             = 0,
            #lambda            = 5,
            #alpha             = 0,
            #num_parallel_tree = 5,
            subsample         = 0.5,
            colsample_bytree  = 0.5,
            num_class         = nr.classes,
            annoying = T
          )
          
          set.seed(1234)
          xgb.fit = xgb.train(
            params            = xgb.params,
            nrounds           = xgb.params$nrounds,
            maximize          = (xgb.params$objective == 'rank:pairwise'),
            data              = xtrain,
            #watchlist         = watchlist,
            print.every.n     = 20
          )
          
          preds0.xgb = t(matrix(predict(xgb.fit, xtest), nrow = nr.classes)) 
          
          # Blend FIXME try geommean
          preds0 = (1 - alpha.xgb) * preds0.knn + alpha.xgb * preds0.xgb
          
          # Keep only the top 3 classes for submission
          preds0 = t(apply(preds0, 1, order, decreasing = T))[, 1:3]
          preds0 = matrix(place_ids.kept[preds0], ncol = 3)
          preds[test.bidx, ] = preds0
        }
      }
    }

    generate.submission(preds)    
  }
}

# Do stuff
# ==================================================================================================

if (config$mode == 'single') {
  cat(date(), 'Starting single mode\n')
  
  if (config$do.load) {
    load.data()
  }
  
  if (config$do.stuff) {
    #gen.preds.fixed.bins()
    gen.preds.nnxgb.bins()
    gc()
  }
}

cat(date(), 'Done.\n')
