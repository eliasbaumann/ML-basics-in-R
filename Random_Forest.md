# Random Forest Custom




```r
library(rpart) # using the tree implementation of rpart 

data("iris") # Example dataset to demonstrate the use of a random forest

# Define parameters:
number_of_trees = 100
m_of_M = 2 # Number of columns to consider
n = dim(iris)[1]
M = dim(iris)[2]
trees = list()

#Fit forest to data:
#Steps:
for(i in 1:number_of_trees){
  # sample dataset for each tree with replacement (bootstrap sample)
  rows = sample(1:n,size=n,replace=TRUE)  
  
  # sample columns for each tree but with n<M number of total columns
  cols = sample(1:M,size=m_of_M)
  
  # grow tree to largest extend possible (do not prune) and find best split
  baseLearner = rpart(formula = iris$Species ~., data = iris[rows,cols],method = "class")
  
  #predict
  res = predict(baseLearner)
  trees = c(trees,list(res))
}

n_classes = length(levels(iris$Species))
```

Classification prediction using Random Forest (Majority Vote)

```r
result = matrix(ncol=n_classes,nrow=n)

#Classification
for(i in 1:n){
  allTreeResult = matrix(data=0,nrow=1,ncol=n_classes)
  for (j in 1:number_of_trees){
    curr = trees[[j]][grepl(paste(c("(^|\\b)",i,"(\\.|\\b)"),collapse=""),row.names(trees[[j]])),] 
    curr = matrix(curr,ncol=n_classes)
    votes = max.col(curr)
    for(k in dim(curr)[1]){
      allTreeResult[,votes[k]] = allTreeResult[,votes[k]] +1
    }
    
  }
  
  result[i,]=allTreeResult[1,]
  
}
result = as.data.frame(result)
colnames(result) = levels(iris$Species)
votedResults = colnames(result)[max.col(result,ties.method="random")]
print(votedResults)
```

```
##   [1] "virginica"  "virginica"  "virginica"  "virginica"  "setosa"    
##   [6] "versicolor" "virginica"  "setosa"     "virginica"  "virginica" 
##  [11] "setosa"     "virginica"  "virginica"  "virginica"  "setosa"    
##  [16] "versicolor" "virginica"  "virginica"  "setosa"     "virginica" 
##  [21] "virginica"  "virginica"  "virginica"  "versicolor" "virginica" 
##  [26] "versicolor" "versicolor" "virginica"  "setosa"     "versicolor"
##  [31] "versicolor" "versicolor" "virginica"  "setosa"     "versicolor"
##  [36] "virginica"  "setosa"     "virginica"  "virginica"  "setosa"    
##  [41] "virginica"  "virginica"  "virginica"  "versicolor" "virginica" 
##  [46] "virginica"  "virginica"  "virginica"  "setosa"     "setosa"    
##  [51] "setosa"     "versicolor" "virginica"  "setosa"     "versicolor"
##  [56] "versicolor" "virginica"  "versicolor" "versicolor" "setosa"    
##  [61] "versicolor" "setosa"     "versicolor" "setosa"     "versicolor"
##  [66] "setosa"     "versicolor" "setosa"     "versicolor" "setosa"    
##  [71] "virginica"  "versicolor" "setosa"     "virginica"  "versicolor"
##  [76] "versicolor" "versicolor" "virginica"  "virginica"  "setosa"    
##  [81] "setosa"     "setosa"     "setosa"     "versicolor" "virginica" 
##  [86] "versicolor" "setosa"     "versicolor" "versicolor" "setosa"    
##  [91] "versicolor" "virginica"  "setosa"     "setosa"     "versicolor"
##  [96] "setosa"     "virginica"  "versicolor" "setosa"     "virginica" 
## [101] "virginica"  "virginica"  "virginica"  "virginica"  "versicolor"
## [106] "virginica"  "virginica"  "virginica"  "virginica"  "setosa"    
## [111] "virginica"  "versicolor" "virginica"  "setosa"     "setosa"    
## [116] "virginica"  "virginica"  "setosa"     "virginica"  "virginica" 
## [121] "virginica"  "virginica"  "virginica"  "virginica"  "versicolor"
## [126] "virginica"  "virginica"  "versicolor" "virginica"  "setosa"    
## [131] "virginica"  "setosa"     "versicolor" "virginica"  "virginica" 
## [136] "setosa"     "virginica"  "versicolor" "virginica"  "virginica" 
## [141] "versicolor" "virginica"  "virginica"  "virginica"  "virginica" 
## [146] "versicolor" "virginica"  "virginica"  "setosa"     "virginica"
```


Regression prediction using Random Forest (Mean over all trees for every Value)

```r
result = matrix(nrow=n,ncol=n_classes)
for(i in 1:n){
  perTreeResult = matrix(data=0,nrow=number_of_trees,ncol=n_classes)
  for (j in 1:number_of_trees){
    curr = trees[[j]][grepl(paste(c("(^|\\b)",i,"(\\.|\\b)"),collapse=""),row.names(trees[[j]])),] 
    curr = matrix(curr,ncol=n_classes)
    aggregated = colMeans(curr)
    perTreeResult[j,]=aggregated    
  }
  result[i,]=colMeans(perTreeResult,na.rm = T)
}
print(result)
```

```
##             [,1]      [,2]      [,3]
##   [1,] 0.3355152 0.3275185 0.3369663
##   [2,] 0.3266199 0.3363756 0.3370045
##   [3,] 0.3394435 0.3119876 0.3485690
##   [4,] 0.3122601 0.3452719 0.3424679
##   [5,] 0.3491595 0.3307277 0.3201128
##   [6,] 0.3381484 0.3706696 0.2911820
##   [7,] 0.3390512 0.3204130 0.3405357
##   [8,] 0.3425006 0.3347639 0.3227354
##   [9,] 0.3356985 0.3274375 0.3368640
##  [10,] 0.3135512 0.3277371 0.3587117
##  [11,] 0.3496301 0.3354636 0.3149063
##  [12,] 0.3240050 0.3331625 0.3428325
##  [13,] 0.3408384 0.2835454 0.3756162
##  [14,] 0.3396020 0.2933922 0.3670058
##  [15,] 0.3802267 0.2951043 0.3246690
##  [16,] 0.3447754 0.3352851 0.3199395
##  [17,] 0.3502527 0.3215313 0.3282160
##  [18,] 0.3380942 0.3278189 0.3340869
##  [19,] 0.3640325 0.3219233 0.3140443
##  [20,] 0.3459479 0.3222590 0.3317931
##  [21,] 0.3415745 0.2883368 0.3700886
##  [22,] 0.3312663 0.3385233 0.3302104
##  [23,] 0.3438754 0.2933824 0.3627422
##  [24,] 0.3441869 0.3419526 0.3138606
##  [25,] 0.3466923 0.3331385 0.3201692
##  [26,] 0.3222515 0.3456033 0.3321453
##  [27,] 0.3254931 0.3591811 0.3153258
##  [28,] 0.3290051 0.3352670 0.3357279
##  [29,] 0.3375020 0.3275604 0.3349375
##  [30,] 0.3079494 0.3543337 0.3377169
##  [31,] 0.3360244 0.3521487 0.3118270
##  [32,] 0.3317335 0.3589728 0.3092937
##  [33,] 0.3284651 0.3042267 0.3673083
##  [34,] 0.3636323 0.3006746 0.3356931
##  [35,] 0.3232206 0.3534094 0.3233699
##  [36,] 0.3399103 0.3106973 0.3493923
##  [37,] 0.3592231 0.2948324 0.3459446
##  [38,] 0.3407721 0.3086458 0.3505820
##  [39,] 0.3530519 0.3066648 0.3402832
##  [40,] 0.3434528 0.3367279 0.3198193
##  [41,] 0.3435606 0.2960524 0.3603870
##  [42,] 0.3136435 0.3374862 0.3488703
##  [43,] 0.3333925 0.3017145 0.3648930
##  [44,] 0.3386062 0.3587616 0.3026322
##  [45,] 0.3599803 0.3073538 0.3326660
##  [46,] 0.3372449 0.3157014 0.3470537
##  [47,] 0.3523895 0.3177613 0.3298492
##  [48,] 0.3137440 0.3372023 0.3490537
##  [49,] 0.3382792 0.3303330 0.3313878
##  [50,] 0.3416347 0.3215558 0.3368095
##  [51,] 0.3495464 0.3173536 0.3331000
##  [52,] 0.3189257 0.3556841 0.3253902
##  [53,] 0.3479351 0.3298934 0.3221715
##  [54,] 0.3405408 0.3140061 0.3454530
##  [55,] 0.3335432 0.3433301 0.3231267
##  [56,] 0.2942417 0.3681316 0.3376267
##  [57,] 0.3401765 0.3360269 0.3237966
##  [58,] 0.3249826 0.3402835 0.3347339
##  [59,] 0.3301480 0.3443218 0.3255302
##  [60,] 0.3371670 0.3292308 0.3336022
##  [61,] 0.3101045 0.3663684 0.3235272
##  [62,] 0.3340110 0.3360090 0.3299799
##  [63,] 0.3444274 0.3421270 0.3134456
##  [64,] 0.3543357 0.3086089 0.3370553
##  [65,] 0.2855057 0.3766798 0.3378145
##  [66,] 0.3443717 0.3283068 0.3273215
##  [67,] 0.3011347 0.3784907 0.3203746
##  [68,] 0.3519663 0.3324101 0.3156236
##  [69,] 0.2856421 0.3608872 0.3534707
##  [70,] 0.3492845 0.3377315 0.3129840
##  [71,] 0.3554322 0.3317128 0.3128550
##  [72,] 0.3201539 0.3409470 0.3388991
##  [73,] 0.3603700 0.3171386 0.3224913
##  [74,] 0.3266587 0.3284506 0.3448907
##  [75,] 0.3232662 0.3479887 0.3287451
##  [76,] 0.3424459 0.3363339 0.3212202
##  [77,] 0.3133824 0.3579606 0.3286570
##  [78,] 0.3404575 0.3114347 0.3481077
##  [79,] 0.3113588 0.3555444 0.3330968
##  [80,] 0.3242088 0.3515805 0.3242107
##  [81,] 0.3375851 0.3308434 0.3315715
##  [82,] 0.3483810 0.3271595 0.3244595
##  [83,] 0.3569184 0.3374216 0.3056599
##  [84,] 0.3570131 0.3468307 0.2961562
##  [85,] 0.3132173 0.3373829 0.3493998
##  [86,] 0.3268389 0.3500027 0.3231584
##  [87,] 0.3497250 0.3339837 0.3162913
##  [88,] 0.3018047 0.3589740 0.3392213
##  [89,] 0.2770520 0.3863272 0.3366208
##  [90,] 0.3385824 0.3288499 0.3325677
##  [91,] 0.3155840 0.3433056 0.3411105
##  [92,] 0.3205491 0.3393408 0.3401101
##  [93,] 0.3459564 0.3219054 0.3321382
##  [94,] 0.3239037 0.3362983 0.3397981
##  [95,] 0.2948926 0.3620731 0.3430343
##  [96,] 0.3237394 0.3388954 0.3373653
##  [97,] 0.2966313 0.3444983 0.3588704
##  [98,] 0.3261473 0.3375862 0.3362665
##  [99,] 0.3478637 0.3363539 0.3157824
## [100,] 0.3131489 0.3361430 0.3507081
## [101,] 0.3355262 0.3484257 0.3160480
## [102,] 0.3322462 0.3201768 0.3475770
## [103,] 0.3368681 0.3170511 0.3460808
## [104,] 0.3275950 0.3326113 0.3397937
## [105,] 0.3325944 0.3594817 0.3079239
## [106,] 0.3149842 0.3111058 0.3739100
## [107,] 0.3252999 0.3177587 0.3569413
## [108,] 0.3343410 0.3139927 0.3516663
## [109,] 0.3567704 0.3226530 0.3205766
## [110,] 0.3570092 0.3269448 0.3160460
## [111,] 0.3327546 0.3378186 0.3294268
## [112,] 0.3380127 0.3470819 0.3149054
## [113,] 0.3184720 0.3593139 0.3222141
## [114,] 0.3689821 0.2989432 0.3320747
## [115,] 0.3499127 0.3295700 0.3205173
## [116,] 0.3289438 0.3517529 0.3193033
## [117,] 0.3295068 0.3404973 0.3299958
## [118,] 0.3506331 0.3335379 0.3158290
## [119,] 0.3290436 0.3204425 0.3505138
## [120,] 0.3414808 0.3347067 0.3238125
## [121,] 0.3505703 0.3205761 0.3288536
## [122,] 0.3344718 0.3139335 0.3515947
## [123,] 0.3124085 0.3421506 0.3454409
## [124,] 0.3479011 0.3120179 0.3400810
## [125,] 0.3310433 0.3399103 0.3290465
## [126,] 0.3212417 0.3389045 0.3398539
## [127,] 0.3169498 0.3298806 0.3531696
## [128,] 0.3458809 0.3565660 0.2975531
## [129,] 0.3237974 0.3420447 0.3341579
## [130,] 0.3458951 0.3258282 0.3282766
## [131,] 0.3077792 0.3368964 0.3553244
## [132,] 0.3530926 0.3205946 0.3263128
## [133,] 0.3195373 0.3655647 0.3148980
## [134,] 0.3328559 0.3328302 0.3343139
## [135,] 0.3053220 0.3487973 0.3458807
## [136,] 0.3615417 0.3207768 0.3176815
## [137,] 0.3178275 0.3432836 0.3388889
## [138,] 0.3339127 0.3626080 0.3034794
## [139,] 0.3526777 0.3358337 0.3114887
## [140,] 0.3327910 0.3329819 0.3342271
## [141,] 0.3313812 0.3679113 0.3007074
## [142,] 0.3384783 0.3384405 0.3230812
## [143,] 0.3488143 0.3219503 0.3292354
## [144,] 0.3314360 0.3516933 0.3168707
## [145,] 0.3297383 0.3228769 0.3473848
## [146,] 0.3411126 0.3555155 0.3033719
## [147,] 0.3471779 0.3181310 0.3346911
## [148,] 0.3360610 0.3115054 0.3524337
## [149,] 0.3698739 0.3316755 0.2984507
## [150,] 0.3378811 0.3147229 0.3473960
```
