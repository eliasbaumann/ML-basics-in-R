# Random Forest Custom




```r
library(rpart) # using the tree implementation of rpart 

data("iris") # Example dataset to demonstrate the use of a random forest

# Define parameters:
number_of_trees = 200
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
  
  trees[[i]] = baseLearner
}
```

New dataset (in this case using the same as fit)

```r
newdata = iris
n_classes = length(levels(newdata$Species))
n_rows = dim(newdata)[1]
```

Classification prediction using Random Forest (Majority Vote)

```r
result = matrix(ncol=n_classes,nrow=n)


#initialize empty result row
result = matrix(data=0,nrow=n_rows,ncol=n_classes)
for (j in 1:number_of_trees){
  curr = predict(trees[[j]],newdata=newdata)
  
  # get maximum column and add a 1 as a vote
  votes = cbind(1:n_rows,max.col(curr,ties.method = 'random'))
  result[votes] = result[votes]+1
  
}
result = as.data.frame(result)

# Get final votes for each observed value
colnames(result) = levels(iris$Species)
votedResults = colnames(result)[max.col(result,ties.method="random")]
print(votedResults)
```

```
##   [1] "versicolor" "setosa"     "setosa"     "versicolor" "setosa"    
##   [6] "virginica"  "setosa"     "setosa"     "versicolor" "setosa"    
##  [11] "virginica"  "versicolor" "versicolor" "setosa"     "virginica" 
##  [16] "versicolor" "virginica"  "setosa"     "versicolor" "setosa"    
##  [21] "virginica"  "setosa"     "setosa"     "versicolor" "versicolor"
##  [26] "setosa"     "setosa"     "versicolor" "setosa"     "setosa"    
##  [31] "versicolor" "virginica"  "virginica"  "virginica"  "setosa"    
##  [36] "setosa"     "versicolor" "setosa"     "versicolor" "setosa"    
##  [41] "setosa"     "setosa"     "setosa"     "versicolor" "versicolor"
##  [46] "setosa"     "setosa"     "versicolor" "virginica"  "setosa"    
##  [51] "virginica"  "virginica"  "virginica"  "setosa"     "virginica" 
##  [56] "versicolor" "virginica"  "versicolor" "versicolor" "versicolor"
##  [61] "virginica"  "setosa"     "virginica"  "virginica"  "versicolor"
##  [66] "versicolor" "setosa"     "versicolor" "virginica"  "virginica" 
##  [71] "virginica"  "virginica"  "virginica"  "virginica"  "versicolor"
##  [76] "versicolor" "virginica"  "virginica"  "virginica"  "versicolor"
##  [81] "virginica"  "virginica"  "versicolor" "virginica"  "setosa"    
##  [86] "virginica"  "versicolor" "setosa"     "setosa"     "setosa"    
##  [91] "versicolor" "setosa"     "virginica"  "virginica"  "setosa"    
##  [96] "versicolor" "versicolor" "setosa"     "versicolor" "versicolor"
## [101] "virginica"  "virginica"  "virginica"  "versicolor" "versicolor"
## [106] "virginica"  "setosa"     "versicolor" "versicolor" "virginica" 
## [111] "virginica"  "virginica"  "virginica"  "virginica"  "setosa"    
## [116] "virginica"  "versicolor" "virginica"  "setosa"     "virginica" 
## [121] "virginica"  "virginica"  "virginica"  "virginica"  "virginica" 
## [126] "versicolor" "virginica"  "virginica"  "virginica"  "versicolor"
## [131] "virginica"  "virginica"  "virginica"  "virginica"  "virginica" 
## [136] "versicolor" "virginica"  "versicolor" "virginica"  "virginica" 
## [141] "setosa"     "virginica"  "virginica"  "virginica"  "virginica" 
## [146] "virginica"  "virginica"  "versicolor" "virginica"  "setosa"
```


Regression prediction using Random Forest (Mean over all trees for every Value)

```r
result = matrix(data=0,nrow=n_rows,ncol=n_classes)
for (j in 1:number_of_trees){
  curr = predict(trees[[j]],newdata=newdata)
  result = result+curr
}
result = result/number_of_trees
print(result)
```

```
##        setosa versicolor virginica
## 1   0.3392452  0.3414618 0.3192931
## 2   0.3476688  0.3357632 0.3165680
## 3   0.3472947  0.3312306 0.3214747
## 4   0.3213192  0.3471229 0.3315579
## 5   0.3458393  0.3335273 0.3206334
## 6   0.3202553  0.3345410 0.3452038
## 7   0.3335401  0.3378437 0.3286162
## 8   0.3329604  0.3354569 0.3315828
## 9   0.3212180  0.3470447 0.3317372
## 10  0.3433408  0.3386624 0.3179968
## 11  0.3263123  0.3273006 0.3463871
## 12  0.3330517  0.3373371 0.3296112
## 13  0.3393413  0.3395560 0.3211027
## 14  0.3300295  0.3342478 0.3357227
## 15  0.3151624  0.3339385 0.3508991
## 16  0.3083799  0.3471907 0.3444294
## 17  0.3315107  0.3283027 0.3401866
## 18  0.3496428  0.3365078 0.3138494
## 19  0.3190426  0.3415786 0.3393788
## 20  0.3464576  0.3300743 0.3234682
## 21  0.3243537  0.3316817 0.3439646
## 22  0.3389784  0.3317589 0.3292627
## 23  0.3344556  0.3320544 0.3334900
## 24  0.3278706  0.3477974 0.3243320
## 25  0.3200460  0.3454541 0.3345000
## 26  0.3420877  0.3339024 0.3240100
## 27  0.3357960  0.3401964 0.3240076
## 28  0.3291281  0.3413207 0.3295512
## 29  0.3353955  0.3325434 0.3320612
## 30  0.3388503  0.3388210 0.3223286
## 31  0.3401930  0.3376388 0.3221681
## 32  0.3253135  0.3359135 0.3387730
## 33  0.3208051  0.3337358 0.3454591
## 34  0.3268803  0.3274967 0.3456230
## 35  0.3436960  0.3372815 0.3190224
## 36  0.3485810  0.3315811 0.3198379
## 37  0.3387106  0.3325784 0.3287110
## 38  0.3465925  0.3379941 0.3154134
## 39  0.3302181  0.3358048 0.3339771
## 40  0.3325701  0.3347125 0.3327173
## 41  0.3549171  0.3317599 0.3133230
## 42  0.3344074  0.3173197 0.3482729
## 43  0.3317904  0.3396772 0.3285323
## 44  0.3373710  0.3499002 0.3127288
## 45  0.3301996  0.3454724 0.3243279
## 46  0.3472756  0.3346414 0.3180830
## 47  0.3417641  0.3329966 0.3252393
## 48  0.3269975  0.3469380 0.3260645
## 49  0.3285203  0.3325970 0.3388827
## 50  0.3406613  0.3386515 0.3206872
## 51  0.3237677  0.3357166 0.3405157
## 52  0.3223550  0.3308981 0.3467470
## 53  0.3252971  0.3265195 0.3481834
## 54  0.3404809  0.3231750 0.3363441
## 55  0.3298950  0.3185154 0.3515897
## 56  0.3301882  0.3385056 0.3313062
## 57  0.3333005  0.3312966 0.3354029
## 58  0.3085446  0.3491549 0.3423006
## 59  0.3311263  0.3414444 0.3274293
## 60  0.3262511  0.3502115 0.3235374
## 61  0.3179983  0.3389904 0.3430114
## 62  0.3382499  0.3263039 0.3354463
## 63  0.3168929  0.3227383 0.3603688
## 64  0.3415969  0.3271520 0.3312512
## 65  0.3321298  0.3360936 0.3317767
## 66  0.3374740  0.3450708 0.3174552
## 67  0.3495222  0.3139746 0.3365033
## 68  0.3098029  0.3519780 0.3382191
## 69  0.3303755  0.3024046 0.3672199
## 70  0.3187282  0.3296435 0.3516283
## 71  0.3359566  0.3294156 0.3346277
## 72  0.3289978  0.3293406 0.3416616
## 73  0.3331959  0.3180956 0.3487085
## 74  0.3320892  0.3157964 0.3521144
## 75  0.3308420  0.3526647 0.3164933
## 76  0.3277186  0.3458386 0.3264429
## 77  0.3220618  0.3302517 0.3476865
## 78  0.3392796  0.3230035 0.3377169
## 79  0.3294348  0.3270976 0.3434676
## 80  0.2916695  0.3618636 0.3464669
## 81  0.3175247  0.3296864 0.3527889
## 82  0.3184625  0.3360312 0.3455063
## 83  0.3142173  0.3449226 0.3408601
## 84  0.3448091  0.3281294 0.3270615
## 85  0.3457180  0.3240423 0.3302397
## 86  0.3318001  0.3247833 0.3434166
## 87  0.3361000  0.3286480 0.3352520
## 88  0.3450175  0.3156367 0.3393458
## 89  0.3443928  0.3289282 0.3266789
## 90  0.3469231  0.3336749 0.3194020
## 91  0.3387981  0.3390142 0.3221877
## 92  0.3504594  0.3159005 0.3336401
## 93  0.3189406  0.3400059 0.3410535
## 94  0.3163172  0.3420087 0.3416741
## 95  0.3476263  0.3333327 0.3190410
## 96  0.3298963  0.3432346 0.3268691
## 97  0.3351739  0.3483834 0.3164427
## 98  0.3414161  0.3361074 0.3224765
## 99  0.3203074  0.3447313 0.3349613
## 100 0.3241499  0.3481393 0.3277108
## 101 0.3289020  0.3359408 0.3351572
## 102 0.3461147  0.3271635 0.3267218
## 103 0.3333315  0.3273444 0.3393240
## 104 0.3404977  0.3331912 0.3263110
## 105 0.3275789  0.3464961 0.3259249
## 106 0.3360853  0.3218562 0.3420585
## 107 0.3548365  0.3324992 0.3126643
## 108 0.3438513  0.3302536 0.3258951
## 109 0.3356569  0.3481715 0.3161717
## 110 0.3229668  0.3316025 0.3454306
## 111 0.3273322  0.3338826 0.3387852
## 112 0.3371158  0.3309011 0.3319831
## 113 0.3325244  0.3295313 0.3379443
## 114 0.3457523  0.3229499 0.3312977
## 115 0.3500351  0.3232701 0.3266948
## 116 0.3417899  0.3298461 0.3283640
## 117 0.3408578  0.3327469 0.3263953
## 118 0.3224031  0.3299716 0.3476253
## 119 0.3446412  0.3308920 0.3244668
## 120 0.3335957  0.3055279 0.3608764
## 121 0.3230255  0.3371319 0.3398427
## 122 0.3384661  0.3198753 0.3416585
## 123 0.3332587  0.3286915 0.3380498
## 124 0.3322831  0.3310042 0.3367127
## 125 0.3137167  0.3402542 0.3460291
## 126 0.3329370  0.3394180 0.3276451
## 127 0.3239647  0.3206998 0.3553355
## 128 0.3431362  0.3211428 0.3357210
## 129 0.3358922  0.3284143 0.3356935
## 130 0.3320741  0.3477030 0.3202229
## 131 0.3399458  0.3253962 0.3346580
## 132 0.3214002  0.3356470 0.3429528
## 133 0.3359292  0.3322494 0.3318214
## 134 0.3263488  0.3188382 0.3548131
## 135 0.3417571  0.3226957 0.3355471
## 136 0.3373275  0.3343928 0.3282797
## 137 0.3290066  0.3281023 0.3428911
## 138 0.3506330  0.3381810 0.3111861
## 139 0.3380563  0.3261329 0.3358107
## 140 0.3334674  0.3330243 0.3335083
## 141 0.3397996  0.3341359 0.3260646
## 142 0.3449238  0.3205737 0.3345025
## 143 0.3461147  0.3271635 0.3267218
## 144 0.3213147  0.3389861 0.3396992
## 145 0.3199404  0.3371721 0.3428875
## 146 0.3364211  0.3334912 0.3300877
## 147 0.3464676  0.3196605 0.3338719
## 148 0.3303857  0.3408174 0.3287969
## 149 0.3339261  0.3245744 0.3414995
## 150 0.3513607  0.3223134 0.3263259
```
