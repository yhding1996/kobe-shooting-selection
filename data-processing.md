## 关于数据

该数据集涵盖了科比布莱恩特20年职业生涯的所有投篮数据，其中，有5000个投篮的命中情况（是否命中）在题目中是缺失项。我们的目标就是利用剩下的投篮数据来预测这5000个投篮是否命中。

我们利用R来进行数据的预处理。

## 数据预处理

① 首先我们将需要用到的R程序包载入，并将所给的数据读入R程序。

``` R
#载入data.table、lubridate、dummies包
library(data.table)
library(lubridate)
library(dummies)
```

```R
#读取数据
data1 = fread("data.csv")
```

```R
#列出变量
names(data1)
```

输出以下变量：

[1] "action_type"        "combined_shot_type" "game_event_id"     

 [4] "game_id"            "lat"                "loc_x"             

 [7] "loc_y"              "lon"                "minutes_remaining" 

[10] "period"             "playoffs"           "season"            

[13] "seconds_remaining"  "shot_distance"      "shot_made_flag"    

[16] "shot_type"          "shot_zone_area"     "shot_zone_basic"   

[19] "shot_zone_range"    "team_id"            "team_name"         

[22] "game_date"          "matchup"            "opponent"          

[25] "shot_id"



 我们先作图来直观地观察这些数据体现的内容。

编写courtplot函数来描绘出投篮的位置。

先以combined_shot_type这一项来观察：

```R
train <- data1[!is.na(data1$shot_made_flag),]
train$shot_made_flag = as.factor(train$shot_made_flag)
courtplot = function(feat) {
        feat <- substitute(feat)
    train %>% 
    ggplot(aes(x = lon, y = lat)) +
        geom_point(aes_q(color = feat), alpha = 0.7, size = 3) +
        ylim(c(33.7, 34.0883)) +
        scale_color_brewer(palette = "Set1") +
        theme_void() +
        ggtitle(paste(feat))
}
courtplot(combined_shot_type)
```

结果如下图。

![1566877886205](https://github.com/yhding1996/kobe-shooting-selection/blob/master/pp/combined_shot_type.png?raw=true)

由于jump shot的数量过多，比较难看出其他投篮类型的分布情况。于是我们将jump shot的点隐去进一步观察：

```R
ggplot() +
    geom_point(data = filter(train, combined_shot_type == "Jump Shot"),
               aes(x = lon, y = lat), color = "grey", alpha = 0.3, size = 2) +
    geom_point(data = filter(train, combined_shot_type != "Jump Shot"),
                   aes(x = lon, y = lat, 
                       color = combined_shot_type), alpha = 0.7, size = 3) +
    ylim(c(33.7, 34.0883)) +
    scale_color_brewer(palette = "Set1") +
    theme_void() +
ggtitle("Shot Types")
```

输入结果如下。

![1566878052597](https://github.com/yhding1996/kobe-shooting-selection/blob/master/pp/shot%20types.png?raw=true)

再以shot_zone_area 这一项来观察：

```R
courtplot(shot_zone_area)
```

![1566878118815](https://github.com/yhding1996/kobe-shooting-selection/blob/master/pp/shot.png?raw=true)

以shot_zone_basic这一项来观察：

```R
courtplot(shot_zone_basic)
```

![1566878155098](https://github.com/yhding1996/kobe-shooting-selection/blob/master/pp/shot_zone_basic.png?raw=true)

以shot_zone_range这一项来观察：

```R
courtplot(shot_zone_range)
```

![1566878206739](https://github.com/yhding1996/kobe-shooting-selection/blob/master/pp/shot_zone_range.png?raw=true)

这样，我们就对这组投篮数据有了直观的理解。

我们可以大致将这些数据分成两组。

数值型变量：lat, loc_x, loc_y, lon, minutes_remaining, seconds_remaining, shot_distance

字符型变量：action_type, combined_shot_type, period, playoffs, season, shot_type, shot_zone_area, shot_zone_basic, shot_zone_range

对于日期型，我们需要单独进行处理。

② 为了能够确定每次投篮的位置，便于训练。我们将投篮的位置坐标进行z-score标准化。

经过z-score标准化后的数据符合标准正态分布。其转化函数为：

![img](file:///C:\Users\lenovo\AppData\Local\Temp\ksohtml8024\wps1.jpg) 

其中μ为所有样本数据的均值，σ为所有样本数据的标准差。

```R
#对投篮的位置坐标进行标准化
data1[,lon:=with(data1 , (data1$lon-mean(data1$lon))/sd(data1$lon))]
data1[,lat:=with(data1 , (data1$lat-mean(data1$lat))/sd(data1$lat))]
data1[,loc_x:=with(data1 , (data1$loc_x-mean(data1$loc_x))/sd(data1$loc_x))]
data1[,loc_y:=with(data1 , (data1$loc_y-mean(data1$loc_y))/sd(data1$loc_y))]
```



③ 为了使各个数值型变量在同一个标准下进行训练。我们对其进行去量纲化处理。方法是是使用min-max标准化。其是对原始数据的线性变换，使结果落到[0,1]区间，转换函数如下：

![img](file:///C:\Users\lenovo\AppData\Local\Temp\ksohtml8024\wps2.jpg) 

其中max为样本数据的最大值，min为样本数据的最小值。

```R
#对数值型的数据进行min-max标准化
data1[,time:=with(data1 , (data1$minutes_remaining/60)+data1$seconds_remaining)]
data1[,time:=with(data1,((data1$time-min(data1$time)))/(max(data1$time)-min(data1$time)))]
data1[,shot_distance:=with(data1,((data1$shot_distance-min(data1$shot_distance)))/(max(data1$shot_distance)-min(data1$shot_distance)))]
data1[,minutes_remaining:=NULL]
data1[,seconds_remaining:=NULL]
data1[,game_event_id:=with(data1,((data1$game_event_id-min(data1$game_event_id)))/(max(data1$game_event_id)-min(data1$game_event_id)))]
data1[,game_id:=with(data1,((data1$game_id-min(data1$game_id)))/(max(data1$game_id)-min(data1$game_id)))]
data1$game_id= NULL
data1$game_event_id= NULL
data1[,period:=with(data1,((data1$period-min(data1$period)))/(max(data1$period)-min(data1$period)))]
data1[,season:=NULL]
data1[,shot_id:=NULL]
```

④ 对于日期型的变量，我们利用R程序内的lubridate包内的lubridate函数将game_date内的年、月、日化为数值型，再进行max-min标准化。

```R
#将日期变量化为数值型，再进行max-min标准化
data1[,month:=lubridate::month(game_date)]
data1[,year:=lubridate::year(game_date)]
data1[,wday:=lubridate::wday(game_date)]
data1[,game_date:=NULL]
data1[,wday:=with(data1,((data1$wday-min(data1$wday)))/(max(data1$wday)-min(data1$wday)))]
data1[,year:=with(data1,((data1$year-min(data1$year)))/(max(data1$year)-min(data1$year)))]
data1[,month:=with(data1,((data1$month-min(data1$month)))/(max(data1$month)-min(data1$month)))]
```

 由于一些训练方法（例如xgboost）仅适用于数值型变量，所以对于那些非数值型的变量，需要变成哑变量。我们利用R程序内的dummies包将那些字符型变量全部转化成哑变量。其中，matchup那一列数据需要先将其中反映主客场的信息提取出来，再进行哑变量处理。

```R
#将非数值型的变量化为哑变量
data1[matchup %like% "@", matchup := 'Away']
data1[matchup %like% "vs.", matchup := 'Home']
data1$playoffs= as.character(data1$playoffs)
df = dummy.data.frame(data1 , names = c('action_type' , 'combined_shot_type' , 'shot_type' , 'shot_zone_area' ,'playoffs' , 'shot_zone_basic' , 'shot_zone_range' , 'matchup' , 'opponent') , sep='_')
```

至此，我们已经完成了数据预处理。