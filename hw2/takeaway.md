1. initialzation: fan_in,fan_out的场景, 初始化shape = (fan_in,fan_out)的weight矩阵
   1. 均匀分布, 使用(-a, a)的min max值 rand分布
   2. 正态分布, 使用标准差+E=0 randn
   3. 推导均匀分布
      1.  E(x)^2 = 0 , E(x^2) = 1/2a * x^2 从-a积到a => var = 1/3 * a^3
      2.  期望的方差为 2 / (fan_in + fan_out)
      3.  计算标准差
2. 