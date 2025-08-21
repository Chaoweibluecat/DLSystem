1. struct "> I" 大端整数
2. np.nrange(x) => np的range(x)
3. Z[np.arange(Z.shape[0]), y] => 高级索引, 对Z的第i行取 y[i] index
4. y_oh = np.eye(num_classes, dtype=np.float32)[y_batch] => y onehot encoding
5. 多使用np.buffer, 全读完之后可以reshape
6. keepdim 保留维度可以用于广播
7. np.max会压缩维度,(求第二个参数对应维度的最小值), np.maxinum才是逐元素的操作
   
 