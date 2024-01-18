## np.asarray
`np.asarray` 是 NumPy 函式庫中的一個函數，用於將輸入轉換為陣列。 它的作用是將輸入物件（如列表、元組、其他陣列等）轉換為 NumPy 陣列。

主要特點包括：
* 如果输入已经是一个 NumPy 数组，`np.asarray` 不会复制该数组，而是直接返回原始数组。这有助于避免不必要的内存使用。
* 如果输入是一个不是 NumPy 数组的对象（如列表或元组），`np.asarray` 会将它们转换为 NumPy 数组。转换后的数组会与原始对象共享数据，但可能会有不同的数据类型。

這裡是一個例子：
```python
import numpy as np

# 轉換列表為 NumPy 數組
my_list = [1, 2, 3, 4]
arr = np.asarray(my_list)
print(arr)
# 输出：[1 2 3 4]

# 如果已經是 NumPy 數組，不進行複製
existing_arr = np.array([5, 6, 7])
arr_asarray = np.asarray(existing_arr)
print(arr_asarray)
# 输出：[5 6 7]
```
在上面的例子中，`np.asarray` 分別將列表和現有的 NumPy 數組轉換為新的 NumPy 數組。請注意，即使對於已經是 NumPy 數組的情況，`np.asarray` 也不會進行複製，這有助於節省內存。
