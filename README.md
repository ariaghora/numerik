Numerik is a numerical library for object pascal. It supports:
- Multidimensional array data structure with single precision
- Array broadcasting and multidimensional slicing
- BLAS-accelerated operation (you can get `libopenblas.dll` from [here](https://github.com/xianyi/OpenBLAS/releases))
- Familiar pascal style math API

### Example
```pascal
uses
  multiarray,
  numerik;

var
  A, B: TMultiArray;
  RowIndices, ColIndices: TLongVector;

begin

  A := Random([5, 5]); // 5 by 5 random array
  PrintMultiArray(A);

  WriteLn;

  RowIndices := [0, 3];
  ColIndices := [0, 1, 3, 4];
  B := A.Slice([RowIndices, ColIndices]);

  PrintMultiArray(B);

  WriteLn;

  PrintMultiArray(B.T + 2); // Transpose and add 2

  readln;
end.      
```

Read the API reference [here](http://ariaghora.github.io/numerik).

### TODO:
- [x] ~~Array slicing~~
- [x] ~~BLAS support~~(for now only matrix multiplication is supported)
