Numerik is a numerical library for object pascal. It supports:
- Multidimensional array data structure with single precision
- Efficient array broadcasting and multidimensional slicing
- Some arithmetic and linear algebra functionalities
- BLAS-accelerated operation (requires `libopenblas.dll` which can be obtained from [here](https://github.com/xianyi/OpenBLAS/releases))

### Example
```pascal
uses
  multiarray, numerik;

var
  A, B: TMultiArray;
  RowIndices, ColIndices: TLongVector;
  SVDResult: TSVDResult;

begin
  A := [1, 2, 3, 4, 5, 6]; // TMultiArray accepts open array assignment
  A := A.Reshape([3, 2]);  // Reshape it into a 3x2 array
  PrintMultiArray(A);

  { Load data from CSV }
  A := ReadCSV('data.csv'); 
  PrintMultiArray(A);
  
  { 5 x 5 random array }
  A := Random([5, 5]);
  PrintMultiArray(A);

  { Multidimensional slicing }
  RowIndices := [0, 3];
  ColIndices := [0, 1, 3, 4];
  B := A.Slice([RowIndices, ColIndices]);
  PrintMultiArray(B);

  { Math operation }
  PrintMultiArray(Transpose(B));   // Equivalent to B.T
  PrintMultiArray(Exp(B));         // Element-wise exponentiation
  PrintMultiArray(Add(B, 2));      // Equivalent to B + 2
  PrintMultiArray(Multiply(B, B)); // Element-wise multiplication, Equivalent to B * B
  PrintMultiArray(Matmul(B.T, B)); // Matrix multiplication 
  
  { Singular value decomposition }
  SVDResult := SVD(B);
  PrintMultiArray(SVDResult.U);
  PrintMultiArray(SVDResult.Sigma);
  PrintMultiArray(SVDResult.VT);

  { Pseudo-inverse }
  PrintMultiArray(PseudoInverse(B));
end.      
```

Read the API reference [here](http://ariaghora.github.io/numerik).

### TODO:
- [x] ~~Array slicing~~
- [x] ~~BLAS support~~(for now only matrix multiplication is supported)
