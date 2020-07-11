Numerik is a numerical library for object pascal. It supports:
- Multidimensional array data structure with single precision
- Array broadcasting and multidimensional slicing
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

> Note: Numerik's TMultiArray will be used in my neural network framework, [Noe](https://github.com/ariaghora/noe).

### TODO:
- [x] ~~Array slicing~~
- [ ] BLAS support
