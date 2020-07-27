{
  A unit containing:

  @unorderedList(
    @item(Mathematical operation over TMultiArrays)
    @item(Some linear algebra functionality )
    @item(Random number generation )
  )
}

unit numerik;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, multiarray, numerik.blas;

type
  TReduceFunc = function(A, B: TMultiArray): TMultiArray;

  { Holding the result of SVD }
  TSVDResult = record
    U, Sigma, VT: TMultiArray;
  end;

{ ----------- Basic array creation --------------------------------------------}
{ Create multiarray with shape Shape containing Ones }
function FullMultiArray(Shape: TLongVector; val: single): TMultiArray;
{ Create multiarray with shape Shape containing Ones }
function Ones(Shape: TLongVector): TMultiArray;
{ Create multiarray with shape Shape containing zeros }
function Zeros(Shape: TLongVector): TMultiArray;

{ ----------- Arithmetic ----------------------------------------------------- }

function Abs(A: TMultiArray): TMultiArray; overload;
function ArgMax(A: TMultiArray; axis: integer = -1; KeepDims: boolean=False): TMultiArray; overload;
function Max(A: TMultiArray; axis: integer = -1; KeepDims: boolean=False): TMultiArray; overload;
{ Compute the mean of a A along axis. The default axis is -1, meaning the mean is
  computed over all items in A. }
function Mean(A: TMultiArray; axis: integer = -1; KeepDims: boolean=False): TMultiArray; overload;

function Round(A: TMultiArray): TMultiArray; overload;

{ Compute the sum of a A along axis. The default axis is -1, meaning the sum is
  computed over all items in A. }
function Sum(A: TMultiArray; axis: integer = -1; KeepDims: boolean=False): TMultiArray; overload;

{ ----------- Trigonometry --------------------------------------------------- }

{ Compute element wise cosine over A }
function Cos(A: TMultiArray): TMultiArray; overload;
{ Convert degree to radian element wise over A }
function DegToRad(A: TMultiArray): TMultiArray; overload;
{ Compute element wise sine over A }
function Sin(A: TMultiArray): TMultiArray; overload;

{ ----------- Exponential ---------------------------------------------------- }

{ Compute element wise exponent over A }
function Exp(A: TMultiArray): TMultiArray; overload;
function Ln(A: TMultiArray): TMultiArray; overload;

{ ----------- Random number generator ---------------------------------------- }

{ Wrapper for pascal's Random function, as a TMultiArray. }
function Random(Shape: array of longint): TMultiArray; overload;

{ Gaussian random array generator. }
function RandG(mean, stddev: float; Shape: array of longint): TMultiArray; overload;

{ Reduce TMultiArray along a specified axis with specified ReduceFunc. This is
  for convenience rather than performance. }
function ReduceAlongAxis(A: TMultiArray; ReduceFunc: TReduceFunc; axis: integer;
     KeepDims: Boolean = False): TMultiArray;

{ Perform pseudoinverse over a matrix A }
function PseudoInverse(A: TMultiArray): TMultiArray;

{ Perform singular value decomposition (SVD) over a matrix A }
function SVD(A: TMultiArray; SigmasAsMatrix: Boolean=False): TSVDResult;


implementation

function _Cos(X: single; params: array of single): single;
begin
  Exit(System.Cos(X));
end;

function Cos(A: TMultiArray): TMultiArray;
begin
  Exit(ApplyUFunc(A, @_Cos, []));
end;

function _DegToRad(X: single; params: array of single): single;
begin
  Exit(math.DegToRad(X));
end;

function DegToRad(A: TMultiArray): TMultiArray; overload;
begin
  Exit(ApplyUFunc(A, @_DegToRad, []));
end;

function _Exp(X: single; params: array of single): single;
begin
  Exit(System.Exp(X));
end;

function Exp(A: TMultiArray): TMultiArray;
begin
  Exit(ApplyUFunc(A, @_Exp, []));
end;

function _Ln(X: single; params: array of single): single;
begin
  Exit(System.Ln(X));
end;

function Ln(A: TMultiArray): TMultiArray;
begin
  Exit(ApplyUFunc(A, @_Ln, []));
end;

function _Sin(X: single; params: array of single): single;
begin
  Exit(System.Sin(X));
end;

function Sin(A: TMultiArray): TMultiArray;
begin
  Exit(ApplyUFunc(A, @_Sin, []));
end;

function _Random(Params: array of single): single;
begin
  Exit(System.Random);
end;

function Random(Shape: array of longint): TMultiArray;
begin
  Exit(GenerateMultiArray(Shape, @_Random, []));
end;

function _RandG(Params: array of single): single;
begin
  Exit(math.RandG(Params[0], Params[1]));
end;

function RandG(mean, stddev: float; Shape: array of longint): TMultiArray; overload;
begin
  Exit(GenerateMultiArray(Shape, @_RandG, [mean, stddev]));
end;

function ReduceAlongAxis(A: TMultiArray; ReduceFunc: TReduceFunc; axis: integer;
     KeepDims: Boolean = False): TMultiArray;
var
  i: longint;
  IterIndices: array of TLongVector;
begin
  SetLength(IterIndices, Length(A.Indices));
  for i := 0 to High(IterIndices) do
    IterIndices[i] := []; // slicing by [_ALL_, _ALL_, ..., _ALL_]

  IterIndices[axis] := [0];
  Result := A.Slice(IterIndices);  // Slice([_ALL_, ..., [0], ..., _ALL_])

  for i := 1 to A.Shape[axis] - 1 do
  begin
    IterIndices[axis] := [i];
    Result := ReduceFunc(Result, A.Slice(IterIndices));//Result + A.Slice(IterIndices);
  end;

  if Not KeepDims then
    SqueezeMultiArrayAt(Result, axis);
end;

function _Abs(X: single; params: array of single): single;
begin
  Exit(System.Abs(X));
end;

function _Const(Params: array of single): single;
begin
  Exit(Params[0]);
end;

function FullMultiArray(Shape: TLongVector; val: single): TMultiArray;
begin
  Exit(GenerateMultiArray(Shape, @_Const, [val]));
end;

function Ones(Shape: TLongVector): TMultiArray;
begin
  Exit(GenerateMultiArray(Shape, @_Const, [1]));
end;

function _Zero(Params: array of single): single;
begin
  Exit(0);
end;

function Zeros(Shape: TLongVector): TMultiArray;
begin
  Exit(GenerateMultiArray(Shape, @_Zero, []));
end;

function Abs(A: TMultiArray): TMultiArray;
begin
  Exit(ApplyUFunc(A, @_Abs, []));
end;

function ArgMax(const x: array of Single): Integer;
var
  Index: Integer;
  MaxValue: Single;
begin
  Result := -1;
  MaxValue := -MaxSingle;
  for Index := 0 to high(x) do begin
    if x[Index] > MaxValue then begin
      Result := Index;
      MaxValue := x[Index];
    end;
  end;
end;

function ArgMax(A: TMultiArray; axis: integer = -1; KeepDims: boolean=False): TMultiArray;
var
  i: longint;
begin
  if axis = -1 then
    Exit(ArgMax(A.GetVirtualData));
  for i := 0 to A.Shape[axis] - 1 do
  begin

  end;
end;

function Max(A: TMultiArray; axis: integer = -1; KeepDims: boolean=False): TMultiArray;
begin
  if axis = -1 then
    Exit(math.MaxValue(A.GetVirtualData));
  //if KeepDims then
  //  Exit();
  Exit(ReduceAlongAxis(A, @Maximum, axis, KeepDims));
end;

function Mean(A: TMultiArray; axis: integer = -1; KeepDims: boolean=False): TMultiArray;
begin
  if axis = -1 then
    Exit(math.Mean(A.GetVirtualData));
  Result := Sum(A, axis) / A.Shape[axis];
  if KeepDims then
    SqueezeMultiArrayAt(Result, axis);
end;

function _Round(X: single; params: array of single): single;
begin
  Exit(System.round(X));
end;

function Round(A: TMultiArray): TMultiArray;
begin
  Exit(ApplyUFunc(A, @_Round, []));
end;

function Sum(A: TMultiArray; axis: integer = -1; KeepDims: boolean=False): TMultiArray;
var
  outdata: TSingleVector;
  i, j: longint;
begin
  { Specific case for vector (NDims=1) or summation over _ALL_ elements }
  if (axis = -1) or (A.NDims = 1) then
    Exit(math.Sum(A.GetVirtualData));

  { Specific case for matrix }
  if A.NDims = 2 then
  begin
    if axis = 0 then { Case 1: column-wise sum }
    begin
      SetLength(outdata, A.Shape[1]);
      for i := 0 to A.Shape[1] - 1 do
      begin
        outdata[i] := 0;
        for j := 0 to A.Shape[0] - 1 do
        begin
          outdata[i] := outdata[i] + A.Get(IndexToStridedOffset([j, i], A.Strides));
        end;
      end;
      if KeepDims then
        Exit(CreateMultiArray(outdata).Reshape([1, A.Shape[1]]))
      else
        Exit(CreateMultiArray(outdata));
    end
    else if axis = 1 then { Case 2: row-wise sum }
    begin
      SetLength(outdata, A.Shape[0]);
      for i := 0 to A.Shape[0] - 1 do
      begin
        outdata[i] := 0;
        for j := 0 to A.Shape[1] - 1 do
        begin
          outdata[i] := outdata[i] + A.Get(IndexToStridedOffset([i, j], A.Strides));
        end;
      end;
      if KeepDims then
        Exit(CreateMultiArray(outdata).Reshape([A.Shape[0], 1]))
      else
        Exit(CreateMultiArray(outdata));
    end;
  end;

  { Sum along axis for arbitrary dimension }
  Exit(ReduceAlongAxis(A, @Add, axis, KeepDims));
end;

function PseudoInverse(A: TMultiArray): TMultiArray;
var
  SVDRes: TSVDResult;
  i : integer;
begin
  SVDRes := SVD(A, True);
  for i := 0 to SVDRes.Sigma.Size - 1 do
    if SVDRes.Sigma.Get(i) > 0 then
       SVDRes.Sigma.Put(OffsetToIndex(SVDRes.Sigma, i), 1 / SVDRes.Sigma.Get(i));
  Exit(SVDRes.VT.T.Matmul(SVDRes.Sigma.T).Matmul(SVDRes.U.T));
end;

function SVD(A: TMultiArray; SigmasAsMatrix: Boolean=False): TSVDResult;
var
  i, m, n: integer;
  U, Sigma, VT: TMultiArray;
  Superb, SigmaVal: TSingleVector;
begin
  EnsureNDims(A, 2);

  m := A.Shape[0];
  n := A.Shape[1];
  SetLength(Superb, min(m, n) - 1);
  SetLength(SigmaVal, min(m, n));

  U := AllocateMultiArray(m ** 2).Reshape([m, m]);
  VT := AllocateMultiArray(n ** 2).Reshape([n, n]);

  LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, CopyVector(A.GetVirtualData),
    n, SigmaVal, U.Data, m, VT.Data, n, Superb);

  Result.U := U;
  Result.VT := VT;

  if not SigmasAsMatrix then
    Result.Sigma := SigmaVal
  else
  begin
    Sigma := AllocateMultiArray(m * n).Reshape([m, n]);
    for i := 0 to min(m, n) - 1 do
      Sigma.Put([i, i], SigmaVal[i]);
    Result.Sigma := Sigma;
  end;

  superb := nil;
end;

end.

